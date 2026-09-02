"""FastAPI Central ICCC Hub Server Application.

Exposes RESTful endpoints for telemetry ingestion, geospatial defect deduplication,
repair workflow state machines, ANPR police watchlist enforcement, and dashboard rendering.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from central_iccc_hub.db import (
    db_session,
    get_anpr_alert_by_plate,
    get_defect_by_id,
    init_db,
    list_anpr_alerts,
    list_defects,
    update_anpr_alert_status,
    update_defect_status,
    upsert_anpr_alert,
    upsert_defect,
)

# --- Structured Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("central_iccc_hub")


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context managing database schema setup and application teardown."""
    logger.info("Initializing ICCC Hub database schema...")
    init_db()
    logger.info("Database schema initialized successfully.")
    yield
    logger.info("Shutting down ICCC Hub server.")


app = FastAPI(
    title="Central ICCC Hub Server",
    version="1.0.0",
    description="Integrated Command and Control Center API Hub",
    lifespan=lifespan,
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic v2 Data Transfer Models ---
class TelemetryPayload(BaseModel):
    bus_id: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lng: float = Field(..., ge=-180.0, le=180.0)
    defect_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    z_axis_g: float = Field(default=1.0)


class ANPRReportPayload(BaseModel):
    bus_id: str
    plate_number: str
    lat: float
    lng: float


class FlagVehiclePayload(BaseModel):
    plate_number: str
    incident_type: str


# --- Geospatial Utility ---
def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two geographic coordinates in meters."""
    r = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


# --- API Endpoint Handlers ---

@app.post("/api/v1/telemetry", status_code=status.HTTP_200_OK)
def ingest_telemetry(payload: TelemetryPayload) -> Dict[str, Any]:
    """Ingests road telemetry data, performing geospatial deduplication within a 5m radius."""
    now_iso = datetime.now(timezone.utc).isoformat()

    with db_session() as conn:
        cursor = conn.execute("SELECT * FROM defects WHERE status != 'VERIFIED_FIXED'")
        active_defects = [dict(row) for row in cursor.fetchall()]

        matching_defect = None
        min_dist = float("inf")

        for defect in active_defects:
            if defect["defect_type"].lower() == payload.defect_type.lower():
                dist = calculate_haversine_distance(
                    payload.lat, payload.lng, defect["lat"], defect["lng"]
                )
                if dist <= 5.0 and dist < min_dist:
                    min_dist = dist
                    matching_defect = defect

        if matching_defect:
            prev_status = matching_defect["status"]
            new_status = "SLA_BREACH" if prev_status == "AUDIT_PENDING" else prev_status
            new_count = matching_defect["detection_count"] + 1

            conn.execute(
                """
                UPDATE defects
                SET detection_count = ?,
                    last_seen_at = ?,
                    status = ?
                WHERE id = ?
                """,
                (new_count, now_iso, new_status, matching_defect["id"]),
            )

            logger.info(
                f"Deduplicated defect {matching_defect['id']} at {min_dist:.2f}m radius. "
                f"Status: {prev_status} -> {new_status}"
            )
            return {
                "status": "deduplicated",
                "defect_id": matching_defect["id"],
                "detection_count": new_count,
                "defect_status": new_status,
                "distance_m": round(min_dist, 2),
            }

        # Calculate fused severity score
        severity_score = round(
            (payload.confidence * 5.0) + (min(payload.z_axis_g, 2.5) * 2.0), 1
        )

        # Generate sequential unique defect ID
        cursor = conn.execute("SELECT id FROM defects WHERE id LIKE 'BEL-DEL-%'")
        existing_ids = cursor.fetchall()
        seq_nums = []
        for row in existing_ids:
            try:
                seq_nums.append(int(row["id"].split("-")[-1]))
            except (ValueError, IndexError):
                pass
        next_seq = max(seq_nums, default=1000) + 1
        defect_id = f"BEL-DEL-{next_seq}"

        defect_record = {
            "id": defect_id,
            "bus_id": payload.bus_id,
            "lat": payload.lat,
            "lng": payload.lng,
            "defect_type": payload.defect_type,
            "confidence": payload.confidence,
            "z_axis_g": payload.z_axis_g,
            "severity_score": severity_score,
            "status": "OPEN",
            "detection_count": 1,
            "created_at": now_iso,
            "last_seen_at": now_iso,
        }
        upsert_defect(conn, defect_record)

        logger.info(f"Registered new defect {defect_id} (Severity: {severity_score})")
        return {"status": "created", "defect": defect_record}


@app.post("/api/v1/defects/{defect_id}/mark_repaired", status_code=status.HTTP_200_OK)
def mark_defect_repaired(defect_id: str) -> Dict[str, Any]:
    """Transitions a defect status to AUDIT_PENDING."""
    with db_session() as conn:
        existing = get_defect_by_id(conn, defect_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Defect record '{defect_id}' not found.",
            )
        update_defect_status(conn, defect_id, "AUDIT_PENDING")
        return {"status": "success", "defect_id": defect_id, "new_status": "AUDIT_PENDING"}


@app.post("/api/v1/defects/{defect_id}/verify_manual", status_code=status.HTTP_200_OK)
def verify_defect_manual(defect_id: str) -> Dict[str, Any]:
    """Transitions a defect status to VERIFIED_FIXED."""
    with db_session() as conn:
        existing = get_defect_by_id(conn, defect_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Defect record '{defect_id}' not found.",
            )
        update_defect_status(conn, defect_id, "VERIFIED_FIXED")
        return {"status": "success", "defect_id": defect_id, "new_status": "VERIFIED_FIXED"}


@app.get("/api/v1/defects", status_code=status.HTTP_200_OK)
def get_defects() -> Dict[str, Any]:
    """Retrieves all defects sorted by most recent sighting."""
    with db_session() as conn:
        cursor = conn.execute("SELECT * FROM defects ORDER BY last_seen_at DESC")
        defects = [dict(row) for row in cursor.fetchall()]
        return {"total": len(defects), "defects": defects}


@app.get("/api/v1/defects/{defect_id}/export_work_order")
def export_work_order_pdf(defect_id: str) -> StreamingResponse:
    """Generates an official PDF Work Order document for a specific defect record."""
    with db_session() as conn:
        defect = get_defect_by_id(conn, defect_id)
        if not defect:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Defect record '{defect_id}' not found.",
            )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "HeaderSubtitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e40af"),
        alignment=1,
    )
    normal_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    clause_style = ParagraphStyle(
        "ClauseText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.0,
        leading=13,
        textColor=colors.HexColor("#991b1b"),
    )

    story = []

    story.append(Paragraph("GOVERNMENT OF NCT OF DELHI - PUBLIC WORKS DEPARTMENT", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("MARGDRISHTI AUTOMATED SLA REPAIR WORK ORDER", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e40af"), spaceAfter=15))

    wo_number = f"WO-BEL-{defect['id']}"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    maps_url = f"https://www.google.com/maps?q={defect['lat']},{defect['lng']}"
    gps_str = f"({defect['lat']}, {defect['lng']})"

    table_data = [
        [
            Paragraph("<b>Work Order No:</b>", normal_style),
            Paragraph(wo_number, normal_style),
            Paragraph("<b>Date of Issue:</b>", normal_style),
            Paragraph(date_str, normal_style),
        ],
        [
            Paragraph("<b>Defect Identifier:</b>", normal_style),
            Paragraph(defect["id"], normal_style),
            Paragraph("<b>Current Status:</b>", normal_style),
            Paragraph(defect["status"], normal_style),
        ],
        [
            Paragraph("<b>Bus Node Source:</b>", normal_style),
            Paragraph(defect["bus_id"], normal_style),
            Paragraph("<b>Defect Classification:</b>", normal_style),
            Paragraph(defect["defect_type"].upper(), normal_style),
        ],
        [
            Paragraph("<b>GPS Coordinates:</b>", normal_style),
            Paragraph(f'<a href="{maps_url}" color="blue"><u>{gps_str}</u></a>', normal_style),
            Paragraph("<b>Deduplicated Passes:</b>", normal_style),
            Paragraph(f"{defect['detection_count']} passes", normal_style),
        ],
        [
            Paragraph("<b>Severity Index:</b>", normal_style),
            Paragraph(f"<b>{defect['severity_score']} / 10.0</b>", normal_style),
            Paragraph("<b>Vibro-Vision Shock:</b>", normal_style),
            Paragraph(f"<b>{defect['z_axis_g']} G</b>", normal_style),
        ],
    ]

    meta_table = Table(table_data, colWidths=[110, 150, 110, 150])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 15))

    clause_text = (
        "<b>Notice to Contractor:</b> This defect was detected via autonomous public transit spatial telemetry. "
        "Repair verification is automated via follow-up bus passes. Failure to restore surface within 72 hours "
        "invokes liquidated damages under Clause 14-B."
    )
    clause_table = Table([[Paragraph(clause_text, clause_style)]], colWidths=[520])
    clause_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
            ("PADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(clause_table)
    story.append(Spacer(1, 25))

    sig_data = [
        [
            Paragraph(
                "<b>AUTOMATED SYSTEM STAMP:</b><br/><font color='#166534'>[DIGITALLY SIGNED & VERIFIED BY MARGDRISHTI ICCC ENGINE]</font>",
                normal_style,
            ),
            Paragraph(
                "<b>EXECUTIVE ENGINEER</b><br/>Public Works Department<br/>Govt. of NCT of Delhi",
                normal_style,
            ),
        ]
    ]
    sig_table = Table(sig_data, colWidths=[260, 260])
    sig_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(sig_table)

    doc.build(story)
    buffer.seek(0)

    headers = {
        "Content-Disposition": f'inline; filename="Work_Order_{defect_id}.pdf"'
    }
    return StreamingResponse(buffer, media_type="application/pdf", headers=headers)


@app.post("/api/v1/police/flag_vehicle", status_code=status.HTTP_200_OK)
def flag_vehicle(payload: FlagVehiclePayload) -> Dict[str, Any]:
    """Flags a vehicle plate number in the ANPR watchlist with status SEARCHING."""
    with db_session() as conn:
        alert_data = {
            "plate_number": payload.plate_number,
            "incident_type": payload.incident_type,
            "status": "SEARCHING",
        }
        upsert_anpr_alert(conn, alert_data)
        logger.info(f"Flagged vehicle plate {payload.plate_number} ({payload.incident_type})")
        return {"status": "success", "plate_number": payload.plate_number, "alert_status": "SEARCHING"}


@app.post("/api/v1/anpr/sighting", status_code=status.HTTP_200_OK)
def report_anpr_sighting(payload: ANPRReportPayload) -> Dict[str, Any]:
    """Processes ANPR camera sightings, intercepting flagged watchlist vehicles."""
    now_iso = datetime.now(timezone.utc).isoformat()

    with db_session() as conn:
        alert = get_anpr_alert_by_plate(conn, payload.plate_number)
        if not alert:
            return {
                "status": "NO_MATCH",
                "message": f"Plate '{payload.plate_number}' is not currently flagged.",
            }

        alert_update = {
            "plate_number": payload.plate_number,
            "incident_type": alert["incident_type"],
            "status": "INTERCEPTED",
            "last_lat": payload.lat,
            "last_lng": payload.lng,
            "last_seen_by": payload.bus_id,
            "last_seen_time": now_iso,
        }
        upsert_anpr_alert(conn, alert_update)

        logger.warning(
            f"ANPR INTERCEPT ALERT: Plate {payload.plate_number} spotted by bus {payload.bus_id} "
            f"at ({payload.lat}, {payload.lng})"
        )
        return {
            "status": "INTERCEPTED",
            "plate_number": payload.plate_number,
            "alert": alert_update,
        }


@app.get("/api/v1/police/alerts", status_code=status.HTTP_200_OK)
def get_police_alerts() -> Dict[str, Any]:
    """Retrieves all registered police ANPR alerts."""
    with db_session() as conn:
        alerts = list_anpr_alerts(conn)
        return {"total": len(alerts), "alerts": alerts}


def generate_video_stream():
    """Generates continuous MJPEG video stream with dynamic HUD OSD overlay."""
    video_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bus_edge_node",
        "road_sample.mp4",
    )
    cap = None
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)

    frame_counter = 0

    try:
        while True:
            frame = None

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if frame is not None:
                    frame = cv2.resize(frame, (320, 240))

            if frame is None:
                frame_counter += 1
                frame = np.full((240, 320, 3), (30, 30, 35), dtype=np.uint8)

                # Draw moving perspective lane markings
                offset = (frame_counter * 8) % 40
                for y in range(50 + offset, 240, 40):
                    cv2.line(frame, (160, y), (160, y + 20), (180, 180, 180), 2)

                # Animated bounding box
                box_y = 140 + int(math.sin(frame_counter * 0.2) * 10)
                cv2.rectangle(frame, (130, box_y), (190, box_y + 30), (255, 229, 0), 2)
                cv2.putText(
                    frame,
                    "POTHOLE (CONF: 0.94)",
                    (110, box_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 229, 0),
                    1,
                )
            else:
                frame_counter += 1
                box_y = 140 + int(math.sin(frame_counter * 0.2) * 10)
                cv2.rectangle(frame, (130, box_y), (190, box_y + 30), (255, 229, 0), 2)
                cv2.putText(
                    frame,
                    "POTHOLE (CONF: 0.94)",
                    (110, box_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 229, 0),
                    1,
                )

            # Header HUD text (Emerald #10b981 -> BGR: (129, 185, 16))
            time_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            osd_header = f"CAM-01 // BUS-104 | {time_str}"
            cv2.putText(
                frame,
                osd_header,
                (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (129, 185, 16),
                1,
            )

            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.06)
    finally:
        if cap is not None:
            cap.release()


@app.get("/api/v1/video/stream")
def get_video_stream() -> StreamingResponse:
    """Streams live MJPEG camera feed with real-time HUD OSD overlays."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    """Serves the central hub web dashboard interface."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if not os.path.exists(template_path):
        logger.error(f"Dashboard template not found at '{template_path}'.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dashboard template missing.",
        )

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as exc:
        logger.error(f"Failed to read dashboard template: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read dashboard template.",
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("central_iccc_hub.app:app", host="0.0.0.0", port=8000, reload=True)
