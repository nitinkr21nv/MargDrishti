"""FastAPI Central ICCC Hub Server Application.

Exposes RESTful endpoints for telemetry ingestion, geospatial defect deduplication,
repair workflow state machines, ANPR police watchlist enforcement, and dashboard rendering.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import math
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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
