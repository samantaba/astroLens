"""
AstroLens Web Interface

Browser-based UI for Galaxy and Transient analysis.
Proxies requests to the main API backend.

Usage:
    python -m web.app          # Start web UI on port 8080
    python -m web.app --port 9000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Form, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import httpx
import uvicorn

logger = logging.getLogger(__name__)

# API backend URL
API_BASE = os.environ.get("ASTROLENS_API", "http://localhost:8000")

# Setup app
app = FastAPI(title="AstroLens Web", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def api_client() -> httpx.Client:
    """Get API client."""
    return httpx.Client(base_url=API_BASE, timeout=30.0)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home / Dashboard page."""
    stats = {}
    anomalies = []
    device_info = {}
    
    try:
        with api_client() as client:
            r = client.get("/stats")
            if r.status_code == 200:
                stats = r.json()
            
            r = client.get("/candidates", params={"limit": 20})
            if r.status_code == 200:
                anomalies = r.json()
            
            r = client.get("/health")
            if r.status_code == 200:
                device_info = r.json()
    except Exception as e:
        logger.error(f"API error: {e}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "anomalies": anomalies,
        "device_info": device_info,
        "mode": "galaxy",
    })


@app.get("/galaxy", response_class=HTMLResponse)
async def galaxy_mode(request: Request, page: int = 0, limit: int = 50):
    """Galaxy Mode - Browse and analyze galaxy images."""
    images = []
    stats = {}
    
    try:
        with api_client() as client:
            r = client.get("/images", params={"skip": page * limit, "limit": limit})
            if r.status_code == 200:
                images = r.json()
            
            r = client.get("/stats")
            if r.status_code == 200:
                stats = r.json()
    except Exception as e:
        logger.error(f"API error: {e}")
    
    return templates.TemplateResponse("galaxy.html", {
        "request": request,
        "images": images,
        "stats": stats,
        "page": page,
        "limit": limit,
        "mode": "galaxy",
    })


@app.get("/transient", response_class=HTMLResponse)
async def transient_mode(request: Request):
    """Transient Mode - YOLO-based transient detection."""
    pipeline_state = {}
    state_file = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "pipeline_state.json"
    
    if state_file.exists():
        try:
            with open(state_file) as f:
                pipeline_state = json.load(f)
        except Exception:
            pass
    
    return templates.TemplateResponse("transient.html", {
        "request": request,
        "pipeline_state": pipeline_state,
        "mode": "transient",
    })


@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    """Verification page - Cross-reference anomalies."""
    anomalies = []
    xref_results = []
    
    try:
        with api_client() as client:
            r = client.get("/candidates", params={"limit": 100})
            if r.status_code == 200:
                anomalies = r.json()
    except Exception as e:
        logger.error(f"API error: {e}")
    
    # Load cross-ref results
    xref_file = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "data" / "cross_reference_results.json"
    if xref_file.exists():
        try:
            with open(xref_file) as f:
                data = json.load(f)
                xref_results = data.get("results", [])
        except Exception:
            pass
    
    return templates.TemplateResponse("verify.html", {
        "request": request,
        "anomalies": anomalies,
        "xref_results": xref_results,
        "mode": "verify",
    })


@app.get("/export", response_class=HTMLResponse)
async def export_page(request: Request):
    """Export results page."""
    return templates.TemplateResponse("export.html", {
        "request": request,
        "mode": "export",
    })


# ─────────────────────────────────────────────────────────────────────────────
# API Proxy Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def proxy_stats():
    """Proxy stats from backend."""
    try:
        with api_client() as client:
            r = client.get("/stats")
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/anomalies")
async def proxy_anomalies(limit: int = 100):
    """Proxy anomalies from backend."""
    try:
        with api_client() as client:
            r = client.get("/candidates", params={"limit": limit})
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/images/{image_id}/file")
async def proxy_image_file(image_id: int):
    """Proxy image file from backend."""
    try:
        with api_client() as client:
            r = client.get(f"/images/{image_id}/file")
            from fastapi.responses import Response
            return Response(
                content=r.content,
                media_type=r.headers.get("content-type", "image/jpeg"),
            )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/analyze/{image_id}")
async def proxy_analyze(image_id: int):
    """Proxy full analysis from backend."""
    try:
        with api_client() as client:
            r = client.post(f"/analysis/full/{image_id}")
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/crossref/{image_id}")
async def proxy_crossref(image_id: int):
    """Proxy cross-reference from backend."""
    try:
        with api_client() as client:
            r = client.post(f"/crossref/{image_id}")
            return r.json()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/export/{format}")
async def do_export(format: str):
    """Export results in the specified format."""
    try:
        from features.export import ResultsExporter
        exporter = ResultsExporter(api_base=API_BASE)
        
        if format == "csv":
            path = exporter.export_csv()
        elif format == "json":
            path = exporter.export_json()
        elif format == "html":
            path = exporter.export_html()
        elif format == "votable":
            path = exporter.export_votable()
        else:
            return {"error": f"Unknown format: {format}"}
        
        exporter.close()
        return {"ok": True, "path": path, "format": format}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/device")
async def device_info():
    """Get GPU/device information."""
    try:
        from inference.gpu_utils import DeviceInfo
        info = DeviceInfo.detect()
        return info.to_dict()
    except Exception as e:
        return {"device_type": "unknown", "error": str(e)}


@app.get("/api/pipeline-state")
async def pipeline_state():
    """Get transient pipeline state."""
    state_file = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "pipeline_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"status": "not_started"}


def main():
    parser = argparse.ArgumentParser(description="AstroLens Web Interface")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()
    
    print(f"\n  AstroLens Web Interface")
    print(f"  http://localhost:{args.port}")
    print(f"  API Backend: {API_BASE}\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
