"""
Export Module for AstroLens

Exports analysis results in multiple formats:
- CSV: For spreadsheets and data analysis
- JSON: For programmatic access
- HTML: For shareable reports
- VOTable: For astronomical tools (TOPCAT, Aladin)

Usage:
    from features.export import ResultsExporter
    
    exporter = ResultsExporter(api_base="http://localhost:8000")
    exporter.export_csv("results.csv")
    exporter.export_html("report.html")
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import httpx

logger = logging.getLogger(__name__)

# Default export directory
EXPORT_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class ResultsExporter:
    """Export analysis results in multiple formats."""
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.client = httpx.Client(base_url=api_base, timeout=30.0)
    
    def _fetch_anomalies(self, limit: int = 5000) -> List[dict]:
        """Fetch all anomaly candidates from API."""
        try:
            resp = self.client.get("/candidates", params={"limit": limit})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch anomalies: {e}")
            return []
    
    def _fetch_all_images(self, limit: int = 10000) -> List[dict]:
        """Fetch all images from API."""
        try:
            resp = self.client.get("/images", params={"limit": limit})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch images: {e}")
            return []
    
    def _fetch_crossref_results(self) -> dict:
        """Fetch cross-reference results from file."""
        results_file = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "data" / "cross_reference_results.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return {"results": []}
    
    def export_csv(self, output_path: Optional[str] = None, anomalies_only: bool = True) -> str:
        """
        Export results as CSV.
        
        Args:
            output_path: Output file path (auto-generated if None)
            anomalies_only: Whether to export only anomalies
        
        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(EXPORT_DIR / f"astrolens_results_{timestamp}.csv")
        
        images = self._fetch_anomalies() if anomalies_only else self._fetch_all_images()
        
        if not images:
            logger.warning("No data to export")
            return output_path
        
        fieldnames = [
            "id", "filename", "filepath", "file_type",
            "class_label", "class_confidence", "ood_score", "is_anomaly",
            "width", "height", "source", "created_at", "analyzed_at",
        ]
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for img in images:
                writer.writerow(img)
        
        logger.info(f"Exported {len(images)} records to {output_path}")
        return output_path
    
    def export_json(self, output_path: Optional[str] = None, include_crossref: bool = True) -> str:
        """
        Export results as JSON.
        
        Args:
            output_path: Output file path
            include_crossref: Include cross-reference results
        
        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(EXPORT_DIR / f"astrolens_results_{timestamp}.json")
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "anomalies": self._fetch_anomalies(),
            "stats": {},
        }
        
        # Fetch stats
        try:
            resp = self.client.get("/stats")
            if resp.status_code == 200:
                data["stats"] = resp.json()
        except Exception:
            pass
        
        if include_crossref:
            xref = self._fetch_crossref_results()
            data["cross_reference"] = xref.get("results", [])
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported JSON to {output_path}")
        return output_path
    
    def export_html(self, output_path: Optional[str] = None) -> str:
        """
        Export results as a shareable HTML report.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(EXPORT_DIR / f"astrolens_report_{timestamp}.html")
        
        anomalies = self._fetch_anomalies()
        xref = self._fetch_crossref_results()
        xref_results = xref.get("results", [])
        
        # Build stats
        total = len(anomalies)
        known = sum(1 for r in xref_results if r.get("is_known"))
        unknown = sum(1 for r in xref_results if not r.get("is_known"))
        
        # Generate HTML
        rows_html = ""
        for i, a in enumerate(anomalies[:200]):  # Limit to 200 for performance
            xr = next((r for r in xref_results if r.get("image_id") == a.get("id")), {})
            status = "Known" if xr.get("is_known") else "Unknown" if xr else "Not checked"
            match_name = ""
            if xr.get("primary_match"):
                match_name = xr["primary_match"].get("object_name", "")
            
            rows_html += f"""
            <tr>
                <td>{a.get('id', '')}</td>
                <td>{a.get('filename', '')}</td>
                <td>{a.get('class_label', 'N/A')}</td>
                <td>{a.get('ood_score', 0):.3f}</td>
                <td class="{'known' if xr.get('is_known') else 'unknown'}">{status}</td>
                <td>{match_name}</td>
            </tr>"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AstroLens Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0a0d12; color: #c8d0e0;
            padding: 40px; max-width: 1200px; margin: 0 auto;
        }}
        h1 {{ font-size: 28px; font-weight: 300; margin-bottom: 8px; }}
        .subtitle {{ color: #7a8599; font-size: 14px; margin-bottom: 32px; }}
        .stats {{
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px;
        }}
        .stat-card {{
            background: rgba(20, 24, 32, 0.8); border: 1px solid rgba(60, 80, 120, 0.3);
            border-radius: 12px; padding: 20px;
        }}
        .stat-value {{ font-size: 32px; font-weight: 300; color: #5b8def; }}
        .stat-label {{ font-size: 11px; color: #4a5568; text-transform: uppercase; letter-spacing: 1px; }}
        table {{
            width: 100%; border-collapse: collapse; background: rgba(15, 20, 28, 0.5);
            border-radius: 12px; overflow: hidden;
        }}
        th {{
            background: rgba(20, 27, 38, 0.8); padding: 12px 16px; text-align: left;
            font-size: 11px; color: #7a8599; text-transform: uppercase; letter-spacing: 1px;
        }}
        td {{ padding: 10px 16px; border-bottom: 1px solid rgba(30, 40, 55, 0.3); font-size: 13px; }}
        tr:hover {{ background: rgba(30, 40, 55, 0.3); }}
        .known {{ color: #7a8599; }}
        .unknown {{ color: #34d399; font-weight: 500; }}
        .footer {{ margin-top: 32px; color: #4a5568; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>AstroLens Analysis Report</h1>
    <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Anomalies</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{known}</div>
            <div class="stat-label">Known Objects</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #34d399;">{unknown}</div>
            <div class="stat-label">Unknown (Potential Discoveries)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(xref_results)}</div>
            <div class="stat-label">Cross-Referenced</div>
        </div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>ID</th><th>Filename</th><th>Class</th>
                <th>OOD Score</th><th>Status</th><th>Catalog Match</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    
    <div class="footer">
        <p>AstroLens - Galaxy Anomaly Discovery System</p>
        <p>Results based on ViT + OOD ensemble detection (MSP, Energy, Mahalanobis)</p>
    </div>
</body>
</html>"""
        
        Path(output_path).write_text(html)
        logger.info(f"Exported HTML report to {output_path}")
        return output_path
    
    def export_votable(self, output_path: Optional[str] = None) -> str:
        """
        Export as VOTable (XML format for astronomical tools).
        
        Compatible with: TOPCAT, Aladin, DS9, etc.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(EXPORT_DIR / f"astrolens_results_{timestamp}.vot")
        
        anomalies = self._fetch_anomalies()
        
        # Build VOTable XML
        rows_xml = ""
        for a in anomalies:
            filename = a.get("filename", "")
            # Try to extract RA/Dec from filename
            ra, dec = self._extract_coords(filename)
            rows_xml += f"""      <TR>
        <TD>{a.get('id', '')}</TD>
        <TD>{filename}</TD>
        <TD>{ra}</TD>
        <TD>{dec}</TD>
        <TD>{a.get('class_label', '')}</TD>
        <TD>{a.get('ood_score', 0):.4f}</TD>
        <TD>{a.get('is_anomaly', False)}</TD>
      </TR>\n"""
        
        votable = f"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.4">
  <RESOURCE name="AstroLens Results">
    <DESCRIPTION>Galaxy anomaly detection results from AstroLens</DESCRIPTION>
    <TABLE name="anomalies">
      <FIELD name="id" datatype="int" ucd="meta.id"/>
      <FIELD name="filename" datatype="char" arraysize="*" ucd="meta.file"/>
      <FIELD name="ra" datatype="double" unit="deg" ucd="pos.eq.ra"/>
      <FIELD name="dec" datatype="double" unit="deg" ucd="pos.eq.dec"/>
      <FIELD name="class_label" datatype="char" arraysize="*" ucd="meta.code.class"/>
      <FIELD name="ood_score" datatype="double" ucd="stat.value"/>
      <FIELD name="is_anomaly" datatype="boolean" ucd="meta.code"/>
      <DATA>
        <TABLEDATA>
{rows_xml}        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>"""
        
        Path(output_path).write_text(votable)
        logger.info(f"Exported VOTable to {output_path}")
        return output_path
    
    def _extract_coords(self, filename: str) -> tuple:
        """Extract RA/Dec from filename if possible."""
        import re
        ra_match = re.search(r'ra([\d.]+)', filename)
        dec_match = re.search(r'dec(-?[\d.]+)', filename)
        ra = float(ra_match.group(1)) if ra_match else 0.0
        dec = float(dec_match.group(1)) if dec_match else 0.0
        return ra, dec
    
    def close(self):
        self.client.close()
