#!/usr/bin/env python3
"""
AstroLens Streaming Report Generator

Generates HTML reports with embedded charts for daily and final summaries.
Uses matplotlib for chart generation, embedded as base64 in self-contained HTML.

Reports include:
- Anomaly rate trend over days
- Source effectiveness comparison
- Threshold evolution
- Top candidate gallery
- Self-correction log
- Model improvement tracking
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _create_anomaly_trend_chart(snapshots: List[dict]) -> str:
    """Create anomaly rate trend chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        rates = [s.get("anomaly_rate", 0) for s in snapshots]
        anomalies = [s.get("anomalies_found", 0) for s in snapshots]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                        facecolor="#0d1117")

        # Anomaly rate
        ax1.set_facecolor("#161b22")
        ax1.plot(days, rates, "o-", color="#58a6ff", linewidth=2, markersize=6)
        ax1.fill_between(days, rates, alpha=0.1, color="#58a6ff")
        ax1.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax1.set_ylabel("Anomaly Rate (%)", color="#8b949e", fontsize=11)
        ax1.set_title("Anomaly Detection Rate", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax1.tick_params(colors="#8b949e")
        ax1.spines["bottom"].set_color("#30363d")
        ax1.spines["left"].set_color("#30363d")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, alpha=0.15, color="#30363d")

        # Cumulative anomalies
        cumulative = []
        total = 0
        for a in anomalies:
            total += a
            cumulative.append(total)

        ax2.set_facecolor("#161b22")
        ax2.bar(days, anomalies, color="#238636", alpha=0.7, label="Daily")
        ax2.plot(days, cumulative, "o-", color="#f0883e", linewidth=2,
                 markersize=5, label="Cumulative")
        ax2.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax2.set_ylabel("Anomalies", color="#8b949e", fontsize=11)
        ax2.set_title("Anomalies Found", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax2.tick_params(colors="#8b949e")
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, alpha=0.15, color="#30363d")
        ax2.legend(facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=10)

        fig.tight_layout(pad=2)
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


def _create_threshold_chart(snapshots: List[dict]) -> str:
    """Create threshold evolution chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        starts = [s.get("threshold_start", 3.0) for s in snapshots]
        ends = [s.get("threshold_end", 3.0) for s in snapshots]
        highest = [s.get("highest_ood_score", 0) for s in snapshots]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        ax.plot(days, ends, "o-", color="#da3633", linewidth=2,
                markersize=6, label="Threshold")
        ax.plot(days, highest, "s--", color="#f0883e", linewidth=1.5,
                markersize=5, label="Highest OOD Score")

        # Fill area between threshold and highest score
        ax.fill_between(days, ends, highest, alpha=0.1, color="#f0883e")

        ax.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax.set_ylabel("Score", color="#8b949e", fontsize=11)
        ax.set_title("Threshold Evolution vs Highest OOD Score",
                      color="#c9d1d9", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="#30363d")
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=10)

        fig.tight_layout()
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


def _create_source_chart(snapshots: List[dict]) -> str:
    """Create source effectiveness comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Aggregate source data from latest snapshot
        if not snapshots:
            return ""

        source_data = snapshots[-1].get("source_effectiveness", {})
        if not source_data:
            return ""

        sources = []
        analyzed = []
        anomalies = []

        for name, stats in source_data.items():
            if isinstance(stats, dict) and stats.get("total_analyzed", 0) > 0:
                sources.append(name[:12])  # Truncate long names
                analyzed.append(stats.get("total_analyzed", 0))
                anomalies.append(stats.get("anomalies", 0))

        if not sources:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                        facecolor="#0d1117")

        # Images analyzed per source
        colors = ["#58a6ff", "#238636", "#f0883e", "#da3633",
                  "#a371f7", "#3fb950", "#d2a8ff", "#79c0ff"]
        bar_colors = [colors[i % len(colors)] for i in range(len(sources))]

        ax1.set_facecolor("#161b22")
        ax1.barh(sources, analyzed, color=bar_colors, alpha=0.8)
        ax1.set_xlabel("Images Analyzed", color="#8b949e", fontsize=11)
        ax1.set_title("Volume by Source", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax1.tick_params(colors="#8b949e")
        ax1.spines["bottom"].set_color("#30363d")
        ax1.spines["left"].set_color("#30363d")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Anomaly rate per source
        rates = [
            (a / t * 100) if t > 0 else 0
            for a, t in zip(anomalies, analyzed)
        ]

        ax2.set_facecolor("#161b22")
        ax2.barh(sources, rates, color=bar_colors, alpha=0.8)
        ax2.set_xlabel("Anomaly Rate (%)", color="#8b949e", fontsize=11)
        ax2.set_title("Effectiveness by Source", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax2.tick_params(colors="#8b949e")
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig.tight_layout(pad=2)
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


def _create_yolo_chart(snapshots: List[dict]) -> str:
    """Create YOLO transient detection chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        yolo_scanned = [s.get("yolo_images_scanned", 0) for s in snapshots]
        yolo_confirmed = [s.get("yolo_confirmations", 0) for s in snapshots]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                        facecolor="#0d1117")

        # YOLO detections bar chart
        ax1.set_facecolor("#161b22")
        width = 0.35
        x = range(len(days))
        ax1.bar([i - width / 2 for i in x], yolo_scanned, width,
                color="#a371f7", alpha=0.7, label="Scanned")
        ax1.bar([i + width / 2 for i in x], yolo_confirmed, width,
                color="#da3633", alpha=0.8, label="Confirmed")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([f"Day {d}" for d in days])
        ax1.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax1.set_ylabel("Images", color="#8b949e", fontsize=11)
        ax1.set_title("YOLO Transient Detection", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax1.tick_params(colors="#8b949e")
        ax1.spines["bottom"].set_color("#30363d")
        ax1.spines["left"].set_color("#30363d")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, alpha=0.15, color="#30363d")
        ax1.legend(facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=10)

        # Detection rate over time
        rates = [
            (c / s * 100) if s > 0 else 0
            for c, s in zip(yolo_confirmed, yolo_scanned)
        ]
        ax2.set_facecolor("#161b22")
        ax2.plot(days, rates, "o-", color="#da3633", linewidth=2, markersize=6)
        ax2.fill_between(days, rates, alpha=0.1, color="#da3633")
        ax2.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax2.set_ylabel("Detection Rate (%)", color="#8b949e", fontsize=11)
        ax2.set_title("YOLO Detection Rate", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax2.tick_params(colors="#8b949e")
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, alpha=0.15, color="#30363d")

        fig.tight_layout(pad=2)
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


def _create_throughput_chart(snapshots: List[dict]) -> str:
    """Create throughput and images/hour chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [s.get("day", i + 1) for i, s in enumerate(snapshots)]
        analyzed = [s.get("images_analyzed", 0) for s in snapshots]
        iph = [s.get("images_per_hour", 0) for s in snapshots]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                        facecolor="#0d1117")

        ax1.set_facecolor("#161b22")
        ax1.bar(days, analyzed, color="#a371f7", alpha=0.8)
        ax1.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax1.set_ylabel("Images Analyzed", color="#8b949e", fontsize=11)
        ax1.set_title("Daily Throughput", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax1.tick_params(colors="#8b949e")
        ax1.spines["bottom"].set_color("#30363d")
        ax1.spines["left"].set_color("#30363d")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(True, alpha=0.15, color="#30363d")

        ax2.set_facecolor("#161b22")
        ax2.plot(days, iph, "o-", color="#3fb950", linewidth=2, markersize=6)
        ax2.fill_between(days, iph, alpha=0.1, color="#3fb950")
        ax2.set_xlabel("Day", color="#8b949e", fontsize=11)
        ax2.set_ylabel("Images / Hour", color="#8b949e", fontsize=11)
        ax2.set_title("Processing Speed", color="#c9d1d9",
                       fontsize=13, fontweight="bold")
        ax2.tick_params(colors="#8b949e")
        ax2.spines["bottom"].set_color("#30363d")
        ax2.spines["left"].set_color("#30363d")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, alpha=0.15, color="#30363d")

        fig.tight_layout(pad=2)
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64
    except ImportError:
        return ""


# ── HTML Templates ────────────────────────────────────────────────────

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    margin: 0;
    padding: 24px;
    line-height: 1.6;
}
.container { max-width: 1100px; margin: 0 auto; }
h1 { color: #58a6ff; font-size: 28px; margin-bottom: 4px; }
h2 { color: #c9d1d9; font-size: 20px; border-bottom: 1px solid #30363d;
     padding-bottom: 8px; margin-top: 32px; }
h3 { color: #8b949e; font-size: 16px; }
.subtitle { color: #8b949e; font-size: 14px; margin-bottom: 24px; }
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    margin: 12px 0;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
}
.stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
}
.stat-label {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-card.highlight .stat-value { color: #3fb950; }
.stat-card.warning .stat-value { color: #f0883e; }
.stat-card.danger .stat-value { color: #da3633; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
}
th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #21262d;
}
th { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
td { color: #c9d1d9; font-size: 13px; }
tr:hover td { background: rgba(88, 166, 255, 0.04); }
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.tag-anomaly { background: rgba(218, 54, 51, 0.2); color: #da3633; }
.tag-correction { background: rgba(240, 136, 62, 0.2); color: #f0883e; }
.tag-source { background: rgba(88, 166, 255, 0.2); color: #58a6ff; }
.chart-container { margin: 16px 0; text-align: center; }
.chart-container img { max-width: 100%; border-radius: 6px; }
.correction-item {
    padding: 8px 12px;
    margin: 4px 0;
    background: rgba(240, 136, 62, 0.08);
    border-left: 3px solid #f0883e;
    border-radius: 0 4px 4px 0;
    font-size: 13px;
}
.footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #30363d;
    text-align: center;
    color: #484f58;
    font-size: 12px;
}
.footer a { color: #58a6ff; text-decoration: none; }
"""


def generate_daily_report(
    streaming_state: dict,
    output_dir: Path,
) -> Path:
    """Generate a daily HTML report with charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = streaming_state.get("daily_snapshots", [])
    current_day = streaming_state.get("current_day", 0)
    target_days = streaming_state.get("target_days", 7)

    # Latest snapshot
    latest = snapshots[-1] if snapshots else {}

    # Generate charts
    trend_chart = _create_anomaly_trend_chart(snapshots) if len(snapshots) > 0 else ""
    threshold_chart = _create_threshold_chart(snapshots) if len(snapshots) > 0 else ""
    source_chart = _create_source_chart(snapshots)
    throughput_chart = _create_throughput_chart(snapshots) if len(snapshots) > 0 else ""
    yolo_chart = _create_yolo_chart(snapshots) if len(snapshots) > 0 else ""

    # Build candidates table
    candidates = latest.get("top_candidates", [])
    candidates_rows = ""
    for i, c in enumerate(candidates[:10], 1):
        candidates_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{c.get('ood_score', 0):.4f}</td>
            <td>{c.get('classification', '?')}</td>
            <td>{c.get('confidence', 0):.1%}</td>
            <td><span class="tag tag-source">{c.get('source', '?')}</span></td>
            <td>{c.get('detected_at', '?')[:16]}</td>
        </tr>"""

    # Build corrections list
    corrections_html = ""
    for snap in snapshots:
        for c in snap.get("corrections_applied", []):
            corrections_html += f'<div class="correction-item">'
            corrections_html += f'<strong>Day {snap.get("day", "?")}:</strong> {c}'
            corrections_html += "</div>"

    if not corrections_html:
        corrections_html = '<div class="card">No corrections needed yet.</div>'

    # Images for charts
    def chart_img(b64, alt):
        if b64:
            return f'<div class="chart-container"><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'
        return f'<div class="card">Chart unavailable (install matplotlib)</div>'

    total_images = streaming_state.get("total_images", 0)
    total_anomalies = streaming_state.get("total_anomalies", 0)
    total_near = streaming_state.get("total_near_misses", 0)
    total_corrections = streaming_state.get("total_corrections", 0)
    strategy = streaming_state.get("current_strategy", "normal")
    runtime = streaming_state.get("total_runtime_hours", 0)

    day_images = latest.get("images_analyzed", 0)
    day_anomalies = latest.get("anomalies_found", 0)
    day_rate = latest.get("anomaly_rate", 0)
    day_iph = latest.get("images_per_hour", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AstroLens Streaming Report - Day {current_day}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<h1>AstroLens Streaming Discovery</h1>
<div class="subtitle">
    Day {current_day} of {target_days} |
    {datetime.now().strftime('%Y-%m-%d %H:%M')} |
    Strategy: {strategy.upper()} |
    Runtime: {runtime:.1f}h
</div>

<h2>Overview</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_images:,}</div>
        <div class="stat-label">Total Images</div>
    </div>
    <div class="stat-card highlight">
        <div class="stat-value">{total_anomalies}</div>
        <div class="stat-label">Anomalies Found</div>
    </div>
    <div class="stat-card warning">
        <div class="stat-value">{total_near}</div>
        <div class="stat-label">Near Misses</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{total_corrections}</div>
        <div class="stat-label">Self-Corrections</div>
    </div>
</div>

<h2>Today (Day {current_day})</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{day_images:,}</div>
        <div class="stat-label">Images Today</div>
    </div>
    <div class="stat-card {'highlight' if day_anomalies > 0 else ''}">
        <div class="stat-value">{day_anomalies}</div>
        <div class="stat-label">Anomalies Today</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{day_rate:.2f}%</div>
        <div class="stat-label">Anomaly Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{day_iph:.0f}</div>
        <div class="stat-label">Images/Hour</div>
    </div>
</div>

<h2>Trends</h2>
{chart_img(trend_chart, "Anomaly Trend")}
{chart_img(throughput_chart, "Throughput")}

<h2>Threshold Evolution</h2>
{chart_img(threshold_chart, "Threshold")}

<h2>Source Effectiveness</h2>
{chart_img(source_chart, "Sources")}

<h2>Top Candidates</h2>
<div class="card">
<table>
<thead>
    <tr><th>#</th><th>OOD Score</th><th>Class</th><th>Confidence</th><th>Source</th><th>Detected</th></tr>
</thead>
<tbody>
    {candidates_rows if candidates_rows else '<tr><td colspan="6" style="text-align:center;color:#484f58">No candidates yet</td></tr>'}
</tbody>
</table>
</div>

<h2>YOLO Transient Detection (Primary Feature)</h2>
<div class="stats-grid">
    <div class="stat-card highlight">
        <div class="stat-value">{latest.get('yolo_confirmations', 0)}</div>
        <div class="stat-label">YOLO Detections</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{latest.get('yolo_images_scanned', 0)}</div>
        <div class="stat-label">Transient Images Scanned</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{latest.get('yolo_retrain_runs', 0)}</div>
        <div class="stat-label">YOLO Retrain Runs</div>
    </div>
    <div class="stat-card {'highlight' if latest.get('yolo_available') else 'danger'}">
        <div class="stat-value">{'ACTIVE' if latest.get('yolo_available') else 'STANDBY'}</div>
        <div class="stat-label">YOLO Status</div>
    </div>
</div>
<div class="card">
    <p>YOLO transient detection runs <strong>only on transient-relevant sources</strong>
    (ZTF alerts, supernovae, gravitational lenses, galaxy mergers, peculiar galaxies).
    It does NOT run on regular galaxy morphology images from SDSS/DECaLS.</p>
    <p>When YOLO confirms a transient, the candidate is promoted to a definite anomaly
    and stored as a publishable detection.</p>
</div>
{chart_img(yolo_chart, "YOLO Detection Trend")}

<h2>Health & Errors</h2>
<div class="stats-grid">
    <div class="stat-card {'danger' if streaming_state.get('total_errors', 0) > 10 else ''}">
        <div class="stat-value">{streaming_state.get('total_errors', 0)}</div>
        <div class="stat-label">Total Errors</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{streaming_state.get('api_restarts', 0)}</div>
        <div class="stat-label">API Restarts</div>
    </div>
</div>
{"<div class='card'><h3>Current Issues</h3>" + "".join(f"<div class='correction-item'>{i}</div>" for i in latest.get('health_issues', [])) + "</div>" if latest.get('health_issues') else "<div class='card'>No health issues detected.</div>"}

<h2>Self-Corrections</h2>
{corrections_html}

<div class="footer">
    Generated by <a href="https://github.com/samantaba/astroLens">AstroLens</a> v1.1.0<br>
    If this tool helps your research, please star the repo.
</div>

</div>
</body>
</html>"""

    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = output_dir / f"day_{current_day}_{date_str}.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def generate_final_summary(
    streaming_state: dict,
    output_dir: Path,
) -> Path:
    """Generate the final publishing-ready summary report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = streaming_state.get("daily_snapshots", [])
    best_candidates = streaming_state.get("best_candidates", [])
    total_images = streaming_state.get("total_images", 0)
    total_anomalies = streaming_state.get("total_anomalies", 0)
    total_near = streaming_state.get("total_near_misses", 0)
    runtime = streaming_state.get("total_runtime_hours", 0)
    total_days = streaming_state.get("current_day", 0)
    strategy = streaming_state.get("current_strategy", "normal")
    strategy_history = streaming_state.get("strategy_history", [])

    # Charts
    trend_chart = _create_anomaly_trend_chart(snapshots) if snapshots else ""
    threshold_chart = _create_threshold_chart(snapshots) if snapshots else ""
    source_chart = _create_source_chart(snapshots)
    throughput_chart = _create_throughput_chart(snapshots) if snapshots else ""
    yolo_chart = _create_yolo_chart(snapshots) if snapshots else ""

    # YOLO stats (primary publishable output)
    yolo_confirmations = streaming_state.get("yolo_confirmations", 0)
    yolo_scanned = streaming_state.get("yolo_images_scanned", 0)
    yolo_retrain_runs = streaming_state.get("yolo_retrain_runs", 0)
    yolo_available = streaming_state.get("yolo_available", False)
    yolo_detections = streaming_state.get("yolo_detections", [])
    yolo_rate = (yolo_confirmations / yolo_scanned * 100) if yolo_scanned > 0 else 0

    # Build YOLO detections table
    yolo_det_rows = ""
    for i, d in enumerate(yolo_detections[:20], 1):
        yolo_det_rows += f"""
        <tr>
            <td>{i}</td>
            <td style="color:#da3633;font-weight:700">{d.get('yolo_confidence', 0):.1%}</td>
            <td>{d.get('ood_score', 0):.4f}</td>
            <td>{d.get('classification', '?')}</td>
            <td><span class="tag tag-source">{d.get('source', '?')}</span></td>
            <td>{d.get('detected_at', '?')[:16]}</td>
        </tr>"""

    # Candidates table
    candidates_rows = ""
    for i, c in enumerate(best_candidates[:20], 1):
        candidates_rows += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{c.get('ood_score', 0):.4f}</strong></td>
            <td>{c.get('classification', '?')}</td>
            <td>{c.get('confidence', 0):.1%}</td>
            <td><span class="tag tag-source">{c.get('source', '?')}</span></td>
            <td>{c.get('detected_at', '?')[:16]}</td>
            <td>{Path(c.get('image_path', '')).name if c.get('image_path') else '?'}</td>
        </tr>"""

    # Strategy changes
    strategy_rows = ""
    for s in strategy_history:
        strategy_rows += f"""
        <tr>
            <td>Day {s.get('day', '?')}</td>
            <td>{s.get('from', '?')} &rarr; {s.get('to', '?')}</td>
            <td>{s.get('reason', '')}</td>
        </tr>"""

    # All corrections
    all_corrections = []
    for snap in snapshots:
        for c in snap.get("corrections_applied", []):
            all_corrections.append(
                f"<strong>Day {snap.get('day', '?')}:</strong> {c}"
            )

    corrections_html = "".join(
        f'<div class="correction-item">{c}</div>' for c in all_corrections
    )
    if not corrections_html:
        corrections_html = '<div class="card">No corrections were needed.</div>'

    # Per-day summary table
    day_rows = ""
    for s in snapshots:
        day_rows += f"""
        <tr>
            <td>{s.get('day', '?')}</td>
            <td>{s.get('date', '?')}</td>
            <td>{s.get('images_analyzed', 0):,}</td>
            <td>{s.get('anomalies_found', 0)}</td>
            <td>{s.get('yolo_confirmations', 0)}/{s.get('yolo_images_scanned', 0)}</td>
            <td>{s.get('anomaly_rate', 0):.2f}%</td>
            <td>{s.get('threshold_end', 0):.3f}</td>
            <td>{s.get('images_per_hour', 0):.0f}</td>
            <td>{len(s.get('corrections_applied', []))}</td>
        </tr>"""

    def chart_img(b64, alt):
        if b64:
            return f'<div class="chart-container"><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'
        return ""

    anomaly_rate = (total_anomalies / total_images * 100) if total_images > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AstroLens Streaming Discovery - Final Summary</title>
<style>{_CSS}
.summary-box {{
    background: linear-gradient(135deg, rgba(88, 166, 255, 0.08), rgba(35, 134, 54, 0.08));
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 24px;
    margin: 24px 0;
}}
.summary-box h3 {{ color: #58a6ff; margin-top: 0; }}
</style>
</head>
<body>
<div class="container">

<h1>AstroLens Streaming Discovery</h1>
<div class="subtitle">
    Final Summary | {total_days} days |
    {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="summary-box">
    <h3>Executive Summary</h3>
    <p>
        Over <strong>{total_days} days</strong> ({runtime:.1f} hours), AstroLens analyzed
        <strong>{total_images:,}</strong> astronomical images from multiple survey sources.
        The system identified <strong>{total_anomalies} anomaly candidates</strong>
        (rate: {anomaly_rate:.2f}%) and <strong>{total_near} near-misses</strong>
        that warrant further investigation.
    </p>
    <p>
        <strong>YOLO Transient Detection (v1.1.0):</strong> YOLO scanned
        <strong>{yolo_scanned}</strong> transient-source images and confirmed
        <strong>{yolo_confirmations}</strong> transient detections
        (rate: {yolo_rate:.2f}%). YOLO was retrained <strong>{yolo_retrain_runs}</strong>
        time(s) during the run.
    </p>
    <p>
        The self-correcting intelligence applied <strong>{len(all_corrections)} adjustments</strong>
        to optimize detection throughout the run. Final strategy: <strong>{strategy.upper()}</strong>.
    </p>
</div>

<h2>Key Metrics</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_days}</div>
        <div class="stat-label">Days</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{total_images:,}</div>
        <div class="stat-label">Images Analyzed</div>
    </div>
    <div class="stat-card highlight">
        <div class="stat-value">{total_anomalies}</div>
        <div class="stat-label">Anomalies</div>
    </div>
    <div class="stat-card warning">
        <div class="stat-value">{total_near}</div>
        <div class="stat-label">Near Misses</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{anomaly_rate:.2f}%</div>
        <div class="stat-label">Anomaly Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{len(all_corrections)}</div>
        <div class="stat-label">Self-Corrections</div>
    </div>
</div>

<h2>YOLO Transient Detection (Primary Feature)</h2>
<div class="stats-grid">
    <div class="stat-card danger">
        <div class="stat-value">{yolo_confirmations}</div>
        <div class="stat-label">YOLO Detections</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{yolo_scanned}</div>
        <div class="stat-label">Transient Images Scanned</div>
    </div>
    <div class="stat-card warning">
        <div class="stat-value">{yolo_rate:.2f}%</div>
        <div class="stat-label">Detection Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{yolo_retrain_runs}</div>
        <div class="stat-label">YOLO Retrain Runs</div>
    </div>
</div>
<div class="card">
    <p>YOLO transient detection ran on <strong>transient-relevant sources only</strong>
    (ZTF alerts, supernovae, gravitational lenses, galaxy mergers, peculiar galaxies)
    as designed for v1.1.0. Regular galaxy images were analyzed by ViT+OOD ensemble only.</p>
</div>
{chart_img(yolo_chart, "YOLO Detection Trend")}
{'<h3>Top YOLO Detections (Publishable)</h3><div class="card"><table><thead><tr><th>#</th><th>YOLO Conf</th><th>OOD Score</th><th>Class</th><th>Source</th><th>Detected</th></tr></thead><tbody>' + yolo_det_rows + '</tbody></table></div>' if yolo_det_rows else '<div class="card"><p style="color:#484f58;text-align:center;">No YOLO detections recorded.</p></div>'}

<h2>Trends Over Time</h2>
{chart_img(trend_chart, "Anomaly Trend")}
{chart_img(throughput_chart, "Throughput")}
{chart_img(threshold_chart, "Threshold Evolution")}
{chart_img(source_chart, "Source Effectiveness")}

<h2>Per-Day Breakdown</h2>
<div class="card">
<table>
<thead>
    <tr><th>Day</th><th>Date</th><th>Images</th><th>Anomalies</th><th>YOLO</th><th>Rate</th><th>Threshold</th><th>Img/Hr</th><th>Corrections</th></tr>
</thead>
<tbody>
    {day_rows if day_rows else '<tr><td colspan="9" style="text-align:center;color:#484f58">No data yet</td></tr>'}
</tbody>
</table>
</div>

<h2>Top {min(20, len(best_candidates))} Candidates</h2>
<div class="card">
<table>
<thead>
    <tr><th>#</th><th>OOD Score</th><th>Class</th><th>Confidence</th><th>Source</th><th>Detected</th><th>File</th></tr>
</thead>
<tbody>
    {candidates_rows if candidates_rows else '<tr><td colspan="7" style="text-align:center;color:#484f58">No candidates yet</td></tr>'}
</tbody>
</table>
</div>

<h2>Self-Correction Log</h2>
{corrections_html}

{"<h2>Strategy Changes</h2><div class='card'><table><thead><tr><th>Day</th><th>Change</th><th>Reason</th></tr></thead><tbody>" + strategy_rows + "</tbody></table></div>" if strategy_rows else ""}

<div class="footer">
    Generated by <a href="https://github.com/samantaba/astroLens">AstroLens</a> v1.1.0<br>
    If this tool helps your research, please
    <a href="https://github.com/samantaba/astroLens">star the repo</a>.
</div>

</div>
</body>
</html>"""

    summary_path = output_dir / f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    summary_path.write_text(html, encoding="utf-8")

    # Also save a JSON version for programmatic access
    json_path = output_dir / "final_summary.json"
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "total_days": total_days,
        "total_runtime_hours": runtime,
        "total_images": total_images,
        "total_anomalies": total_anomalies,
        "total_near_misses": total_near,
        "anomaly_rate": anomaly_rate,
        "strategy": strategy,
        "corrections": len(all_corrections),
        "best_candidates": best_candidates[:20],
        "daily_snapshots": snapshots,
        # YOLO primary output
        "yolo_confirmations": yolo_confirmations,
        "yolo_images_scanned": yolo_scanned,
        "yolo_detection_rate": yolo_rate,
        "yolo_retrain_runs": yolo_retrain_runs,
        "yolo_detections": yolo_detections[:30],
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str),
                         encoding="utf-8")

    return summary_path
