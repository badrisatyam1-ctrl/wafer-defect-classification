"""
Wafer Defect Analysis — Report Generator
==========================================
Generates professional PDF and CSV reports from classification results.
Designed for fab engineers and quality assurance teams.

API:
    generate_pdf_report(results, output_path) -> Path
    generate_csv_report(results, output_path) -> Path
    generate_pdf_bytes(results) -> bytes
    generate_csv_bytes(results) -> bytes
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# fpdf2 for PDF generation — lightweight, no system deps
from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deployment.chatbot import DEFECT_KNOWLEDGE


# ═══════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════

class WaferReportPDF(FPDF):
    """Custom PDF with branded header/footer for wafer defect reports."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 58, 138)  # Dark blue
        self.cell(0, 10, "Wafer Defect Classification Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        # Divider line
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Wafer Yield Analytics Console", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 58, 138)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(3)

    def key_value(self, key: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(55, 7, f"{key}:", new_x="RIGHT")
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def _sanitize(text: str) -> str:
    """Replace Unicode characters that FPDF built-in Latin-1 fonts cannot encode."""
    return (
        str(text)
        .replace("\u2014", "-")    # em dash
        .replace("\u2013", "-")    # en dash
        .replace("\u2012", "-")    # figure dash
        .replace("\u2011", "-")    # non-breaking hyphen
        .replace("\u2010", "-")    # hyphen
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .replace("\u00b0", " deg")
        .replace("\u03bc", "u")
        .replace("\u2192", "->")
        .replace("\u2022", "*")
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _save_array_as_temp_png(arr: np.ndarray) -> str:
    """Save a numpy array as a temporary PNG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(tmp.name)
    return tmp.name


def generate_pdf_bytes(results: List[Dict[str, Any]], batch_name: str = "Analysis") -> bytes:
    """
    Generate a PDF report as bytes from a list of prediction results.

    Each result dict should have:
        - predicted_class: str
        - confidence: float
        - class_probabilities: dict
        - overlay: np.ndarray (optional)
        - input_image: np.ndarray (optional)
        - filename: str (optional)
        - timestamp: str (optional)
    """
    pdf = WaferReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Summary Page ────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Executive Summary")

    total = len(results)
    defects = sum(1 for r in results if r.get("predicted_class", "normal") != "normal")
    normals = total - defects
    avg_conf = np.mean([r.get("confidence", 0) for r in results]) if results else 0

    pdf.key_value("Batch Name", _sanitize(batch_name), bold_value=True)
    pdf.key_value("Total Wafers Analyzed", str(total))
    pdf.key_value("Normal Wafers", f"{normals} ({normals/total*100:.1f}%)" if total else "0")
    pdf.key_value("Defective Wafers", f"{defects} ({defects/total*100:.1f}%)" if total else "0")
    pdf.key_value("Average Confidence", f"{avg_conf:.1%}")
    pdf.key_value("Yield Rate", f"{normals/total*100:.1f}%" if total else "N/A")
    pdf.ln(5)

    # Defect distribution table
    if total > 0:
        pdf.section_title("Defect Distribution")
        defect_counts: Dict[str, int] = {}
        for r in results:
            cls = r.get("predicted_class", "unknown")
            defect_counts[cls] = defect_counts.get(cls, 0) + 1

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(70, 8, "Defect Class", border=1, fill=True, align="C")
        pdf.cell(40, 8, "Count", border=1, fill=True, align="C")
        pdf.cell(40, 8, "Percentage", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(30, 30, 30)
        for cls, count in sorted(defect_counts.items(), key=lambda x: -x[1]):
            pdf.set_font("Helvetica", "", 10)
            color = (220, 245, 225) if cls == "normal" else (254, 226, 226)
            pdf.set_fill_color(*color)
            pdf.cell(70, 7, cls.replace("_", " ").title(), border=1, align="C", fill=True)
            pdf.cell(40, 7, str(count), border=1, align="C", fill=True)
            pdf.cell(40, 7, f"{count/total*100:.1f}%", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    # ── Individual Wafer Pages ──────────────────────────────────────
    temp_files = []
    for i, result in enumerate(results):
        pdf.add_page()
        filename = result.get("filename", f"Wafer #{i+1}")
        pred = result.get("predicted_class", "unknown")
        conf = result.get("confidence", 0)
        timestamp = result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        pdf.section_title(_sanitize(f"Wafer Analysis - {filename}"))
        pdf.key_value("Predicted Class", _sanitize(pred.replace("_", " ").title()), bold_value=True)
        pdf.key_value("Confidence", f"{conf:.1%}")
        pdf.key_value("Timestamp", _sanitize(timestamp))

        # Severity
        if pred == "normal":
            severity = "None - Wafer OK"
        elif conf > 0.85:
            severity = "HIGH - Immediate attention required"
        elif conf > 0.60:
            severity = "MEDIUM - Schedule inspection"
        else:
            severity = "LOW - Monitor closely"
        pdf.key_value("Severity", severity, bold_value=True)

        # Class probabilities
        probs = result.get("class_probabilities", {})
        if probs:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Class Probabilities:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
            for cls_name, prob in sorted_probs:
                bar_width = prob * 100
                pdf.cell(40, 6, cls_name.replace("_", " ").title())
                pdf.cell(20, 6, f"{prob:.1%}")
                # Draw bar
                x = pdf.get_x()
                y = pdf.get_y() + 1
                pdf.set_fill_color(59, 130, 246)
                pdf.rect(x, y, bar_width * 0.8, 4, style="F")
                pdf.ln(6)

        # Images
        overlay = result.get("overlay")
        input_img = result.get("input_image")
        if overlay is not None:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Grad-CAM Visualization:", new_x="LMARGIN", new_y="NEXT")
            tmp = _save_array_as_temp_png(overlay)
            temp_files.append(tmp)
            try:
                pdf.image(tmp, x=15, w=80)
            except Exception:
                pass

        if input_img is not None:
            tmp = _save_array_as_temp_png(input_img)
            temp_files.append(tmp)
            try:
                pdf.image(tmp, x=100, w=80)
            except Exception:
                pass

        # Root cause analysis from knowledge base
        knowledge = DEFECT_KNOWLEDGE.get(pred)
        if knowledge and pred != "normal":
            pdf.ln(5)
            pdf.section_title("Engineering Analysis")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _sanitize(f"Description: {knowledge['description']}"))
            pdf.ln(2)

            if knowledge.get("root_causes"):
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Root Causes:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                for j, cause in enumerate(knowledge["root_causes"][:3], 1):
                    pdf.multi_cell(0, 5, _sanitize(f"  {j}. {cause}"))
                    pdf.ln(1)

            if knowledge.get("solutions"):
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Recommended Actions:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                for j, sol in enumerate(knowledge["solutions"][:3], 1):
                    pdf.multi_cell(0, 5, _sanitize(f"  {j}. {sol}"))
                    pdf.ln(1)

            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, "Yield Impact:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _sanitize(knowledge.get("impact", "N/A")))

    # Output
    pdf_bytes = pdf.output()

    # Cleanup temp files
    for f in temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    return bytes(pdf_bytes)


def generate_pdf_report(results: List[Dict[str, Any]], output_path: Path, batch_name: str = "Analysis") -> Path:
    """Generate and save a PDF report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = generate_pdf_bytes(results, batch_name)
    output_path.write_bytes(pdf_bytes)
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# CSV REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_csv_bytes(results: List[Dict[str, Any]]) -> bytes:
    """Generate a CSV report as bytes from prediction results."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Wafer ID", "Predicted Class", "Confidence",
        "Severity", "Timestamp", "Top-2 Margin",
        "Normal Prob", "Center Prob", "Edge Ring Prob",
        "Edge Loss Prob", "Scratch Prob", "Ring Prob",
        "Cluster Prob", "Full Fail Prob"
    ])

    for i, result in enumerate(results):
        pred = result.get("predicted_class", "unknown")
        conf = result.get("confidence", 0)
        probs = result.get("class_probabilities", {})
        timestamp = result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        filename = result.get("filename", f"wafer_{i+1}")

        # Calculate severity
        if pred == "normal":
            severity = "None"
        elif conf > 0.85:
            severity = "High"
        elif conf > 0.60:
            severity = "Medium"
        else:
            severity = "Low"

        # Top-2 margin
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0] if sorted_probs else 0

        writer.writerow([
            filename, pred, f"{conf:.4f}",
            severity, timestamp, f"{margin:.4f}",
            f"{probs.get('normal', 0):.4f}",
            f"{probs.get('center', 0):.4f}",
            f"{probs.get('edge_ring', 0):.4f}",
            f"{probs.get('edge_loss', 0):.4f}",
            f"{probs.get('scratch', 0):.4f}",
            f"{probs.get('ring', 0):.4f}",
            f"{probs.get('cluster', 0):.4f}",
            f"{probs.get('full_fail', 0):.4f}",
        ])

    return output.getvalue().encode("utf-8")


def generate_csv_report(results: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate and save a CSV report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = generate_csv_bytes(results)
    output_path.write_bytes(csv_bytes)
    return output_path
