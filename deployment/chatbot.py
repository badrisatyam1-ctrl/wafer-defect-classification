import streamlit as st
import random


def generate_wafer_response(question):
    prediction = st.session_state.get("prediction", "unknown")
    confidence = st.session_state.get("confidence", 0)
    probs = st.session_state.get("probabilities", {})

    knowledge = {
        "scratch": {
            "cause": "Mechanical abrasion during wafer transport or chuck contact. Likely root cause: end-effector misalignment or particulate on handling surfaces.",
            "impact": "Linear surface defects disrupt interconnect layers, causing open/short failures in affected dies.",
            "fix": "Inspect robotic end-effectors and load-port contact surfaces. Run particle verification on transfer arms. Tighten mechanical tolerances."
        },
        "edge_loss": {
            "cause": "Non-uniform etch rate or CMP over-polishing at wafer periphery. Edge exclusion zone violated during lithography exposure.",
            "impact": "Peripheral dies exhibit parametric drift or complete functional failure. Yield loss scales with exclusion zone width.",
            "fix": "Recalibrate edge exclusion parameters. Verify etch uniformity profile across full wafer radius. Adjust CMP pressure profile near bevel."
        },
        "edge_ring": {
            "cause": "Radial non-uniformity in CVD/PVD deposition chamber. Susceptor edge temperature gradient or showerhead flow imbalance.",
            "impact": "Concentric ring pattern at wafer edge causes systematic yield loss in outer die rows.",
            "fix": "Profile deposition thickness across wafer radius. Verify chamber susceptor concentricity and showerhead gas distribution uniformity."
        },
        "cluster": {
            "cause": "Localized particle contamination from process chamber, gas delivery, or ambient environment. Possible source: degraded O-ring or chamber liner flaking.",
            "impact": "Clustered defects cause localized die failures. Kill ratio depends on particle density and size distribution.",
            "fix": "Run particle scan on suspect chamber. Inspect and replace degraded seals, liners, and gas filters. Verify HEPA/ULPA filter integrity."
        },
        "center": {
            "cause": "Process non-uniformity at wafer center due to gas starvation, temperature hot-spot on electrostatic chuck, or plasma density variation.",
            "impact": "Central dies exhibit elevated defect density. Critical dimension (CD) drift or film thickness variation impacts device performance.",
            "fix": "Map thermal profile of ESC. Verify gas flow distribution through showerhead center zone. Check plasma density uniformity via Langmuir probe data."
        },
        "ring": {
            "cause": "Radial standing wave in plasma chamber or CMP slurry flow pattern creating concentric thickness variation.",
            "impact": "Ring-shaped defect signature affects multiple die rows at consistent radius. Systematic yield loss pattern.",
            "fix": "Audit CMP pad conditioning profile. Verify RF power coupling uniformity. Cross-reference with in-situ thickness metrology data."
        },
        "full_fail": {
            "cause": "Catastrophic process excursion — equipment malfunction, incorrect recipe execution, or gross contamination event.",
            "impact": "Total wafer scrap. All dies non-functional. Immediate lot hold required to prevent downstream propagation.",
            "fix": "Initiate equipment lockout. Perform full fault trace on process tool logs. Quarantine affected lot and adjacent lots. Root cause analysis mandatory before tool release."
        },
        "normal": {
            "cause": "No anomalous defect signature detected. Wafer within process control limits.",
            "impact": "Nominal yield expected. No corrective action required.",
            "fix": "Continue standard production flow. Log baseline metrics for SPC trending."
        }
    }

    info = knowledge.get(prediction, {
        "cause": "Defect class not catalogued in current knowledge base.",
        "impact": "Unable to assess yield impact without further inline metrology data.",
        "fix": "Escalate to process integration team. Cross-reference with SPC charts and tool maintenance logs."
    })

    # Confidence-tiered severity assessment
    if confidence > 0.8:
        confidence_text = "High confidence — classification statistically validated"
    elif confidence > 0.5:
        confidence_text = "Moderate confidence — secondary defect mode overlap possible"
    else:
        confidence_text = "Low confidence — ambiguous signature, manual review required"

    # Randomized technical intros to prevent repetitive output
    intro_options = [
        f"Inline inspection classifies this wafer as **{prediction}**.",
        f"Defect signature identified: **{prediction}**.",
        f"Automated optical inspection result: **{prediction}** pattern detected."
    ]

    intro = random.choice(intro_options)

    return f"""
{intro}

**Confidence:** {confidence:.2f} ({confidence_text})

**Root Cause Analysis:**
{info['cause']}

**Yield Impact Assessment:**
{info['impact']}

**Corrective Action:**
{info['fix']}
"""
