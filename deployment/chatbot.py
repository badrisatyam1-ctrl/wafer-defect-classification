"""
Wafer Defect Intelligence Chatbot
─────────────────────────────────
A comprehensive, context-aware chatbot that provides detailed semiconductor
engineering knowledge for each detected defect class. Supports real-time Q&A
with root cause analysis, corrective actions, impact assessment, and
process improvement recommendations.

API: chatbot_response(question, pred_class, confidence) -> str
"""
import os
import json
import google.generativeai as genai
# ═══════════════════════════════════════════════════════════════════════
# SEMICONDUCTOR KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════

DEFECT_KNOWLEDGE = {
    "center": {
        "description": "Center defects appear as concentrated anomalies in the middle of the wafer, typically forming a circular or radial pattern around the wafer center.",
        "root_causes": [
            "Non-uniform gas flow distribution in the CVD/PVD chamber causing center-heavy deposition",
            "Chuck temperature gradient — the center of the electrostatic chuck runs hotter or cooler than the edge",
            "Spin-coating speed too low during photoresist application, causing thick center buildup",
            "Improper wafer centering on the susceptor during epitaxial growth",
            "CMP (Chemical Mechanical Polishing) pad wear creating uneven pressure at the center",
        ],
        "solutions": [
            "Recalibrate gas showerhead nozzle alignment to ensure uniform flow across the wafer surface",
            "Inspect and replace the electrostatic chuck heater zones — verify center zone temperature ±0.5°C",
            "Increase spin-coating RPM by 10-15% and verify resist thickness uniformity via ellipsometry",
            "Run a wafer centering calibration routine on the robot arm and susceptor alignment pins",
            "Replace or re-condition the CMP pad; verify down-force pressure map is within spec",
        ],
        "impact": "Center defects cause 15-25% yield loss in the affected die region. They primarily impact circuit density and transistor gate integrity in the wafer core, which typically contains the highest-value die.",
        "prevention": "Implement regular chamber seasoning runs, monitor center-to-edge thickness uniformity with in-line metrology, and set SPC alarms for center-region thickness deviation >2%.",
    },

    "edge_ring": {
        "description": "Edge ring defects manifest as a ring-shaped pattern near the wafer edge, typically 2-5mm from the bevel.",
        "root_causes": [
            "Edge ring component erosion in the plasma etch chamber causing localized plasma non-uniformity",
            "Wafer edge exclusion zone contamination from handling fixtures or FOUPs",
            "Backside gas leakage around the wafer edge during CVD processing",
            "Edge bead removal (EBR) failure during photolithography leaving resist residue",
            "Temperature ring effect from improper clamp ring pressure during rapid thermal processing (RTP)",
        ],
        "solutions": [
            "Replace the silicon or quartz edge ring in the etch chamber — track RF hours for preventive replacement",
            "Clean FOUP contact surfaces and inspect robot end-effector for particle generation",
            "Check backside helium flow rate and verify edge seal integrity on the chuck",
            "Recalibrate the EBR nozzle position and solvent flow rate; verify edge clearance is 1-3mm",
            "Adjust clamp ring pressure to specification and verify thermal uniformity at wafer edge",
        ],
        "impact": "Edge ring defects typically cause 8-12% yield loss. While edge die are often lower value, persistent edge ring issues indicate chamber component degradation that will worsen over time.",
        "prevention": "Track edge ring RF-hour lifetime, implement periodic edge-region defect monitoring, and establish replacement schedules before erosion reaches critical levels.",
    },

    "edge_loss": {
        "description": "Edge loss defects show as material degradation, chipping, or pattern distortion along the outer 3-8mm of the wafer circumference.",
        "root_causes": [
            "Mechanical handling damage from robot arms, end-effectors, or cassette loading impacts",
            "CMP over-polishing at the wafer edge due to pad rebound effects (edge roll-off)",
            "Improper wafer notch alignment causing edge contact with chamber walls during transfer",
            "Thermal stress-induced micro-cracking from rapid temperature ramps at the exposed edge",
            "Wet etch solution turbulence creating non-uniform etching at the wafer periphery",
        ],
        "solutions": [
            "Inspect and recalibrate robot end-effector grip pressure — should be 0.5-1.5N, no higher",
            "Adjust CMP edge-exclusion zone to 3mm and implement retaining ring pressure optimization",
            "Run wafer notch alignment verification; ensure <0.1° angular deviation during transfers",
            "Reduce RTP ramp rate from >50°C/s to 30°C/s for the first 200°C to minimize thermal shock",
            "Optimize wet etch bath flow dynamics — add baffle plates to reduce edge turbulence",
        ],
        "impact": "Edge loss reduces usable die area by 5-15%. In severe cases, it can cause wafer breakage during subsequent processing steps, resulting in complete wafer loss and potential tool contamination.",
        "prevention": "Implement automated edge inspection after each mechanical handling step, use protective edge coatings, and establish maximum handling cycle limits per robot arm.",
    },

    "scratch": {
        "description": "Scratch defects appear as linear or curved marks on the wafer surface, ranging from micro-scratches (submicron) to macro-scratches visible under optical inspection.",
        "root_causes": [
            "CMP slurry particle agglomeration creating large abrasive particles that gouge the surface",
            "Contaminated or damaged polishing pad with embedded hard particles",
            "Wafer handling contact points with rough or contaminated surfaces (tweezers, end-effectors)",
            "Particle contamination in the spin-coating chuck creating drag lines during rotation",
            "Improper wafer cassette insertion/removal causing sliding contact with slot rails",
        ],
        "solutions": [
            "Filter CMP slurry through 0.1μm point-of-use filters and monitor particle count in real-time",
            "Replace the CMP pad and perform a break-in conditioning cycle before production wafers",
            "Audit all wafer contact surfaces — replace PEEK tweezers and inspect end-effector vacuum pads",
            "Run a chuck cleaning cycle and verify vacuum groove cleanliness with particle counters",
            "Inspect cassette slot rails for burrs or contamination; replace damaged cassettes immediately",
        ],
        "impact": "Scratches cause 10-30% yield loss depending on depth and location. Deep scratches crossing active circuit areas cause immediate device failure. Even shallow scratches can create reliability issues through stress concentration.",
        "prevention": "Implement real-time slurry particle monitoring, establish CMP pad lifetime limits, use automated handling wherever possible, and run regular tactile surface inspections on all contact points.",
    },

    "ring": {
        "description": "Ring defects appear as concentric circular patterns on the wafer, distinct from edge ring defects by occurring at various radial positions across the entire wafer surface.",
        "root_causes": [
            "Standing wave patterns in the plasma during etch or deposition creating radial thickness variations",
            "Spin-coating acceleration profile issues causing concentric resist thickness rings",
            "Chuck vacuum groove imprinting through thin films during high-temperature processing",
            "RF power coupling non-uniformity in the plasma chamber creating radial etch rate variation",
            "Gas flow resonance effects in the showerhead creating annular deposition patterns",
        ],
        "solutions": [
            "Tune plasma source power and bias power to eliminate standing wave modes — use Langmuir probe diagnostics",
            "Optimize spin-coating acceleration curve — use a multi-step ramp profile instead of single-step",
            "Switch to a flat vacuum chuck design or add a thermal pad layer to prevent groove imprinting",
            "Adjust RF matching network and verify reflected power is <1% across the process window",
            "Modify showerhead hole pattern or add a gas diffusion plate to break resonance patterns",
        ],
        "impact": "Ring defects cause 12-20% yield loss in a radially symmetric pattern. They are particularly problematic because they affect die at multiple radial positions, making binning strategies less effective.",
        "prevention": "Use in-situ plasma diagnostics (OES, VI probe) to detect process drift early. Implement wafer-level thickness uniformity monitoring after every deposition/etch step.",
    },

    "cluster": {
        "description": "Cluster defects appear as localized groups of point defects concentrated in specific areas, often caused by particle contamination events or localized process failures.",
        "root_causes": [
            "Particle shedding from chamber components (shields, liners, gas lines) during plasma processing",
            "Photomask defect transferring a localized pattern defect to multiple die in the same field",
            "Localized contamination droplet (chemical splash, condensation) on the wafer surface",
            "Arcing events in the plasma chamber creating localized damage clusters",
            "Cleanroom air handling system failure causing a burst of particle contamination",
        ],
        "solutions": [
            "Perform a full chamber wet clean followed by seasoning wafers — verify particle counts <0.1/cm² for >0.1μm",
            "Inspect the photomask under high-NA inspection and clean or replace if defects are found",
            "Audit chemical delivery systems for drip points, splash guards, and nozzle alignment",
            "Check RF cable connections, chamber ground straps, and electrode gaps for arcing evidence",
            "Verify HEPA/ULPA filter integrity, check for seal leaks around ceiling tiles, and audit pressurization",
        ],
        "impact": "Cluster defects cause 8-35% yield loss depending on cluster size and location. A single large cluster over a high-value die area can cause disproportionate economic loss.",
        "prevention": "Implement real-time in-situ particle monitoring, automate chamber cleaning schedules, and use defect source analysis (DSA) to trace cluster events back to specific process steps.",
    },

    "full_fail": {
        "description": "Full failure indicates catastrophic wafer-wide defect coverage where the majority of the wafer surface is compromised, rendering most or all die non-functional.",
        "root_causes": [
            "Complete process recipe error — wrong gas, wrong temperature, or wrong time applied",
            "Chamber catastrophic failure — heater burnout, sudden vacuum loss, or cooling water leak",
            "Wrong wafer type loaded (bare silicon vs. patterned, wrong orientation, wrong diameter)",
            "Chemical bath contamination or exhaustion causing complete etch/clean failure",
            "Photolithography focus/exposure completely out of specification across entire wafer",
        ],
        "solutions": [
            "Immediately quarantine the lot and halt the process tool — do NOT run additional wafers",
            "Perform root cause analysis: review recipe logs, process parameters, and alarm histories",
            "Implement recipe verification interlocks that cross-check wafer ID, recipe name, and lot ID",
            "Install real-time chemical concentration monitoring with automatic shutdown triggers",
            "Verify stepper/scanner leveling, focus calibration, and dose uniformity before resuming production",
        ],
        "impact": "Full failure represents 100% yield loss for the affected wafer. Cost impact ranges from $500-$50,000+ depending on process stage and wafer technology node. Immediate containment is critical to prevent additional wafer losses.",
        "prevention": "Implement equipment constant monitoring (FDC/FDD), recipe management systems with version control, interlock systems for critical parameters, and automated pre-process wafer ID verification.",
    },

    "normal": {
        "description": "The wafer shows no significant defect patterns. All regions pass inspection criteria within acceptable limits.",
        "root_causes": [],
        "solutions": [],
        "impact": "No yield impact — wafer is within specification. All die in the design area are expected to pass electrical test, subject to random defect density limits.",
        "prevention": "Continue current process controls and monitoring. Maintain equipment PM schedules and SPC charting to sustain this quality level.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ═══════════════════════════════════════════════════════════════════════

INTENT_KEYWORDS = {
    "cause": ["why", "cause", "reason", "root cause", "how does", "what causes", "origin", "source of", "due to"],
    "fix": ["fix", "solve", "solution", "resolve", "correct", "repair", "mitigate", "prevent", "stop", "eliminate", "how to fix", "action", "step"],
    "impact": ["impact", "effect", "damage", "yield", "cost", "loss", "severity", "consequence", "risk", "how bad", "serious"],
    "explain": ["explain", "what is", "describe", "tell me", "what does", "about", "detail", "overview", "summary", "mean"],
    "confidence": ["confidence", "accuracy", "sure", "certain", "reliable", "probability", "score", "how confident", "trust"],
    "prevention": ["prevent", "avoid", "future", "recurring", "again", "maintenance", "monitor", "control", "spc"],
    "all": ["everything", "all", "full report", "complete", "all info", "full analysis", "comprehensive"],
}


def detect_intent(question: str) -> list:
    """Detect one or more intents from the user's question."""
    q = question.lower().strip()
    intents = []

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                intents.append(intent)
                break

    # Default: give a comprehensive overview if no specific intent detected
    if not intents:
        intents = ["explain", "cause", "fix"]

    return intents


# ═══════════════════════════════════════════════════════════════════════
# LOCAL FALLBACK GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def generate_local_response(question: str, pred_class: str) -> str:
    """Generate a highly detailed local fallback response from the DEFECT_KNOWLEDGE base."""
    knowledge = DEFECT_KNOWLEDGE.get(pred_class)
    if not knowledge:
        return f"I am unable to provide analysis for the class '{pred_class}'. Please ensure a valid wafer defect image is loaded."

    intents = detect_intent(question)
    
    # Capitalize class name for display
    display_class = pred_class.replace("_", " ").title()
    
    response_parts = [
        f"### 🔍 Local Engineering Analysis: {display_class} Defect",
        f"*Note: The cloud AI API is currently unavailable or rate-limited. Serving offline domain knowledge.*",
        ""
    ]

    # If user wants a full report or "all"
    if "all" in intents:
        intents = ["explain", "cause", "fix", "impact", "prevention"]

    for intent in intents:
        if intent == "explain" and knowledge.get("description"):
            response_parts.append(f"**Description:**\n{knowledge['description']}\n")
        
        elif intent == "cause" and knowledge.get("root_causes"):
            response_parts.append("**Possible Root Causes:**")
            for idx, cause in enumerate(knowledge["root_causes"], 1):
                response_parts.append(f"{idx}. {cause}")
            response_parts.append("")
            
        elif intent == "fix" and knowledge.get("solutions"):
            response_parts.append("**Recommended Corrective Actions / Solutions:**")
            for idx, sol in enumerate(knowledge["solutions"], 1):
                response_parts.append(f"{idx}. {sol}")
            response_parts.append("")
            
        elif intent == "impact" and knowledge.get("impact"):
            response_parts.append(f"**Yield & Cost Impact:**\n{knowledge['impact']}\n")
            
        elif intent == "prevention" and knowledge.get("prevention"):
            response_parts.append(f"**Long-term Prevention & SPC Controls:**\n{knowledge['prevention']}\n")
            
        elif intent == "confidence":
            response_parts.append(f"**Confidence Analysis:**\nThe classification model identified this defect with statistical features matching known historical lots. Please verify the heatmaps to confirm localized process signature matches.")
            response_parts.append("")

    return "\n".join(response_parts)


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE BUILDER
# ═══════════════════════════════════════════════════════════════════════

def chatbot_response(question: str, pred_class: str, confidence: float) -> str:
    """
    Generate a detailed, context-aware response based on the detected defect,
    using Google Gemini API to dynamically answer the user's specific prompt.
    If the API call fails or the key is missing/rate-limited, gracefully
    falls back to a rule-based response from the local knowledge base.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return generate_local_response(question, pred_class)
    
    try:
        genai.configure(api_key=api_key)
        
        # Get context if available
        knowledge = DEFECT_KNOWLEDGE.get(pred_class, {})
        context_str = json.dumps(knowledge, indent=2) if knowledge else "No specific context available for this defect class."
        
        system_prompt = f"""You are the AI Wafer Assistant, an expert semiconductor engineer specializing in wafer defect analysis.
The user is currently analyzing a wafer map.
Current Prediction: {pred_class} (Confidence: {confidence:.1%})

Here is the domain knowledge regarding this defect:
{context_str}

Use this knowledge to answer the user's question accurately. Keep your answer highly professional, concise, and structured. Use Markdown formatting. Do not make up false semiconductor processes. Focus specifically on what the user asked. DO NOT USE EMOJIS in your response."""
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )
        
        response = model.generate_content(question)
        return response.text
        
    except Exception as e:
        # Fallback to local rule-based response when Gemini API is rate-limited (e.g. 429) or overloaded
        return generate_local_response(question, pred_class)

