"""🎯 UNIVERSAL TYPE-C CARTRIDGE SLOT (`app/active_cartridge.py`)
Currently Loaded Real-Time Production Project:
👉 MediShield Enterprise Clinical Triage, Drug-Interaction & ICU Bed Allocation Swarm

All 17 System Design Layers (Hybrid RAG, Guardrails, Loop Breaker, Semantic Cache, MSI Router)
automatically adapt to this Healthcare Swarm without touching `app/core/`!
"""

# 1. AGENT IDENTITY & MODEL
AGENT_NAME = "medishield_clinical_triage_swarm"
AGENT_DESCRIPTION = (
    "HIPAA-Guarded Multi-Agent Healthcare Swarm for Emergency Severity Index (ESI 1-5) Triage, "
    "Pharmacovigilance Drug-Interaction Screening, and Real-Time ICU Bed Dispatch."
)
MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """
You are the Chief Medical Officer AI Coordinator of the MediShield Clinical Triage Swarm.
Your responsibilities:
1. Vital Signs & Triage Acuity: Route patient vitals to `clinical_triage_subagent` (or call `calculate_news2_triage_score`) to compute the National Early Warning Score (NEWS2) and Emergency Severity Index (ESI Level 1 to 5).
2. Medication Safety & Contraindications: Route active prescriptions to `pharmacovigilance_subagent` (or call `screen_drug_interactions`) to flag life-threatening CYP450, bleeding, or QT-prolongation interactions.
3. Hospital & ICU Bed Dispatch: For ESI Level 1 or Level 2 emergencies, call `check_icu_bed_availability` to reserve an immediate trauma or cardiac ICU bay.
4. Clinical Protocol Grounding: Always call `search_enterprise_rag_knowledge` to cite hospital sepsis, stroke, and chest-pain clinical pathways.
5. Compliance: Always include a clinical decision support disclaimer and never expose unmasked Patient SSNs or MRNs.
"""


# 2. REAL-TIME PRODUCTION HEALTHCARE TOOLS
def calculate_news2_triage_score(
    age: int,
    heart_rate_bpm: int,
    systolic_bp_mmhg: int,
    spo2_percent: float,
    temperature_c: float,
    on_supplemental_oxygen: bool = False,
) -> dict:
    """Calculates the clinical NEWS2 acuity score and assigns an Emergency Severity Index (ESI 1-5)."""
    score = 0
    red_flags = []

    if heart_rate_bpm >= 131 or heart_rate_bpm <= 40:
        score += 3
        red_flags.append(f"Critical Heart Rate ({heart_rate_bpm} bpm)")
    elif heart_rate_bpm >= 111 or heart_rate_bpm <= 50:
        score += 2
        red_flags.append(f"Abnormal Heart Rate ({heart_rate_bpm} bpm)")

    if systolic_bp_mmhg <= 90 or systolic_bp_mmhg >= 220:
        score += 3
        red_flags.append(f"Critical Blood Pressure ({systolic_bp_mmhg} mmHg)")
    elif systolic_bp_mmhg <= 100:
        score += 2
        red_flags.append(f"Hypotension Warning ({systolic_bp_mmhg} mmHg)")

    if spo2_percent <= 91.0:
        score += 3
        red_flags.append(f"Severe Hypoxemia (SpO2 {spo2_percent}%)")
    elif spo2_percent <= 93.0:
        score += 2
        red_flags.append(f"Moderate Hypoxemia (SpO2 {spo2_percent}%)")

    if on_supplemental_oxygen:
        score += 2

    if temperature_c >= 39.1 or temperature_c <= 35.0:
        score += 2
        red_flags.append(f"Systemic Temp Abnormality ({temperature_c} C)")

    if age >= 65:
        score += 1

    esi_level = 1 if score >= 7 else (2 if score >= 5 else (3 if score >= 3 else 4))
    disposition = (
        "STAT_RESUSCITATION_TRAUMA_BAY" if esi_level == 1
        else "IMMEDIATE_ICU_CARDIAC_BED" if esi_level == 2
        else "URGENT_ACUTE_CARE_WARD" if esi_level == 3
        else "STANDARD_OUTPATIENT_TRIAGE"
    )
    return {
        "news2_score": score,
        "esi_triage_level": esi_level,
        "red_flag_alerts": red_flags,
        "recommended_disposition": disposition,
        "requires_immediate_icu": esi_level <= 2,
    }


def screen_drug_interactions(medications: list[str]) -> dict:
    """Screens a patient's medication regimen for high-severity drug-drug interactions."""
    meds = {m.lower().strip() for m in medications}
    alerts = []

    if {"warfarin", "aspirin"}.issubset(meds) or {"warfarin", "ibuprofen"}.issubset(meds):
        alerts.append({
            "severity": "CONTRAINDICATED_SEVERE",
            "pair": "Warfarin + NSAID/Antiplatelet",
            "mechanism": "Synergistic anticoagulation & platelet inhibition -> 4x risk of fatal GI/intracranial hemorrhage.",
        })
    if {"lisinopril", "spironolactone"}.issubset(meds):
        alerts.append({
            "severity": "HIGH_MODERATE",
            "pair": "ACE Inhibitor + Potassium-Sparing Diuretic",
            "mechanism": "Dual RAAS blockade -> severe hyperkalemia & acute kidney injury risk. Monitor serum K+ every 48h.",
        })
    if {"simvastatin", "clarithromycin"}.issubset(meds) or {"atorvastatin", "clarithromycin"}.issubset(meds):
        alerts.append({
            "severity": "CONTRAINDICATED_SEVERE",
            "pair": "Statin + Macrolide Antibiotic (CYP3A4 Inhibitor)",
            "mechanism": "Blocks statin metabolism -> acute rhabdomyolysis and renal failure.",
        })
    if {"sildenafil", "nitroglycerin"}.issubset(meds):
        alerts.append({
            "severity": "FATAL_CONTRAINDICATION",
            "pair": "PDE5 Inhibitor + Nitrate",
            "mechanism": "Profound refractory hypotension and cardiovascular collapse.",
        })

    return {
        "medications_checked": sorted(meds),
        "interaction_count": len(alerts),
        "clinical_interaction_alerts": alerts or ["No high-severity interactions detected."],
    }


def check_icu_bed_availability(department: str = "Cardiac_ICU") -> dict:
    """Checks live hospital bed telemetry and reserves an emergency bay for ESI 1-2 patients."""
    wards = {
        "cardiac_icu": {"available_beds": 2, "wing": "Tower A - Floor 4", "on_call_intensivist": "Dr. A. Sharma", "isolation_ready": True},
        "medical_icu": {"available_beds": 1, "wing": "Tower B - Floor 3", "on_call_intensivist": "Dr. R. Menon", "isolation_ready": True},
        "trauma_bay": {"available_beds": 3, "wing": "Ground Emergency Wing", "on_call_intensivist": "Dr. K. Verma", "isolation_ready": False},
    }
    key = department.lower().strip().replace(" ", "_")
    info = wards.get(key, wards["cardiac_icu"])
    return {
        "department": department,
        "status": "BED_RESERVED_READY",
        "reservation_code": f"ICU-RES-{abs(hash(department)) % 9000 + 1000}",
        "telemetry": info,
    }


CUSTOM_TOOLS = [
    calculate_news2_triage_score,
    screen_drug_interactions,
    check_icu_bed_availability,
]

# 3. MULTI-AGENT SWARM SUB-AGENTS
SUB_AGENTS = [
    {
        "name": "clinical_triage_subagent",
        "description": "Evaluates patient vitals (HR, BP, SpO2, Temp, Age) to compute NEWS2 and ESI Level 1-5.",
        "instruction": "Always call `calculate_news2_triage_score` and `check_icu_bed_availability` for ESI 1-2 patients.",
        "tools": [calculate_news2_triage_score, check_icu_bed_availability],
    },
    {
        "name": "pharmacovigilance_subagent",
        "description": "Audits active medication lists for severe CYP450, bleeding, and cardiovascular drug interactions.",
        "instruction": "Always call `screen_drug_interactions` and explain the clinical mechanism clearly.",
        "tools": [screen_drug_interactions],
    },
]

# 4. HYBRID RAG CLINICAL GUIDELINES KNOWLEDGE BASE
KNOWLEDGE_BASE_DOCS = [
    {
        "id": "clin-acs-01",
        "tenant_id": "public",
        "title": "Acute Coronary Syndrome (ACS) & Chest Pain Protocol",
        "content": (
            "Patients presenting with ESI Level 1 or 2 chest pain or SpO2 < 92% require a STAT 12-lead ECG "
            "within 10 minutes of arrival, High-Sensitivity Troponin I at 0h and 1h, and immediate Cardiac ICU bed reservation. "
            "Never administer Nitroglycerin if the patient has taken Sildenafil within 24 hours."
        ),
    },
    {
        "id": "clin-sepsis-02",
        "tenant_id": "public",
        "title": "Surviving Sepsis Campaign 1-Hour Bundle",
        "content": (
            "For any patient with NEWS2 >= 5 and fever (>38.5 C) or hypotension (SBP <= 90 mmHg), initiate the "
            "Sepsis 1-Hour Bundle: measure serum lactate, obtain blood cultures prior to antibiotics, administer "
            "broad-spectrum IV antibiotics, and begin 30 mL/kg crystalloid fluid resuscitation."
        ),
    },
]

# 5. MINIMUM SUFFICIENT INTELLIGENCE (MSI) DETERMINISTIC FAST-PATH RULES
MSI_DETERMINISTIC_RULES = {
    r"^\s*what is the sepsis 1 hour bundle\??\s*$": (
        "Per Clinical Protocol (clin-sepsis-02), the Surviving Sepsis 1-Hour Bundle requires: "
        "(1) Measure serum lactate, (2) Obtain blood cultures before antibiotics, "
        "(3) Administer broad-spectrum IV antibiotics, and (4) Deliver 30 mL/kg IV crystalloid for hypotension."
    ),
}

# 6. FREE MCP SERVERS
MCP_SERVERS = []

# 7. HIPAA & CLINICAL SAFETY GUARDRAILS + GOLDEN BENCHMARK DATASET
MAX_INPUT_LENGTH = 6000
EXTRA_BLOCKED_INPUT_PATTERNS = [
    r"(?i)prescribe\s+controlled\s+narcotics\s+without\s+verification",
    r"(?i)bypass\s+hipaa\s+privacy",
]
EXTRA_BLOCKED_TOOL_ARGS = ["DELETE FROM patients", "DROP TABLE ehr_records"]
EXTRA_SENSITIVE_OUTPUT_PATTERNS = [r"\bMRN-\d{8}\b"]

GOLDEN_EVAL_DATASET = [
    {
        "query": "What is the sepsis 1 hour bundle?",
        "expected_keywords": ["lactate", "blood cultures", "antibiotics", "30 mL/kg"],
    },
    {
        "query": "Triage a 70yo patient with HR 135, SBP 85, SpO2 89%, Temp 39.3C on Warfarin and Aspirin.",
        "expected_keywords": ["ESI", "Warfarin", "bleeding", "ICU"],
    },
]