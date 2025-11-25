"""
Agent-4 v3.1 — TinyLlama Sleep Recommendation Engine (per-person)

Reads per-person context from:
    outputs/person_reports/<Id>/chronotype.json
    outputs/person_reports/<Id>/behavioral.json
    outputs/person_reports/<Id>/sleep_debt.json
    outputs/person_reports/<Id>/ml_prediction.json   (optional, but expected)

Uses TinyLlama (local) to generate a ~200-word clinical-style
sleep recommendation with a friendly-doctor tone.

Outputs per person:
    outputs/person_reports/<Id>/slm_recommendation.json
    outputs/person_reports/<Id>/slm_recommendation.txt
"""

import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

LOCAL_MODEL_PATH = os.path.join("slm", "tinyllama")  # slm/tinyllama/
BASE_REPORT_DIR = os.path.join("outputs", "person_reports")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 400
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKENIZER = None
_MODEL = None


# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------

def sanitize(obj: Any) -> Any:
    """Convert numpy / torch / datetime types to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (datetime, )):
        return obj.isoformat()
    if isinstance(obj, (torch.Tensor, )):
        if obj.numel() == 1:
            return sanitize(obj.item())
        return sanitize(obj.tolist())
    return obj


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_person_folder(person_id: int) -> str:
    folder = os.path.join(BASE_REPORT_DIR, str(person_id))
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Person folder not found: {folder}. "
            f"Run Agent-2 and Agent-3 for this Id first."
        )
    return folder


# -------------------------------------------------------------------
# LOAD PER-PERSON CONTEXT
# -------------------------------------------------------------------

def load_person_context(person_id: int) -> Tuple[Dict, Dict, Dict, Optional[Dict]]:
    """
    Load chronotype, behavioral, sleep_debt, ml_prediction for a given Id.

    Expected files:
        chronotype.json   -> { "chronotype": { ... } }
        behavioral.json   -> { "behavioral": { ... } }
        sleep_debt.json   -> { "sleep_debt": { ... } }
        ml_prediction.json -> { "person_id": ..., "predicted_sleep_quality_hours": ... }
    """
    folder = ensure_person_folder(person_id)

    chrono_path = os.path.join(folder, "chronotype.json")
    beh_path = os.path.join(folder, "behavioral.json")  # NOTE: behavioral.json
    debt_path = os.path.join(folder, "sleep_debt.json")
    ml_path = os.path.join(folder, "ml_prediction.json")

    chrono_raw = load_json(chrono_path)
    beh_raw = load_json(beh_path)
    debt_raw = load_json(debt_path)
    ml_raw = load_json(ml_path) if os.path.exists(ml_path) else None

    chrono = chrono_raw.get("chronotype", {})
    behavioral = beh_raw.get("behavioral", {})
    sleep_debt = debt_raw.get("sleep_debt", {})

    return chrono, behavioral, sleep_debt, ml_raw


# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------

def load_tinyllama() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _TOKENIZER, _MODEL

    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Local TinyLlama model not found at '{LOCAL_MODEL_PATH}'. "
            "Expected directory: slm/tinyllama/"
        )

    print(f"Loading TinyLlama from {LOCAL_MODEL_PATH} on device {_DEVICE} ...")

    _TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32,
    ).to(_DEVICE)

    _MODEL.eval()
    return _TOKENIZER, _MODEL


# -------------------------------------------------------------------
# SNAPSHOT + SUMMARY LINE
# -------------------------------------------------------------------

def build_snapshot(
    person_id: int,
    chrono: Dict,
    behavioral: Dict,
    debt: Dict,
    ml_pred: Optional[Dict],
) -> Dict:
    """
    Build a compact snapshot to both save in JSON and feed into the prompt.
    """
    avg_sleep_hours = chrono.get("avg_sleep_hours")
    midpoint = chrono.get("avg_midpoint_hour")
    consistency = chrono.get("consistency_score")
    nights_used = chrono.get("nights_used")
    label = chrono.get("label")

    total_debt = debt.get("total_sleep_debt_hours")
    avg_deficit = debt.get("avg_daily_deficit_hours")
    severity = debt.get("severity")

    med_steps = behavioral.get("median_total_steps")
    med_sedentary = behavioral.get("median_sedentary_minutes")
    med_very_active = behavioral.get("median_very_active_minutes")
    med_sleep_eff = behavioral.get("median_sleep_efficiency")
    risk_flags = behavioral.get("risky_flags", [])

    predicted_hours = None
    if ml_pred is not None:
        predicted_hours = ml_pred.get("predicted_sleep_quality_hours")

    snapshot = {
        "person_id": int(person_id),
        "chronotype_label": label,
        "avg_sleep_hours": avg_sleep_hours,
        "avg_midpoint_hour": midpoint,
        "chronotype_consistency": consistency,
        "nights_used": nights_used,
        "sleep_debt_total_hours": total_debt,
        "sleep_debt_avg_daily_deficit": avg_deficit,
        "sleep_debt_severity": severity,
        "median_total_steps": med_steps,
        "median_sedentary_minutes": med_sedentary,
        "median_very_active_minutes": med_very_active,
        "median_sleep_efficiency": med_sleep_eff,
        "behavior_risky_flags": risk_flags,
        "predicted_sleep_quality_hours": predicted_hours,
    }
    return snapshot


def build_summary_line(snapshot: Dict) -> str:
    """
    One-sentence summary based mainly on sleep_debt_severity.
    You chose for High:
        "Your sleep debt is high but fixable — here’s how we’ll recover it week by week."
    """
    severity = (snapshot.get("sleep_debt_severity") or "").lower()

    if severity == "high":
        return "Your sleep debt is high but fixable — here’s how we’ll recover it week by week."
    elif severity == "moderate":
        return "Your sleep pattern shows moderate strain — we’ll steadily close your sleep debt and smooth your routine."
    elif severity == "low":
        return "Your sleep is mostly on track — we’ll fine-tune a few habits to keep it that way and avoid future debt."
    else:
        return "We will review your recent sleep and build a practical plan to improve both quality and consistency."


# -------------------------------------------------------------------
# PROMPT BUILDING
# -------------------------------------------------------------------

def build_prompt(snapshot: Dict) -> str:
    """
    Build a plain-text prompt for TinyLlama.
    No JSON output requirement; just a friendly clinical plan.
    """

    label = snapshot.get("chronotype_label")
    avg_sleep = snapshot.get("avg_sleep_hours")
    midpoint = snapshot.get("avg_midpoint_hour")
    consistency = snapshot.get("chronotype_consistency")
    nights = snapshot.get("nights_used")

    total_debt = snapshot.get("sleep_debt_total_hours")
    avg_deficit = snapshot.get("sleep_debt_avg_daily_deficit")
    severity = snapshot.get("sleep_debt_severity")

    steps = snapshot.get("median_total_steps")
    sedentary = snapshot.get("median_sedentary_minutes")
    very_active = snapshot.get("median_very_active_minutes")
    eff = snapshot.get("median_sleep_efficiency")
    risky_flags = snapshot.get("behavior_risky_flags") or []

    ml_hours = snapshot.get("predicted_sleep_quality_hours")

    flags_text = ", ".join(risky_flags) if risky_flags else "None clearly flagged."

    # Short numerical formatting
    def fmt(x, ndigits=2):
        if x is None:
            return "unknown"
        try:
            return f"{float(x):.{ndigits}f}"
        except Exception:
            return "unknown"

    profile_block = f"""
PATIENT METRICS
---------------
Chronotype label: {label}
Average sleep duration: {fmt(avg_sleep)} hours/night over {nights} nights
Average sleep midpoint (clock hours): {fmt(midpoint)}
Sleep regularity (0–1, higher = more regular): {fmt(consistency)}

Sleep debt:
- Total sleep debt (approx): {fmt(total_debt)} hours
- Average nightly deficit: {fmt(avg_deficit)} hours/night
- Severity: {severity}

Activity / behavior:
- Median daily steps: {fmt(steps, 0)}
- Median sedentary minutes per day: {fmt(sedentary, 0)}
- Median very active minutes per day: {fmt(very_active, 0)}
- Estimated sleep efficiency: {fmt(eff)}
- Risky behaviors: {flags_text}

Model-based estimate of healthy sleep: {fmt(ml_hours)} hours/night
"""

    instruction_block = """
You are a friendly, non-judgmental sleep doctor.

Using ONLY the information above, write a clinical-style sleep improvement plan
of about 200 words.

Requirements:
- Tone: supportive, practical, no blame.
- Do NOT mention devices, models, AI, or training data.
- Structure the answer with these clear headings (plain text, no bullet points):

1) CLINICAL SUMMARY
2) LIKELY CONTRIBUTORS
3) 7-DAY PLAN
4) WHEN TO SEEK MEDICAL HELP

Write in continuous paragraphs under each heading (no lists, no markdown).
Focus especially on sleep debt, short sleep, sedentary time, and chronotype alignment.
"""

    full_prompt = profile_block + "\n" + instruction_block
    return full_prompt.strip()


# -------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------

def generate_recommendation_text(prompt: str) -> str:
    tokenizer, model = load_tinyllama()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(_DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Many small LMs echo the prompt. Strip it if present.
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    return text.strip()


# -------------------------------------------------------------------
# MAIN PER-PERSON PIPELINE
# -------------------------------------------------------------------

def run_agent4_for_person(person_id: int) -> None:
    chrono, behavioral, debt, ml_pred = load_person_context(person_id)
    snapshot = build_snapshot(person_id, chrono, behavioral, debt, ml_pred)
    summary_line = build_summary_line(snapshot)
    prompt = build_prompt(snapshot)
    report_text = generate_recommendation_text(prompt)

    folder = ensure_person_folder(person_id)
    json_path = os.path.join(folder, "slm_recommendation.json")
    txt_path = os.path.join(folder, "slm_recommendation.txt")

    payload = {
        "person_id": int(person_id),
        "model_name": MODEL_NAME,
        "generated_at": datetime.utcnow().isoformat(),
        "sleep_quality_hours_estimate": snapshot.get("predicted_sleep_quality_hours"),
        "input_snapshot": snapshot,
        "summary_line": summary_line,
        "report_text": report_text,
    }

    payload = sanitize(payload)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_line.strip() + "\n\n" + report_text)

    print(f"Generated recommendation for Id={person_id}")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {txt_path}")


# -------------------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------------------

def list_available_person_ids() -> list:
    if not os.path.isdir(BASE_REPORT_DIR):
        return []
    ids = []
    for name in os.listdir(BASE_REPORT_DIR):
        path = os.path.join(BASE_REPORT_DIR, name)
        if os.path.isdir(path) and name.isdigit():
            ids.append(int(name))
    return sorted(ids)


def main():
    print("\n=== Running Agent-4 v3.1 (TinyLlama Recommendation Engine) ===\n")

    if len(sys.argv) > 1:
        try:
            pid = int(sys.argv[1])
        except ValueError:
            raise ValueError("Usage: python agents/agent4_slm_recommender.py <person_id>")
        run_agent4_for_person(pid)
    else:
        ids = list_available_person_ids()
        if not ids:
            print(
                "No person folders found under outputs/person_reports.\n"
                "Run Agent-1, Agent-2, and Agent-3 first."
            )
            return
        print(f"Found {len(ids)} person(s): {ids}")
        for pid in ids:
            try:
                run_agent4_for_person(pid)
            except Exception as e:
                print(f"Skipping Id={pid} due to error: {e}")

    print("\n=== Agent-4 complete ===")


if __name__ == "__main__":
    main()
