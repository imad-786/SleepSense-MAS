# agents/agent2_person_level.py

"""
AGENT-2 (PERSON-LEVEL, FITBIT VERSION)

Reads:
    data/cleaned_data.csv

Expected columns:
    Id
    date
    sleep_minutes
    minutes_awake
    total_time_in_bed_minutes
    sleep_hours
    sleep_efficiency
    TotalSteps
    VeryActiveMinutes
    FairlyActiveMinutes
    LightlyActiveMinutes
    SedentaryMinutes
    Calories

Outputs (per person):
    outputs/person_reports/<Id>/chronotype.json
    outputs/person_reports/<Id>/behavioral.json
    outputs/person_reports/<Id>/sleep_debt.json

Global summary:
    outputs/person_level_summary.json
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------

DATA_PATH = os.path.join("data", "cleaned_data.csv")
OUTPUT_DIR = "outputs"

# New per-person reports root
PERSON_REPORTS_DIR = os.path.join(OUTPUT_DIR, "person_reports")

# Global summary for Agent-3 and Streamlit
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "person_level_summary.json")

RECOMMENDED_SLEEP_HOURS = 8.0
RECOVERY_EXTRA_PER_NIGHT = 0.5  # hours


# ---------------- UTIL ----------------

def sanitize(obj):
    """
    Recursively convert numpy / pandas types into plain Python
    so json.dump does not crash.
    """
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
    if isinstance(obj, (pd.Timestamp, )):
        return obj.isoformat()
    return obj


# ---------------- LOAD DATA ----------------

def load_cleaned_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"cleaned_data.csv not found at {DATA_PATH}. Run Agent-1 first."
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "Id", "date", "sleep_minutes", "minutes_awake",
        "total_time_in_bed_minutes", "sleep_hours", "sleep_efficiency",
        "TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
        "LightlyActiveMinutes", "SedentaryMinutes", "Calories"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"cleaned_data.csv is missing required columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "sleep_minutes",
        "minutes_awake",
        "total_time_in_bed_minutes",
        "sleep_hours",
        "sleep_efficiency",
        "TotalSteps",
        "VeryActiveMinutes",
        "FairlyActiveMinutes",
        "LightlyActiveMinutes",
        "SedentaryMinutes",
        "Calories",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Id"])

    return df


# ---------------- CHRONOTYPE ----------------

def compute_chronotype_for_person(person_df: pd.DataFrame) -> dict:
    """
    Compute chronotype using midpoint:
        midpoint_timestamp = date + sleep_hours/2
        midpoint_hour = hour part of that timestamp.
    """
    g = person_df.copy().sort_values("date")

    g = g.dropna(subset=["date", "sleep_hours"])
    n_nights = len(g)
    if n_nights == 0:
        return {
            "label": "Unknown",
            "avg_midpoint_hour": None,
            "consistency_score": 0.0,
            "avg_sleep_hours": None,
            "nights_used": 0,
        }

    midpoint_ts = g["date"] + pd.to_timedelta(g["sleep_hours"] / 2.0, unit="h")
    midpoint_hour = midpoint_ts.dt.hour + midpoint_ts.dt.minute / 60.0

    avg_mid = float(midpoint_hour.mean())
    std_mid = float(midpoint_hour.std()) if n_nights > 1 else 0.0

    # consistency score in [0,1]: std >= 4h -> ~0
    consistency = max(0.0, 1.0 - min(std_mid / 4.0, 1.0))

    # classify chronotype
    if avg_mid < 3.5:
        label = "Morning Type"
    elif avg_mid <= 5.5:
        label = "Intermediate"
    else:
        label = "Night Owl"

    avg_sleep_hours = float(g["sleep_hours"].mean())

    return {
        "label": label,
        "avg_midpoint_hour": avg_mid,
        "consistency_score": consistency,
        "avg_sleep_hours": avg_sleep_hours,
        "nights_used": int(n_nights),
    }


# ---------------- SLEEP DEBT ----------------

def compute_sleep_debt_for_person(person_df: pd.DataFrame) -> dict:
    g = person_df.copy()
    g = g.dropna(subset=["sleep_hours"])
    if len(g) == 0:
        return {
            "total_sleep_debt_hours": 0.0,
            "avg_daily_deficit_hours": 0.0,
            "severity": "Unknown",
            "recovery_plan": {
                "extra_per_night_hours": RECOVERY_EXTRA_PER_NIGHT,
                "estimated_nights_to_recover": 0,
                "note": "No valid sleep records available."
            },
            "days_considered": 0
        }

    g["daily_deficit"] = (RECOMMENDED_SLEEP_HOURS - g["sleep_hours"]).clip(lower=0.0)

    total_debt = float(g["daily_deficit"].sum())
    avg_deficit = float(g["daily_deficit"].mean()) if len(g) > 0 else 0.0

    if total_debt <= 7:
        severity = "Low"
    elif total_debt <= 21:
        severity = "Moderate"
    else:
        severity = "High"

    if total_debt > 0 and RECOVERY_EXTRA_PER_NIGHT > 0:
        nights = int(np.ceil(total_debt / RECOVERY_EXTRA_PER_NIGHT))
    else:
        nights = 0

    recovery_plan = {
        "extra_per_night_hours": RECOVERY_EXTRA_PER_NIGHT,
        "estimated_nights_to_recover": nights,
        "note": (
            f"Gradual repayment recommended. Aim for +{RECOVERY_EXTRA_PER_NIGHT}h "
            f"sleep per night until debt is repaid."
        ),
    }

    return {
        "total_sleep_debt_hours": total_debt,
        "avg_daily_deficit_hours": avg_deficit,
        "severity": severity,
        "recovery_plan": recovery_plan,
        "days_considered": int(len(g)),
    }


# ---------------- BEHAVIORAL PATTERNS ----------------

def compute_behavior_for_person(person_df: pd.DataFrame) -> dict:
    g = person_df.copy()

    med_steps = float(g["TotalSteps"].median(skipna=True)) if "TotalSteps" in g.columns else None
    med_sedentary = float(g["SedentaryMinutes"].median(skipna=True)) if "SedentaryMinutes" in g.columns else None
    med_very_active = float(g["VeryActiveMinutes"].median(skipna=True)) if "VeryActiveMinutes" in g.columns else None
    med_calories = float(g["Calories"].median(skipna=True)) if "Calories" in g.columns else None
    med_sleep_hours = float(g["sleep_hours"].median(skipna=True)) if "sleep_hours" in g.columns else None
    med_eff = float(g["sleep_efficiency"].median(skipna=True)) if "sleep_efficiency" in g.columns else None

    flags = []

    if med_sleep_hours is not None and med_sleep_hours < 7:
        flags.append("Short sleep duration (<7h median)")
    if med_eff is not None and med_eff < 0.85:
        flags.append("Low sleep efficiency (<85%)")
    if med_steps is not None and med_steps < 5000:
        flags.append("Low physical activity (median steps <5000)")
    if med_sedentary is not None and med_sedentary > 800:
        flags.append("High sedentary time (median >800 minutes)")

    return {
        "median_total_steps": med_steps,
        "median_sedentary_minutes": med_sedentary,
        "median_very_active_minutes": med_very_active,
        "median_calories": med_calories,
        "median_sleep_hours": med_sleep_hours,
        "median_sleep_efficiency": med_eff,
        "risky_flags": flags,
    }


def compute_global_correlations(df: pd.DataFrame) -> dict:
    """
    Correlations across the whole cohort between activity and sleep metrics.
    """
    corr = {}
    behavior_cols = ["TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "Calories"]
    sleep_cols = ["sleep_hours", "sleep_efficiency"]

    for b in behavior_cols:
        if b not in df.columns:
            continue
        for s in sleep_cols:
            if s not in df.columns:
                continue

            tmp = df[[b, s]].dropna()
            if len(tmp) < 5:
                value = None
            else:
                value = float(tmp[b].corr(tmp[s]))
            corr[f"{b}__vs__{s}"] = value

    return corr


# ---------------- MAIN RUNNER ----------------

def run_agent2():
    print("\n=== Running Agent-2 (PERSON-LEVEL MODE, FITBIT) ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERSON_REPORTS_DIR, exist_ok=True)

    df = load_cleaned_data()

    correlations = compute_global_correlations(df)

    per_person_summary = {}
    unique_ids = sorted(df["Id"].unique())

    for pid in unique_ids:
        person_df = df[df["Id"] == pid]

        chrono = compute_chronotype_for_person(person_df)
        debt = compute_sleep_debt_for_person(person_df)
        behavior = compute_behavior_for_person(person_df)

        # per-person directory: outputs/person_reports/<Id>/
        pid_str = str(int(pid))
        person_dir = os.path.join(PERSON_REPORTS_DIR, pid_str)
        os.makedirs(person_dir, exist_ok=True)

        # 1) chronotype.json
        chrono_payload = {
            "Id": int(pid),
            "generated_at": datetime.now().isoformat(),
            "chronotype": chrono,
        }
        with open(os.path.join(person_dir, "chronotype.json"), "w", encoding="utf-8") as f:
            json.dump(sanitize(chrono_payload), f, indent=4)

        # 2) behavioral.json
        behavior_payload = {
            "Id": int(pid),
            "generated_at": datetime.now().isoformat(),
            "behavioral": behavior,
        }
        with open(os.path.join(person_dir, "behavioral.json"), "w", encoding="utf-8") as f:
            json.dump(sanitize(behavior_payload), f, indent=4)

        # 3) sleep_debt.json
        debt_payload = {
            "Id": int(pid),
            "generated_at": datetime.now().isoformat(),
            "sleep_debt": debt,
        }
        with open(os.path.join(person_dir, "sleep_debt.json"), "w", encoding="utf-8") as f:
            json.dump(sanitize(debt_payload), f, indent=4)

        # summary for dashboards / ML
        per_person_summary[pid_str] = {
            "chronotype_label": chrono["label"],
            "sleep_debt_severity": debt["severity"],
            "avg_sleep_hours": chrono["avg_sleep_hours"],
            "risk_flags": behavior["risky_flags"],
        }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "people_count": int(len(unique_ids)),
        "correlations": sanitize(correlations),
        "per_person_summary": sanitize(per_person_summary),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Processed {len(unique_ids)} Id(s).")
    print(f"Per-person reports saved under: {PERSON_REPORTS_DIR}")
    print(f"Global summary saved to: {SUMMARY_PATH}")
    print("=== Agent-2 complete ===")


if __name__ == "__main__":
    run_agent2()
