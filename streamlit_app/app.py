import os
import sys
import json
import math
import subprocess
from typing import Dict, Any, List, Optional

import streamlit as st

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "person_reports")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON safely; return None if missing or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_person_ids() -> List[str]:
    """List person IDs from outputs/person_reports/<Id> folders."""
    if not os.path.isdir(REPORTS_DIR):
        return []
    ids = []
    for name in os.listdir(REPORTS_DIR):
        full = os.path.join(REPORTS_DIR, name)
        if os.path.isdir(full) and name.isdigit():
            ids.append(name)
    return sorted(ids)


def hour_float_to_hhmm(x: Optional[float]) -> str:
    """Convert hour-float (e.g. 3.5) to '03:30' 24h string."""
    if x is None:
        return "N/A"
    try:
        h = int(x)
        m = int(round((x - h) * 60))
        if m == 60:
            h = (h + 1) % 24
            m = 0
        return f"{h:02d}:{m:02d}"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------
# UI SETUP
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="SleepSense-MAS Dashboard",
    layout="wide",
)

st.title("SleepSense-MAS Dashboard")

person_ids = get_person_ids()
if not person_ids:
    st.error(
        "No person reports found.\n\n"
        "Make sure you have run Agent-1, Agent-2 and Agent-3 so that "
        "`outputs/person_reports/<Id>/...` exists."
    )
    st.stop()

selected_id = st.selectbox("Select Person (Fitbit Id)", person_ids)

person_dir = os.path.join(REPORTS_DIR, selected_id)
chronotype_path = os.path.join(person_dir, "chronotype.json")
behavioral_path = os.path.join(person_dir, "behavioral.json")
sleep_debt_path = os.path.join(person_dir, "sleep_debt.json")
ml_pred_path = os.path.join(person_dir, "ml_prediction.json")
slm_rec_path = os.path.join(person_dir, "slm_recommendation.json")

chronotype_data = load_json(chronotype_path)
behavioral_data = load_json(behavioral_path)
sleep_debt_data = load_json(sleep_debt_path)
ml_pred_data = load_json(ml_pred_path)
slm_rec_data = load_json(slm_rec_path)

tabs = st.tabs(
    ["Chronotype", "Behavioral Patterns", "Sleep Debt", "ML Prediction", "AI Recommendation"]
)

# ---------------------------------------------------------------------
# TAB 1 – CHRONOTYPE
# ---------------------------------------------------------------------
with tabs[0]:
    st.header("Chronotype Analysis")

    if not chronotype_data or "chronotype" not in chronotype_data:
        st.warning("No chronotype data available for this person. Make sure Agent-2 has run.")
    else:
        c = chronotype_data["chronotype"]
        label = c.get("label", "Unknown")
        avg_midpoint_hour = c.get("avg_midpoint_hour")
        avg_sleep_hours = c.get("avg_sleep_hours")
        consistency = c.get("consistency_score")
        nights_used = c.get("nights_used")

        col1, col2, col3 = st.columns(3)
        col1.metric("Chronotype", label)
        col2.metric("Avg Sleep Duration (hrs)", f"{avg_sleep_hours:.2f}" if isinstance(avg_sleep_hours, (int, float)) else "N/A")
        col3.metric("Consistency Score", f"{consistency:.2f}" if isinstance(consistency, (int, float)) else "N/A")

        st.markdown("---")
        st.subheader("Key Points")
        st.markdown(
            "\n".join(
                [
                    f"- **Average sleep midpoint:** {hour_float_to_hhmm(avg_midpoint_hour)}",
                    f"- **Nights used for analysis:** {nights_used}",
                    f"- **Chronotype label:** {label} (earlier midpoints → more morning type).",
                    "- Higher **consistency score** means your sleep and wake times are more regular.",
                ]
            )
        )

# ---------------------------------------------------------------------
# TAB 2 – BEHAVIORAL PATTERNS
# ---------------------------------------------------------------------
with tabs[1]:
    st.header("Behavioral Patterns")

    if not behavioral_data or "behavioral" not in behavioral_data:
        st.warning("No behavioral pattern data for this person. Make sure Agent-2 has run.")
    else:
        b = behavioral_data["behavioral"]
        steps = b.get("median_total_steps")
        sedentary = b.get("median_sedentary_minutes")
        very_active = b.get("median_very_active_minutes")
        calories = b.get("median_calories")
        sleep_hours = b.get("median_sleep_hours")
        sleep_eff = b.get("median_sleep_efficiency")
        risky_flags = b.get("risky_flags", [])

        col1, col2, col3 = st.columns(3)
        col1.metric("Median Daily Steps", f"{steps:,.0f}" if isinstance(steps, (int, float)) else "N/A")
        col2.metric("Median Sedentary Minutes", f"{sedentary:.0f}" if isinstance(sedentary, (int, float)) else "N/A")
        col3.metric("Median Very Active Minutes", f"{very_active:.1f}" if isinstance(very_active, (int, float)) else "N/A")

        st.markdown("---")
        st.subheader("Sleep & Activity Summary")
        lines = [
            f"- **Median sleep duration:** {sleep_hours:.2f} hours"
            if isinstance(sleep_hours, (int, float))
            else "- **Median sleep duration:** N/A",
            f"- **Median sleep efficiency:** {sleep_eff*100:.1f}%"
            if isinstance(sleep_eff, (int, float))
            else "- **Median sleep efficiency:** N/A",
            f"- **Median calories burned:** {calories:.0f}"
            if isinstance(calories, (int, float))
            else "- **Median calories burned:** N/A",
        ]
        st.markdown("\n".join(lines))

        st.markdown("---")
        st.subheader("Risky Behaviours")
        if not risky_flags:
            st.success("No risky behaviours flagged by Agent-2.")
        else:
            st.warning("The following behaviour patterns may be affecting sleep:")
            for flag in risky_flags:
                st.markdown(f"- {flag}")

# ---------------------------------------------------------------------
# TAB 3 – SLEEP DEBT
# ---------------------------------------------------------------------
with tabs[2]:
    st.header("Sleep Debt Analysis")

    if not sleep_debt_data or "sleep_debt" not in sleep_debt_data:
        st.warning("No sleep debt data available. Make sure Agent-2 has run.")
    else:
        s = sleep_debt_data["sleep_debt"]
        total_debt = s.get("total_sleep_debt_hours")
        avg_deficit = s.get("avg_daily_deficit_hours")
        severity = s.get("severity", "Unknown")
        rec = s.get("recovery_plan", {})
        extra_per_night = rec.get("extra_per_night_hours")
        nights_to_recover = rec.get("estimated_nights_to_recover")
        note = rec.get("note", "")
        days_considered = sleep_debt_data.get("days_considered")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Total Sleep Debt (hrs)",
            f"{total_debt:.1f}" if isinstance(total_debt, (int, float)) else "N/A",
        )
        col2.metric(
            "Average Daily Deficit (hrs)",
            f"{avg_deficit:.2f}" if isinstance(avg_deficit, (int, float)) else "N/A",
        )
        col3.metric("Severity", severity)

        st.markdown("---")
        st.subheader("Recovery Plan")
        bullets = [
            f"- **Days considered:** {days_considered}" if days_considered is not None else "- **Days considered:** N/A",
            f"- **Extra sleep per night:** {extra_per_night:.1f} hours"
            if isinstance(extra_per_night, (int, float))
            else "- **Extra sleep per night:** N/A",
            f"- **Estimated nights to recover:** {nights_to_recover}"
            if isinstance(nights_to_recover, (int, float))
            else "- **Estimated nights to recover:** N/A",
        ]
        if note:
            bullets.append(f"- **Clinician note:** {note}")
        st.markdown("\n".join(bullets))

# ---------------------------------------------------------------------
# TAB 4 – ML PREDICTION
# ---------------------------------------------------------------------
with tabs[3]:
    st.header("ML Sleep Quality Prediction")

    if not ml_pred_data or "predicted_sleep_quality_hours" not in ml_pred_data:
        st.warning("No ML prediction available. Make sure Agent-3 has run.")
    else:
        pred_hours = ml_pred_data.get("predicted_sleep_quality_hours")
        st.subheader("Predicted Sleep Quality (per night)")
        if isinstance(pred_hours, (int, float)):
            st.metric("Predicted good-quality sleep", f"{pred_hours:.2f} hours")
            st.markdown(
                "\n".join(
                    [
                        "- This is the **model’s estimate** of how many hours of reasonably good-quality sleep "
                        "you are likely to get per night given your current patterns.",
                        "- Use this as a **trend indicator**, not as a medical diagnosis.",
                    ]
                )
            )
        else:
            st.write("Prediction not available or invalid.")

# ---------------------------------------------------------------------
# TAB 5 – AI RECOMMENDATION (TinyLlama)
# ---------------------------------------------------------------------
with tabs[4]:
    st.header("AI Recommendation (TinyLlama)")

    st.markdown(
        "This section uses a small local language model (TinyLlama) to turn the metrics from "
        "Agents 2–3 into a clinician-style recommendation."
    )

    # Button to run Agent-4 from Streamlit
    if st.button("Run Agent-4 Now (TinyLlama)"):
        st.info("Running Agent-4… this may take a while on CPU.")
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT_DIR, "agents", "agent4_slm_recommender.py")],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                st.success("Agent-4 completed. Reloading recommendation for this person.")
                # Reload JSON after run
                slm_rec_data = load_json(slm_rec_path)
            else:
                st.error("Agent-4 failed. See log below:")
                st.code(result.stderr or result.stdout)
        except Exception as e:
            st.error(f"Error running Agent-4: {e}")

    # Display existing / newly generated recommendation
    if not slm_rec_data:
        st.warning("No TinyLlama recommendation found yet for this person.")
    else:
        st.success("TinyLlama recommendation available for this person.")

        summary_line = slm_rec_data.get("summary_line")
        if summary_line:
            st.subheader("One-line Summary")
            st.markdown(f"- {summary_line}")
        else:
            st.markdown("- *(No summary line stored in JSON.)*")

        snapshot = slm_rec_data.get("input_snapshot", {})
        st.markdown("---")
        st.subheader("Snapshot Used By AI")
        bullets = []

        chrono_label = snapshot.get("chronotype_label")
        if chrono_label:
            bullets.append(f"- **Chronotype:** {chrono_label}")

        avg_sleep = snapshot.get("avg_sleep_hours")
        if isinstance(avg_sleep, (int, float)):
            bullets.append(f"- **Average sleep duration:** {avg_sleep:.2f} hours/night")

        midpoint = snapshot.get("avg_midpoint_hour")
        if isinstance(midpoint, (int, float)):
            bullets.append(f"- **Typical sleep midpoint:** {hour_float_to_hhmm(midpoint)}")

        consistency = snapshot.get("chronotype_consistency")
        if isinstance(consistency, (int, float)):
            bullets.append(f"- **Sleep schedule regularity score:** {consistency:.2f}")

        total_debt = snapshot.get("sleep_debt_total_hours")
        if isinstance(total_debt, (int, float)):
            bullets.append(f"- **Total sleep debt:** {total_debt:.1f} hours")

        avg_def = snapshot.get("sleep_debt_avg_daily_deficit")
        if isinstance(avg_def, (int, float)):
            bullets.append(f"- **Average daily deficit:** {avg_def:.2f} hours/night")

        severity = snapshot.get("sleep_debt_severity")
        if severity:
            bullets.append(f"- **Sleep debt severity:** {severity}")

        risk_flags = snapshot.get("behavior_risky_flags") or snapshot.get("behavior_risky_flag") or []
        if isinstance(risk_flags, list) and risk_flags:
            bullets.append("- **Risky behaviours:**")
            for rf in risk_flags:
                bullets.append(f"  - {rf}")

        predicted_hours = slm_rec_data.get("predicted_sleep_quality_hours")
        if isinstance(predicted_hours, (int, float)):
            bullets.append(f"- **Model-estimated good sleep:** {predicted_hours:.2f} hours/night")

        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.markdown("*(No detailed snapshot fields available.)*")

        st.markdown("---")
        st.subheader("Full Recommendation Text")
        report_text = slm_rec_data.get("report_text")
        if report_text:
            # simple bullet-style paragraphs: split on newline / sentences if needed
            st.markdown(report_text)
        else:
            st.markdown("*(No detailed report text stored in JSON yet.)*")
