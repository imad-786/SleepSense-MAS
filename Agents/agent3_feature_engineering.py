"""
Agent-3 (Part 1) — Person-Level Feature Engineering for ML

Input  : data/cleaned_data.csv  (from Agent-1, Fitbit-based)
Output : outputs/person_features.csv
         outputs/person_list.json

Each row in person_features.csv = one person (Id) with aggregated features:
- Sleep quantity / quality stats
- Sleep regularity
- Activity & sedentary balance
- Simple trend features
"""

import os
import json
from typing import Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
CLEANED_DATA_PATH = os.path.join("data", "cleaned_data.csv")
OUTPUT_DIR = "outputs"
FEATURES_CSV_PATH = os.path.join(OUTPUT_DIR, "person_features.csv")
PERSON_LIST_PATH = os.path.join(OUTPUT_DIR, "person_list.json")

# We expect these columns from Agent-1
REQUIRED_COLUMNS = [
    "Id",
    "date",
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


# -------------------------------------------------------------------
# 1. LOAD & VALIDATE
# -------------------------------------------------------------------
def load_cleaned_data() -> pd.DataFrame:
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(
            f"cleaned_data.csv not found at '{CLEANED_DATA_PATH}'. "
            f"Run Agent-1 (Fitbit preprocessing) first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    print("✅ cleaned_data.csv loaded")

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"cleaned_data.csv is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Ensure types
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
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with no date or no sleep_hours
    before = len(df)
    df = df.dropna(subset=["date", "sleep_hours"])
    after = len(df)
    if after < before:
        print(f"⚠️ Dropped {before - after} rows due to missing date/sleep_hours")

    return df


# -------------------------------------------------------------------
# 2. PERSON-LEVEL FEATURE ENGINEERING
# -------------------------------------------------------------------
def compute_sleep_regular_cv(sleep_hours: pd.Series) -> float:
    """Coefficient of variation for sleep hours (std / mean)."""
    sleep_hours = sleep_hours.dropna()
    if len(sleep_hours) == 0:
        return np.nan
    mean = sleep_hours.mean()
    std = sleep_hours.std()
    if mean <= 0:
        return np.nan
    return float(std / mean)


def compute_weekend_diff(g: pd.DataFrame) -> float:
    """
    Weekend vs weekday sleep difference:
    mean(sleep_hours on Sat/Sun) - mean(sleep_hours on Mon–Fri)
    """
    if "date" not in g.columns:
        return np.nan

    dow = g["date"].dt.dayofweek  # Monday=0, Sunday=6
    weekend_mask = dow >= 5

    weekend = g.loc[weekend_mask, "sleep_hours"].dropna()
    weekday = g.loc[~weekend_mask, "sleep_hours"].dropna()

    if len(weekend) == 0 or len(weekday) == 0:
        return np.nan

    return float(weekend.mean() - weekday.mean())


def compute_sleep_trend(g: pd.DataFrame) -> float:
    """
    Simple linear trend of sleep_hours over time (hours per day).
    Uses numpy.polyfit on (day_index, sleep_hours).
    """
    g = g.sort_values("date")
    y = g["sleep_hours"].dropna()
    if len(y) < 3:
        return np.nan

    # Align x with y
    x = np.arange(len(y))
    try:
        slope, _ = np.polyfit(x, y.values, deg=1)
        return float(slope)
    except Exception:
        return np.nan


def compute_last7_vs_overall(g: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns (last7_mean_sleep, overall_mean_sleep).
    """
    g = g.sort_values("date")
    y = g["sleep_hours"].dropna()
    if len(y) == 0:
        return np.nan, np.nan

    overall_mean = float(y.mean())
    last7 = y.tail(7)
    last7_mean = float(last7.mean()) if len(last7) > 0 else np.nan
    return last7_mean, overall_mean


def build_person_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Id and compute person-level features.
    Returns a DataFrame with one row per Id.
    """
    features = []

    for pid, g in df.groupby("Id"):
        # sort by date for time-based features
        g = g.sort_values("date")

        n_days = int(len(g))
        if n_days == 0:
            continue

        # Basic sleep stats
        sleep_hours = g["sleep_hours"]
        sleep_eff = g["sleep_efficiency"]

        avg_sleep_hours = float(sleep_hours.mean())
        std_sleep_hours = float(sleep_hours.std(ddof=1)) if n_days > 1 else 0.0
        min_sleep_hours = float(sleep_hours.min())
        max_sleep_hours = float(sleep_hours.max())

        avg_eff = float(sleep_eff.mean())
        std_eff = float(sleep_eff.std(ddof=1)) if n_days > 1 else 0.0

        regular_cv = compute_sleep_regular_cv(sleep_hours)

        # Activity stats
        avg_steps = float(g["TotalSteps"].mean())
        avg_sedentary = float(g["SedentaryMinutes"].mean())
        avg_very_active = float(g["VeryActiveMinutes"].mean())
        avg_fairly_active = float(g["FairlyActiveMinutes"].mean())
        avg_lightly_active = float(g["LightlyActiveMinutes"].mean())
        avg_calories = float(g["Calories"].mean())

        activity_minutes = avg_very_active + avg_fairly_active + avg_lightly_active
        activity_to_sedentary_ratio = (
            float(activity_minutes / avg_sedentary) if avg_sedentary > 0 else np.nan
        )

        # Weekend vs weekday difference
        weekend_sleep_diff = compute_weekend_diff(g)

        # Trend & recent vs overall
        sleep_trend = compute_sleep_trend(g)
        last7_mean, overall_mean = compute_last7_vs_overall(g)
        last7_minus_overall = (
            float(last7_mean - overall_mean)
            if not (np.isnan(last7_mean) or np.isnan(overall_mean))
            else np.nan
        )

        features.append(
            {
                "Id": int(pid),
                "n_days": n_days,

                # Sleep quantity / quality
                "avg_sleep_hours": avg_sleep_hours,
                "std_sleep_hours": std_sleep_hours,
                "min_sleep_hours": min_sleep_hours,
                "max_sleep_hours": max_sleep_hours,
                "avg_sleep_efficiency": avg_eff,
                "std_sleep_efficiency": std_eff,

                # Regularity & chronobiology-ish
                "sleep_regular_cv": regular_cv,
                "weekend_weekday_sleep_diff": weekend_sleep_diff,
                "sleep_trend_hours_per_day": sleep_trend,
                "last7_mean_sleep_hours": last7_mean,
                "last7_minus_overall_sleep": last7_minus_overall,

                # Activity profile
                "avg_steps": avg_steps,
                "avg_sedentary_minutes": avg_sedentary,
                "avg_very_active_minutes": avg_very_active,
                "avg_fairly_active_minutes": avg_fairly_active,
                "avg_lightly_active_minutes": avg_lightly_active,
                "avg_calories": avg_calories,
                "activity_to_sedentary_ratio": activity_to_sedentary_ratio,
            }
        )

    features_df = pd.DataFrame(features)
    return features_df


# -------------------------------------------------------------------
# 3. SAVE FEATURES + PERSON LIST
# -------------------------------------------------------------------
def save_outputs(features_df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV for ML / debugging
    features_df.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"✅ Person-level features saved to {FEATURES_CSV_PATH}")

    # Person list for Streamlit dropdown, etc.
    # Convert to Python ints so json doesn't complain about int64
    ids = sorted(int(i) for i in features_df["Id"].unique())
    with open(PERSON_LIST_PATH, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"✅ Person list saved to {PERSON_LIST_PATH} ({len(ids)} people)")


# -------------------------------------------------------------------
# 4. MAIN RUNNER
# -------------------------------------------------------------------
def run_agent3_feature_engineering():
    print("\n==============================")
    print("  RUNNING AGENT-3 (FEATURES)")
    print("  Fitbit Person-Level Features")
    print("==============================\n")

    df = load_cleaned_data()
    features_df = build_person_features(df)

    if features_df.empty:
        print("❌ No features generated. Check cleaned_data.csv contents.")
        return

    save_outputs(features_df)

    print("\n=== Quick Preview of Engineered Features ===")
    print(features_df.head())
    print(f"\nTotal people with features: {len(features_df)}")


if __name__ == "__main__":
    run_agent3_feature_engineering()
