# agents/agent1_preprocessing.py
import os
import json
import pandas as pd
import numpy as np

RAW_ROOT = os.path.join("data", "raw")
CLEAN_DATA_PATH = os.path.join("data", "cleaned_data.csv")
PERSON_LIST_PATH = os.path.join("data", "person_list.json")


# ---------- Helpers to load from multiple Fitbit folders ----------

def _get_fitbit_folders():
    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(
            f"RAW_ROOT folder not found: {RAW_ROOT}\n"
            "Create data/raw/ and put your Fitabase folders inside it."
        )

    folders = [
        os.path.join(RAW_ROOT, d)
        for d in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, d))
    ]

    if not folders:
        raise FileNotFoundError(
            f"No Fitbit folders found inside {RAW_ROOT}. "
            "Expected something like 'Fitabase Data 3.12.16-4.11.16/'."
        )

    return folders


def _concat_from_all_folders(filename):
    folders = _get_fitbit_folders()
    dfs = []

    for folder in folders:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            print(f"  ✓ Found {filename} in {folder}")
            df = pd.read_csv(path)
            df["__source_folder"] = os.path.basename(folder)
            dfs.append(df)
        else:
            print(f"  • {filename} NOT found in {folder} (skipping)")

    if not dfs:
        raise FileNotFoundError(
            f"{filename} not found in ANY folder under {RAW_ROOT}."
        )

    return pd.concat(dfs, ignore_index=True)


# ---------- Sleep aggregation (minuteSleep_merged.csv) ----------

def build_sleep_daily():
    print("\n[1] Loading minute-level sleep data (minuteSleep_merged.csv)...")
    sleep_raw = _concat_from_all_folders("minuteSleep_merged.csv")

    # Canonicalize columns
    sleep_raw.columns = [c.strip().lower() for c in sleep_raw.columns]

    # Expect columns: id, date, value (1=asleep, 2=restless, 3=awake)
    if "id" not in sleep_raw.columns:
        raise KeyError("minuteSleep_merged.csv is missing an 'Id' column.")
    # Find a datetime column
    datetime_col = None
    for cand in ["date", "datetime", "sleepday", "sleep_day"]:
        if cand in sleep_raw.columns:
            datetime_col = cand
            break
    if datetime_col is None:
        raise KeyError(
            "Could not find a date/datetime column in minuteSleep_merged.csv "
            "(looked for: date, datetime, sleepday, sleep_day)."
        )
    if "value" not in sleep_raw.columns:
        raise KeyError(
            "minuteSleep_merged.csv is missing 'value' column "
            "(1=Asleep, 2=Restless, 3=Awake)."
        )

    # Parse datetime and extract just the calendar date
    sleep_raw[datetime_col] = pd.to_datetime(
        sleep_raw[datetime_col], errors="coerce"
    )
    sleep_raw = sleep_raw.dropna(subset=[datetime_col])
    sleep_raw["date_only"] = sleep_raw[datetime_col].dt.date

    # Flags for minute states
    sleep_raw["is_asleep"] = sleep_raw["value"].isin([1, 2]).astype(int)
    sleep_raw["is_awake"] = (sleep_raw["value"] == 3).astype(int)

    grouped = (
        sleep_raw.groupby(["id", "date_only"], as_index=False)[
            ["is_asleep", "is_awake"]
        ]
        .sum()
    )

    grouped.rename(columns={"id": "Id", "date_only": "date"}, inplace=True)
    grouped["sleep_minutes"] = grouped["is_asleep"]
    grouped["minutes_awake"] = grouped["is_awake"]
    grouped["total_time_in_bed_minutes"] = (
        grouped["sleep_minutes"] + grouped["minutes_awake"]
    )

    # Option B: (TotalTimeInBed - minutesAwake) / 60
    # This simplifies to "minutes actually asleep / 60"
    grouped["sleep_hours"] = (
        grouped["total_time_in_bed_minutes"] - grouped["minutes_awake"]
    ) / 60.0

    # Sleep efficiency: fraction of time in bed that is asleep
    grouped["sleep_efficiency"] = grouped.apply(
        lambda r: (
            r["sleep_minutes"] / r["total_time_in_bed_minutes"]
            if r["total_time_in_bed_minutes"] > 0
            else np.nan
        ),
        axis=1,
    )

    grouped = grouped.drop(columns=["is_asleep", "is_awake"])

    print(
        f"    → Built daily sleep summary for "
        f"{grouped['Id'].nunique()} users and {len(grouped)} person-days."
    )
    return grouped


# ---------- Activity aggregation (dailyActivity_merged.csv) ----------

def build_activity_daily():
    print("\n[2] Loading daily activity data (dailyActivity_merged.csv)...")
    act = _concat_from_all_folders("dailyActivity_merged.csv")

    # Canonicalize columns
    act.columns = [c.strip() for c in act.columns]

    # Normalize column names but keep the original Fitbit ones
    rename_map = {}
    for col in act.columns:
        if col.lower() == "id":
            rename_map[col] = "Id"
        elif col.lower() == "activitydate":
            rename_map[col] = "ActivityDate"
    if rename_map:
        act = act.rename(columns=rename_map)

    required_cols = [
        "Id",
        "ActivityDate",
        "TotalSteps",
        "VeryActiveMinutes",
        "FairlyActiveMinutes",
        "LightlyActiveMinutes",
        "SedentaryMinutes",
        "Calories",
    ]
    missing = [c for c in required_cols if c not in act.columns]
    if missing:
        raise KeyError(
            f"dailyActivity_merged.csv is missing required columns: {missing}"
        )

    act["ActivityDate"] = pd.to_datetime(
        act["ActivityDate"], errors="coerce"
    )
    act = act.dropna(subset=["ActivityDate"])
    act["date"] = act["ActivityDate"].dt.date

    keep_cols = [
        "Id",
        "date",
        "TotalSteps",
        "VeryActiveMinutes",
        "FairlyActiveMinutes",
        "LightlyActiveMinutes",
        "SedentaryMinutes",
        "Calories",
    ]
    act = act[keep_cols].copy()

    print(
        f"    → Loaded daily activity for "
        f"{act['Id'].nunique()} users and {len(act)} person-days."
    )
    return act


# ---------- Combine sleep + activity ----------

def build_cleaned_dataset():
    sleep_daily = build_sleep_daily()
    activity_daily = build_activity_daily()

    print("\n[3] Merging sleep + activity per Id + date...")
    merged = pd.merge(
        sleep_daily,
        activity_daily,
        on=["Id", "date"],
        how="inner",  # only days where we have both sleep & activity
    )

    # Sort for sanity
    merged = merged.sort_values(["Id", "date"]).reset_index(drop=True)

    # Final column order
    final_cols = [
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
    merged = merged[final_cols]

    print(
        f"    → Final cleaned dataset: {len(merged)} rows, "
        f"{merged['Id'].nunique()} unique users."
    )
    return merged


def save_outputs(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\n[4] Saved cleaned data to {CLEAN_DATA_PATH}")

    # Build person list for UI
    ids = sorted(df["Id"].astype(str).unique().tolist())
    with open(PERSON_LIST_PATH, "w") as f:
        json.dump(ids, f, indent=2)
    print(f"    → Saved {len(ids)} unique Ids to {PERSON_LIST_PATH}")


def run_agent1():
    print(
        "\n==============================\n"
        "     RUNNING AGENT-1 (NEW)\n"
        "   Fitbit Person-Level Cleaner\n"
        "=============================="
    )
    df = build_cleaned_dataset()
    save_outputs(df)
    print("\nAgent-1 completed successfully.\n")


if __name__ == "__main__":
    run_agent1()
