import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

OUTPUT_DIR = "outputs"
PERSON_REPORTS_DIR = os.path.join(OUTPUT_DIR, "person_reports")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "person_features.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "ml_sleep_quality_summary.json")
CLUSTER_FILE = os.path.join(OUTPUT_DIR, "cluster_summary.json")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERSON_REPORTS_DIR, exist_ok=True)


def load_features():
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError("Missing person_features.csv — run Agent-3 feature engineering first.")
    return pd.read_csv(FEATURES_FILE)


def train_and_predict(df):
    target_col = "avg_sleep_hours"                # Sleep quality proxy
    feature_cols = [c for c in df.columns if c not in ["Id", target_col]]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(df[feature_cols])

    mse = mean_squared_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))

    return preds, mse, r2


def save_global_summary(df, predictions, mse, r2):
    summary = {
        "n_people": len(df),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "average_predicted_sleep": float(np.mean(predictions)),
        "min_predicted_sleep": float(np.min(predictions)),
        "max_predicted_sleep": float(np.max(predictions)),
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=4)


def save_per_person_predictions(df, predictions):
    for Id, pred in zip(df["Id"], predictions):
        person_dir = os.path.join(PERSON_REPORTS_DIR, str(int(Id)))
        os.makedirs(person_dir, exist_ok=True)

        data = {
            "person_id": int(Id),
            "predicted_sleep_quality_hours": float(pred),
        }

        file_path = os.path.join(person_dir, "ml_prediction.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)


def run_agent3():
    print("\n==============================")
    print(" RUNNING AGENT-3 (ML MODEL)")
    print("==============================\n")

    ensure_dirs()
    df = load_features()
    predictions, mse, r2 = train_and_predict(df)

    save_global_summary(df, predictions, mse, r2)
    save_per_person_predictions(df, predictions)

    print("Agent-3 completed successfully.")
    print("Per-person ML predictions saved inside outputs/person_reports/<Id>/ml_prediction.json")
    print("Global summary saved:", SUMMARY_FILE)


if __name__ == "__main__":
    run_agent3()
