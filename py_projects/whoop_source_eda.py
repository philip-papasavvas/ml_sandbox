"""
Exploratory analysis and linear modelling of WHOOP sleep data.

The goal is to predict the WHOOP "Sleep Score" from the other numeric sleep
metrics and to use the fitted linear model's coefficients as a crude feature
importance ranking. Because the features are standardised first, the magnitude
of each coefficient is directly comparable, which lets us drop the
least-important features and check whether a smaller model performs comparably.

Note: the WHOOP export is personal data and is *not* committed to this
repository. Point ``WHOOP_DATA_DIR`` at a directory containing ``recoveries.csv``
to run this script, e.g.::

    WHOOP_DATA_DIR=/path/to/whoop_export python py_projects/whoop_source_eda.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESPONSE_VARIABLE = "Sleep Score"
# Coefficients (on standardised features) below this magnitude are treated as
# unimportant and dropped from the reduced model.
IMPORTANCE_THRESHOLD = 10.0


def load_sleep_data(data_dir: str) -> pd.DataFrame:
    """Load the WHOOP recoveries export and return the cleaned numeric sleep data.

    Keeps only sleep-related columns, drops rows with missing values, and
    returns the numeric subset ready for modelling.
    """
    recovery_df = pd.read_csv(os.path.join(data_dir, "recoveries.csv"))
    print(f"Loaded recoveries: {recovery_df.shape[0]} rows, {recovery_df.shape[1]} columns")

    sleep_columns = [c for c in recovery_df.columns if "sleep" in c.lower()]
    sleep_data = recovery_df[["Date", "Day of Week"] + sleep_columns]

    numeric_sleep_data = sleep_data.select_dtypes(exclude=object)
    cleaned = numeric_sleep_data.dropna()
    dropped = len(numeric_sleep_data) - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} row(s) containing missing values")
    return cleaned


def report_model(name: str, model: LinearRegression, X_test, y_test) -> None:
    """Print the MSE and R^2 of a fitted model on the held-out test set."""
    y_pred = model.predict(X_test)
    print(f"[{name}] Mean squared error:           {mean_squared_error(y_test, y_pred):.4f}")
    print(f"[{name}] Coefficient of determination: {r2_score(y_test, y_pred):.4f} (1.0 is perfect)")


def main() -> None:
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 600)

    data_dir = os.environ.get("WHOOP_DATA_DIR")
    if not data_dir or not os.path.isdir(data_dir):
        sys.exit(
            "WHOOP data not found. Set WHOOP_DATA_DIR to a directory containing "
            "recoveries.csv (this personal export is not committed to the repo)."
        )

    sleep_df = load_sleep_data(data_dir)
    print(sleep_df.describe().T)

    # Split features (X) from the target (y).
    feature_columns = [c for c in sleep_df.columns if c != RESPONSE_VARIABLE]
    X = sleep_df[feature_columns]
    y = sleep_df[RESPONSE_VARIABLE]

    # Standardise the features so that coefficient magnitudes are comparable
    # and can be used as a feature-importance proxy.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=1
    )

    # --- Full model: all features ------------------------------------------
    full_model = LinearRegression().fit(X_train, y_train)
    report_model("full", full_model, X_test, y_test)

    importance_df = (
        pd.DataFrame({"importance": full_model.coef_}, index=feature_columns)
        .sort_values("importance", key=np.abs, ascending=False)
    )
    print("\nFeature importance (|coefficient| on standardised features):")
    print(importance_df)

    # --- Reduced model: keep only the most important features --------------
    # Drop features whose coefficient magnitude falls below the threshold, then
    # refit on that subset so the comparison is genuine.
    keep_mask = np.abs(full_model.coef_) > IMPORTANCE_THRESHOLD
    kept_features = [f for f, keep in zip(feature_columns, keep_mask) if keep]
    print(f"\nReduced model keeps {len(kept_features)}/{len(feature_columns)} features: {kept_features}")

    if not kept_features:
        print(f"No feature exceeds the importance threshold ({IMPORTANCE_THRESHOLD}); "
              "skipping the reduced model.")
        return

    reduced_model = LinearRegression().fit(X_train[:, keep_mask], y_train)
    report_model("reduced", reduced_model, X_test[:, keep_mask], y_test)


if __name__ == "__main__":
    main()
