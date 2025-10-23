import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
import math
import lightgbm as lgb

NUMERIC_FEATS = [
    "latitude","longitude","accommodates","bedrooms","beds","bathrooms",
    "review_scores_rating","number_of_reviews","host_listings_count",
    "amenities_count","host_tenure_days","has_picture","occupancy_rate_mean"
]

CAT_FEATS = [
    "neighbourhood_cleansed"
]  # room_type_* already one-hot from feature builder

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help="City folder under ./data, e.g., nyc")
    return ap.parse_args()

def rmsle(y_true, y_pred):
    # Root Mean Squared Log Error (on price)
    return math.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def main():
    args = parse_args()
    feats_fp = Path("data") / args.city / "features.csv"
    if not feats_fp.exists():
        raise FileNotFoundError(f"Missing {feats_fp}. Run feature builder first.")

    df = pd.read_csv(feats_fp)

    # Keep rows with target
    df = df.dropna(subset=["target_price"])

    # Basic feature set
    X = df[NUMERIC_FEATS + CAT_FEATS + [c for c in df.columns if c.startswith("room_type_")]]
    y = df["target_price"].astype(float)

    # Log-transform target for stability
    y_log = np.log1p(y)

    # Preprocess
    num_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_proc, NUMERIC_FEATS),
        ("cat", cat_proc, CAT_FEATS),
        # room_type_* are already numeric dummies; pass-through automatically
    ], remainder="passthrough")

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

    pipe.fit(X_train, y_train)

    y_pred_log = pipe.predict(X_val)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(np.expm1(y_val), y_pred)
    rmse = math.sqrt(mean_squared_error(np.expm1(y_val), y_pred))
    rmlse = rmsle(np.expm1(y_val), y_pred)

    print(f"Validation MAE:  {mae:,.2f}")
    print(f"Validation RMSE: {rmse:,.2f}")
    print(f"Validation RMSLE:{rmlse:,.4f}")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_fp = out_dir / f"price_model_{args.city}.joblib"
    dump(pipe, model_fp)
    print(f"Saved model to {model_fp.resolve()}")

if __name__ == "__main__":
    main()
