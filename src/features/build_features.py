import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help="City folder under ./data, e.g., boston")
    return ap.parse_args()

def clean_price_series(s):
    # Handles strings like "$123.00"
    return (
        s.astype(str)
         .str.replace("$","", regex=False)
         .str.replace(",","", regex=False)
         .astype(float)
    )

def engineer_listing_features(df):
    # Minimal, robust features. Add more as you iterate.
    # Coerce numeric columns safely
    for col in ["accommodates","bedrooms","beds","bathrooms","review_scores_rating","number_of_reviews","host_listings_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Amenities count
    if "amenities" in df.columns:
        df["amenities_count"] = df["amenities"].fillna("").apply(lambda x: len(str(x).strip("{}").split(",")) if pd.notnull(x) else 0)
    else:
        df["amenities_count"] = 0

    # Room type one-hot
    if "room_type" in df.columns:
        room_dummies = pd.get_dummies(df["room_type"], prefix="room_type")
        df = pd.concat([df, room_dummies], axis=1)

   # Host tenure (days) — make BOTH sides timezone-aware (UTC) to avoid tz errors
    if "host_since" in df.columns:
        df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce", utc=True)
        now_utc = pd.Timestamp.now(tz="UTC").normalize()
        df["host_tenure_days"] = (now_utc - df["host_since"]).dt.days
    else:
        df["host_tenure_days"] = np.nan

    # Photo count proxy (some datasets have 'picture_url'; many have direct 'picture_url' or derived fields)
    if "picture_url" in df.columns:
        df["has_picture"] = (~df["picture_url"].isna()).astype(int)
    else:
        df["has_picture"] = 1  # default assume yes

    # Keep coordinates if present
    keep_cols = [
        "id", "latitude", "longitude", "neighbourhood_cleansed",
        "accommodates","bedrooms","beds","bathrooms",
        "review_scores_rating","number_of_reviews","host_listings_count",
        "amenities_count","host_tenure_days","has_picture"
    ] + [c for c in df.columns if c.startswith("room_type_")]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].dropna(subset=["latitude","longitude","accommodates"], how="any")

def build_calendar_features(cal_df):
    # Expect columns: listing_id, date, price, available
    # Create simple seasonal features: month, weekday; occupancy proxy from available flag
    cal_df["date"] = pd.to_datetime(cal_df["date"], errors="coerce")
    cal_df["month"] = cal_df["date"].dt.month
    cal_df["weekday"] = cal_df["date"].dt.weekday

    if "available" in cal_df.columns:
        # InsideAirbnb calendar has available = 't' means free, 'f' means booked
        cal_df["booked"] = (cal_df["available"].astype(str).str.lower() == "f").astype(int)
    else:
        cal_df["booked"] = np.nan

    # price cleaning
    if "price" in cal_df.columns:
        cal_df["price_clean"] = (
            cal_df["price"]
            .astype(str).str.replace("$","", regex=False).str.replace(",","", regex=False)
            .astype(float)
        )
    else:
        cal_df["price_clean"] = np.nan

    # Aggregate per listing: occupancy over last 30 days (if data spans enough)
    occ = (cal_df
           .groupby("listing_id", as_index=False)["booked"]
           .mean()
           .rename(columns={"booked":"occupancy_rate_mean"}))

    return occ

def main():
    args = parse_args()
    data_city_dir = Path("data") / args.city

    listings_fp = data_city_dir / "listings.csv"
    calendar_fp = data_city_dir / "calendar.csv"

    if not listings_fp.exists():
        raise FileNotFoundError(f"Missing {listings_fp}. Put your InsideAirbnb listings.csv there.")
    if not calendar_fp.exists():
        raise FileNotFoundError(f"Missing {calendar_fp}. Put your InsideAirbnb calendar.csv there.")

    listings = pd.read_csv(listings_fp, low_memory=False)
    cal = pd.read_csv(calendar_fp, low_memory=False)

    # Some datasets have 'id' for listing id; calendar often uses 'listing_id'
    if "id" not in listings.columns:
        # try alternative
        id_col = [c for c in listings.columns if "id" in c.lower()]
        if not id_col:
            raise ValueError("Could not find listing id column in listings.csv")
        listings = listings.rename(columns={id_col[0]:"id"})

    if "listing_id" not in cal.columns:
        # try to guess; sometimes it's 'listing_id' already; else there might be 'id'
        if "id" in cal.columns:
            cal = cal.rename(columns={"id":"listing_id"})

    feat_listings = engineer_listing_features(listings)
    occ = build_calendar_features(cal)

    # Join
    feats = feat_listings.merge(occ, left_on="id", right_on="listing_id", how="left")
    feats = feats.drop(columns=["listing_id"], errors="ignore")

    # Merge target price from listings.csv (if present)
    if "price" in listings.columns:
        feats = feats.merge(
            listings[["id","price"]].assign(price=lambda d: d["price"].astype(str).str.replace("$","", regex=False).str.replace(",","", regex=False).astype(float)),
            on="id", how="left", suffixes=("","_listings")
        )
        # Use listings price as target (snapshot). Calendar prices vary day-to-day; start simple.
        feats["target_price"] = feats["price"]
        feats = feats.drop(columns=["price"], errors="ignore")
    else:
        feats["target_price"] = np.nan

    out_dir = Path("data") / args.city
    out_fp = out_dir / "features.csv"
    feats.to_csv(out_fp, index=False)
    print(f"Wrote features to {out_fp.resolve()} with shape {feats.shape}")

if __name__ == "__main__":
    main()
