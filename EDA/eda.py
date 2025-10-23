
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_price_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("$", "", regex=False)
         .str.replace(",", "", regex=False)
         .astype(float)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="boston", help="City folder under ./data (default: boston)")
    args = ap.parse_args()

    proj = Path(__file__).resolve().parent
    data_city = proj / "data" / args.city
    plots = proj / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    listings_fp = data_city / "listings.csv"
    calendar_fp = data_city / "calendar.csv"
    if not listings_fp.exists() or not calendar_fp.exists():
        print(f"[Error] Expected files at:\n  {listings_fp}\n  {calendar_fp}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading listings: {listings_fp}")
    listings = pd.read_csv(listings_fp, low_memory=False)
    print(f"[INFO] Loading calendar: {calendar_fp}")
    calendar = pd.read_csv(calendar_fp, low_memory=False)

    # Clean and select basic fields
    if "price" in listings.columns:
        try:
            listings["price"] = clean_price_series(listings["price"])
        except Exception as e:
            print(f"[WARN] Failed to clean price: {e}")
    else:
        print("[WARN] price column not found in listings.csv")

    # --- Summary text ---
    summary_lines = []
    summary_lines.append(f"Listings shape: {listings.shape}")
    summary_lines.append(f"Calendar shape: {calendar.shape}")
    if "price" in listings.columns:
        summary_lines.append("Price describe:")
        summary_lines.append(str(listings["price"].describe(percentiles=[.25,.5,.75,.9,.95])))

    # --- Plot 1: Price distribution ---
    if "price" in listings.columns:
        plt.figure(figsize=(8,4))
        plt.hist(listings["price"].dropna(), bins=50)
        plt.xlim(0, min(1000, listings["price"].dropna().quantile(0.99)))
        plt.title("Distribution of Nightly Prices")
        plt.xlabel("Price")
        plt.ylabel("Count")
        (plots / "price_distribution.png").unlink(missing_ok=True)
        plt.savefig(plots / "price_distribution.png", dpi=200, bbox_inches="tight")
        plt.close()

    # --- Plot 2: Price vs room type (boxplot) ---
    if "price" in listings.columns and "room_type" in listings.columns:
        # Filter extreme outliers for readability
        cap = listings["price"].dropna().quantile(0.99)
        subset = listings[listings["price"] <= cap]
        # Matplotlib boxplot
        plt.figure(figsize=(8,4))
        groups = [g["price"].values for _, g in subset.groupby("room_type")]
        labels = [str(k) for k,_ in subset.groupby("room_type")]
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title("Price by Room Type")
        plt.ylabel("Price")
        (plots / "price_by_roomtype.png").unlink(missing_ok=True)
        plt.savefig(plots / "price_by_roomtype.png", dpi=200, bbox_inches="tight")
        plt.close()

    # --- Plot 3: Price vs accommodates (scatter) ---
    if "price" in listings.columns and "accommodates" in listings.columns:
        cap = listings["price"].dropna().quantile(0.99)
        subset = listings[listings["price"] <= cap]
        plt.figure(figsize=(8,4))
        plt.scatter(subset["accommodates"], subset["price"], alpha=0.4, s=10)
        plt.title("Price vs Accommodates")
        plt.xlabel("Accommodates")
        plt.ylabel("Price")
        (plots / "price_vs_accommodates.png").unlink(missing_ok=True)
        plt.savefig(plots / "price_vs_accommodates.png", dpi=200, bbox_inches="tight")
        plt.close()

    # --- Plot 4: Neighborhood comparison (top 10 by count) ---
    if "price" in listings.columns and "neighbourhood_cleansed" in listings.columns:
        vc = listings["neighbourhood_cleansed"].value_counts().head(10).index
        subset = listings[listings["neighbourhood_cleansed"].isin(vc)].copy()
        cap = subset["price"].dropna().quantile(0.99) if "price" in subset.columns else None
        if cap is not None:
            subset = subset[subset["price"] <= cap]
        medians = subset.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False)
        plt.figure(figsize=(10,4))
        plt.bar(medians.index.astype(str), medians.values)
        plt.xticks(rotation=45, ha="right")
        plt.title("Median Price by Top 10 Neighborhoods")
        plt.ylabel("Median Price")
        (plots / "price_by_neighbourhood_top10.png").unlink(missing_ok=True)
        plt.savefig(plots / "price_by_neighbourhood_top10.png", dpi=200, bbox_inches="tight")
        plt.close()

    # --- Calendar: monthly occupancy ---
    # InsideAirbnb calendar has available 't' (free) / 'f' (booked)
    if "date" in calendar.columns and "available" in calendar.columns:
        try:
            calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
            calendar["booked"] = calendar["available"].astype(str).str.lower().eq("f")
            monthly = calendar.groupby(calendar["date"].dt.month)["booked"].mean()
            plt.figure(figsize=(7,3))
            plt.bar(monthly.index.astype(str), monthly.values)
            plt.title("Average Monthly Occupancy (fraction booked)")
            plt.xlabel("Month")
            plt.ylabel("Occupancy")
            (plots / "monthly_occupancy.png").unlink(missing_ok=True)
            plt.savefig(plots / "monthly_occupancy.png", dpi=200, bbox_inches="tight")
            plt.close()
            summary_lines.append("Monthly occupancy (fraction booked):")
            summary_lines.append(str(monthly.round(3)))
        except Exception as e:
            print(f"[WARN] Calendar occupancy plot failed: {e}")

    # Save summary text
    (plots / "eda_summary.txt").write_text("\n".join(summary_lines))
    print("[OK] Wrote:")
    print(f"  - {plots/'price_distribution.png'}")
    print(f"  - {plots/'price_by_roomtype.png'}")
    print(f"  - {plots/'price_vs_accommodates.png'}")
    print(f"  - {plots/'price_by_neighbourhood_top10.png'}")
    print(f"  - {plots/'monthly_occupancy.png'}")
    print(f"  - {plots/'eda_summary.txt'}")

if __name__ == "__main__":
    main()
