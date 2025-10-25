# 🏠 Airbnb Price Optimizer (Boston)

**Goal:**  
Build a data-driven model that recommends optimal nightly prices for Airbnb listings using InsideAirbnb’s public dataset — helping hosts maximize revenue and guests find fair deals.

---

## 📘 Project Overview

| Stage | Description |
|:------|:-------------|
| **1. Data Collection** | Pulled Boston’s *listings.csv* and *calendar.csv* from [InsideAirbnb](https://insideairbnb.com/get-the-data/). |
| **2. Feature Engineering** | Cleaned price fields, extracted host & property features, merged occupancy signals from calendar data. |
| **3. Modeling** | Trained a LightGBM regressor on log-price with geographic, property, and quality indicators. |
| **4. Visualization** | Streamlit web app to recommend price and visualize a simple revenue curve. |
| **5. EDA Insights** | Analyzed distributions, room-type effects, and neighborhood trends to understand pricing dynamics. |

---

## 🧠 Key Insights (Boston Snapshot)

| Finding | Description |
|:--------|:-------------|
| 💰 **Entire homes** | Typically 2–3× the price of private rooms. |
| 🛏 **Accommodates** | Price rises roughly linearly with guest capacity up to ~6. |
| 📍 **Neighborhoods** | Downtown, Back Bay, and South End have the highest medians. |
| 📆 **Seasonality** | Occupancy peaks in June–August. |
| ⭐ **Reviews** | Higher review scores and host tenure correlate with higher prices. |

---

## 🧩 EDA Visuals

All charts are generated via `eda.py`.

| Visualization | File |
|:--------------|:-----|
| Nightly price distribution | `plots/price_distribution.png` |
| Price vs room type | `plots/price_by_roomtype.png` |
| Price vs accommodates | `plots/price_vs_accommodates.png` |
| Top-10 neighborhoods by price | `plots/price_by_neighbourhood_top10.png` |
| Monthly occupancy | `plots/monthly_occupancy.png` |

---

## ⚙️ Project Setup

```bash
# Clone and enter project
git clone <your_repo_url>
cd airbnb-price-optimizer

# Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download InsideAirbnb data
mkdir -p data/boston
# Place listings.csv and calendar.csv inside data/boston/

# Build features
python -m src.features.build_features --city boston

# Train model
python -m src.models.train_price_model --city boston

# Run Streamlit app
streamlit run app/streamlit_app.py
```

Outputs:
```
data/boston/features.csv
models/price_model_boston.joblib
plots/*.png
```

---

## 🌍 Streamlit App Features

- **Input:** listing details (location, room type, accommodates, amenities, etc.)  
- **Output:**  
  - Recommended nightly price  
  - Toy revenue curve  
  - Revenue-maximizing price suggestion  
  - (Optional future) Comparable listings nearby  

![Streamlit Demo](https://user-images.githubusercontent.com/placeholder/demo.png)

---

## 🧮 Model Details

| Component | Description |
|:-----------|:-------------|
| Algorithm | LightGBM Regressor |
| Target | Log(price) |
| Validation | 80/20 random split (upgradable to GroupKFold by neighborhood) |
| Metrics | MAE, RMSE, RMSLE |
| Storage | `models/price_model_boston.joblib` |

---

## 🧱 Directory Structure
```
airbnb-price-optimizer/
├── app/
│   └── streamlit_app.py
├── data/
│   └── boston/
│       ├── listings.csv
│       ├── calendar.csv
│       └── features.csv
├── notebooks/
│   └── EDA_template.ipynb
├── plots/
│   ├── price_distribution.png
│   ├── price_by_roomtype.png
│   ├── price_vs_accommodates.png
│   ├── price_by_neighbourhood_top10.png
│   └── monthly_occupancy.png
├── src/
│   ├── features/build_features.py
│   └── models/train_price_model.py
├── models/price_model_boston.joblib
├── eda.py
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

- 🔁 **Demand Model:** Predict occupancy from `calendar.csv` to generate true revenue curves.  
- 🗺 **Comparable Listings:** Show top nearby comps using geospatial distance.  
- 🧭 **Distance-to-Center Feature:** Add Haversine distance to improve geographic accuracy.  
- 📦 **Pipeline Automation:** Schedule monthly retraining with Airflow or Prefect.  
- ☁️ **Deployment:** Host the Streamlit app online for interactive demos.

---

## 🧑‍💻 Author
**Shivang Singh Manhas**  
Master’s in Information Technology & Analytics, Rutgers Business School  
Skills: Python, SQL, Power BI, Tableau, Streamlit, scikit-learn, LightGBM  
📧 [shivangmanhas20@gmail.com]
