# EV Adoption Forecasting

A machine learning project that forecasts electric vehicle (EV) adoption at the county level for the next 3 years. It includes an interactive Streamlit app. Built during the **AICTE Internship (Cycle 2)**.

![EV banner](ev-car-factory.jpg)

## Why it matters

As more people switch to EVs, cities need to plan charging infrastructure ahead of demand. This project uses historical EV registration data to predict how EV counts will grow in each county. Planners can use those predictions to see where charging capacity will be needed.

## Features

- **County forecast:** pick a county and see its cumulative EV count, both historical and forecast for the next 36 months.
- **Growth estimate:** see the expected percentage growth in EV adoption over 3 years.
- **County comparison:** compare the forecast trends of up to 3 counties on one chart.

## How it works

1. **Data:** county-level vehicle registration data (`Electric_Vehicle_Population_By_County`, about 20.8k rows), including BEV, PHEV and total EV counts per month.
2. **Cleaning:** parse dates, fill missing county and state values, and cap outliers in `Percent Electric Vehicles` using IQR bounds.
3. **Feature engineering:** for each county, build:
   - time features: `months_since_start` and the label-encoded county
   - lag features: EV totals from 1, 2 and 3 months earlier (`ev_total_lag1–3`)
   - the 3-month rolling mean and the 1-month and 3-month percentage change
   - `ev_growth_slope`: the 6-month linear slope of cumulative EVs
4. **Model:** a `RandomForestRegressor` tuned with `RandomizedSearchCV` over 30 candidates with 3-fold CV. Best parameters: 200 trees, `max_depth=15`, `min_samples_split=4`.
5. **Forecast:** the model predicts one month at a time. Each prediction feeds back in as a lag feature for the next month, repeated for 36 months.

**Test results (last 10% of the data, unshuffled):** MAE 0.01 · RMSE 0.06 · R² 1.00

## Project structure

```
├── EV_Adoption_Forecasting.ipynb                # EDA, feature engineering, training, evaluation
├── Electric_Vehicle_Population_By_County.csv     # Raw dataset
├── preprocessed_ev_data.csv                      # Cleaned data with engineered features
├── forecasting_ev_model.pkl                      # Trained Random Forest model
├── app.py                                        # Streamlit web app
└── requirements.txt
```

## Run locally

```bash
git clone https://github.com/Dracoindie/EV-Charging-Demand-Prediction-.git
cd EV-Charging-Demand-Prediction-
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit · Jupyter

## Author

**Nishchay Singh**: [@Dracoindie](https://github.com/Dracoindie)
