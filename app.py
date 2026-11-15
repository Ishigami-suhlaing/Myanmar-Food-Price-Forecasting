from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Myanmar Rice Price Forecasting",
    page_icon="🌾",
    layout="wide"
)

# Robust path resolution relative to app.py location
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "prices.csv"

TARGET_COMMODITY = "Rice (low quality)"
TARGET_UNIT = "KG"
TARGET_PRICE_TYPE = "Retail"
TARGET_CURRENCY = "MMK"
PAKOKKU_ID = 1664

WINDOW_SIZE = 36
RIDGE_ALPHA = 10.0


# ============================================================
# 2. LOAD AND PREPARE HISTORICAL DATA
# ============================================================

@st.cache_data
def load_monthly_prices(data_path):

    path_obj = Path(data_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {path_obj.resolve()}"
        )

    df = pd.read_csv(path_obj)

    # Standardize column names
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
    )

    required_columns = {
        "date", "commodity", "market_id", "market",
        "unit", "pricetype", "currency", "price"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Clean data types
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"], errors="coerce"
    )

    df["market_id"] = pd.to_numeric(
        df["market_id"], errors="coerce"
    )

    # Strip whitespace from text columns
    for col in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[col] = df[col].str.strip()

    # Filter the selected commodity
    rice_df = df[
        (df["commodity"] == TARGET_COMMODITY)
        & (df["unit"] == TARGET_UNIT)
        & (df["pricetype"] == TARGET_PRICE_TYPE)
        & (df["currency"] == TARGET_CURRENCY)
    ].dropna(subset=["date", "price"]).copy()

    rice_df["month"] = (
        rice_df["date"].dt.to_period("M")
    )

    # Same monthly aggregation as the notebook
    monthly_rice = (
        rice_df.groupby(
            ["market_id", "market", "month"]
        )
        .agg(
            monthly_price=("price", "mean"),
            observation_count=("price", "count")
        )
        .reset_index()
        .sort_values(["market_id", "month"])
    )

    # Select Pakokku
    pakokku = (
        monthly_rice[
            monthly_rice["market_id"] == PAKOKKU_ID
        ]
        .sort_values("month")
        .set_index("month")
        .copy()
    )

    if pakokku.empty:
        raise ValueError(
            "No rice price observations found for Pakokku."
        )

    # Restore complete monthly calendar.
    # Missing prices remain NaN.
    calendar = pd.period_range(
        start=pakokku.index.min(),
        end=pakokku.index.max(),
        freq="M"
    )

    pakokku = pakokku.reindex(calendar)
    pakokku.index.name = "month"

    return pakokku["monthly_price"]


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def build_full_features(monthly_prices):

    out = pd.DataFrame({
        "monthly_price": monthly_prices
    })

    # 12 historical lags
    for lag in range(1, 13):
        out[f"lag_{lag}"] = (
            out["monthly_price"].shift(lag)
        )

    # Rolling means use only previous months
    history = out["monthly_price"].shift(1)

    for window in [3, 6, 12]:
        out[f"rolling_mean_{window}"] = (
            history.rolling(
                window=window,
                min_periods=window
            ).mean()
        )

    return out


def build_compact_features(monthly_prices):

    out = pd.DataFrame({
        "monthly_price": monthly_prices
    })

    for lag in [1, 2, 3]:
        out[f"lag_{lag}"] = (
            out["monthly_price"].shift(lag)
        )

    history = out["monthly_price"].shift(1)

    out["rolling_mean_3"] = (
        history.rolling(
            window=3,
            min_periods=3
        ).mean()
    )

    return out


# ============================================================
# 4. PERCENTAGE-RETURN TRANSFORMATION
# ============================================================

def make_percentage_features(data):

    transformed = data.copy()

    # Target: percentage price change
    y_pct = (
        transformed["monthly_price"]
        - transformed["lag_1"]
    ) / transformed["lag_1"]

    # Convert historical prices into relative changes
    price_cols = [
        col for col in transformed.columns
        if col.startswith(
            ("lag_", "rolling_mean_")
        )
        and col != "lag_1"
    ]

    for col in price_cols:

        transformed[f"{col}_pct"] = (
            transformed[col]
            - transformed["lag_1"]
        ) / transformed["lag_1"]

    X_pct = transformed.drop(
        columns=[
            "monthly_price",
            "lag_1"
        ] + price_cols
    )

    return X_pct, y_pct


# ============================================================
# 5. TRAIN THE SELECTED RIDGE MODEL
# ============================================================

@st.cache_resource
def train_ridge_model(monthly_prices):

    feature_frame = build_full_features(
        monthly_prices
    )

    compact_frame = build_compact_features(
        monthly_prices
    )

    full_feature_cols = [
        col for col in feature_frame.columns
        if col != "monthly_price"
    ]

    compact_feature_cols = [
        col for col in compact_frame.columns
        if col != "monthly_price"
    ]

    # Follow the notebook's complete-row requirements
    full_ready = feature_frame.dropna(
        subset=["monthly_price"] + full_feature_cols
    )

    compact_ready = compact_frame.dropna(
        subset=["monthly_price"] + compact_feature_cols
    )

    # Use the same common-month intersection
    common_months = (
        full_ready.index
        .intersection(compact_ready.index)
        .sort_values()
    )

    full_eval = full_ready.loc[
        common_months
    ].copy()

    # Percentage-based training data
    X_all, y_all = make_percentage_features(
        full_eval
    )

    # Latest 36 prior observations
    X_train = X_all.tail(WINDOW_SIZE).copy()
    y_train = y_all.reindex(X_train.index)

    valid = (
        X_train.notna().all(axis=1)
        & y_train.notna()
    )

    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid]

    if len(X_train) < 12:
        raise ValueError(
            "Not enough valid observations to train Ridge."
        )

    # Selected model from the notebook
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA)
    )

    model.fit(X_train, y_train)

    return model, X_train.columns.tolist(), len(X_train)


# ============================================================
# 6. RECURSIVE FUTURE FORECASTING
# ============================================================

def generate_forecast(
    model,
    feature_columns,
    monthly_prices,
    horizon
):

    history = monthly_prices.copy()

    predictions = []

    for step in range(horizon):

        next_month = history.index[-1] + 1

        # Add an empty target month.
        # Its predictors use only previous prices.
        extended_history = pd.concat([
            history,
            pd.Series(
                [np.nan],
                index=pd.PeriodIndex(
                    [next_month],
                    freq="M"
                ),
                name="monthly_price"
            )
        ])

        feature_frame = build_full_features(
            extended_history
        )

        X_future, _ = make_percentage_features(
            feature_frame
        )

        X_next = X_future.loc[[next_month]]

        # Check required historical information
        if X_next[feature_columns].isna().any().any():
            raise ValueError(
                f"Insufficient consecutive historical data "
                f"to forecast {next_month}."
            )

        X_next = X_next[feature_columns]

        # Predict percentage return
        predicted_return = model.predict(
            X_next
        )[0]

        previous_price = history.iloc[-1]

        # Convert percentage return into MMK/KG
        predicted_price = (
            previous_price * (1 + predicted_return)
        )

        if not np.isfinite(predicted_price) or predicted_price <= 0:
            raise ValueError(
                f"Invalid predicted price for {next_month}."
            )

        predictions.append({
            "month": next_month.to_timestamp(),
            "predicted_price": predicted_price,
            "predicted_return_pct": predicted_return * 100
        })

        # Append prediction for the next recursive step
        history.loc[next_month] = predicted_price

    return pd.DataFrame(predictions).set_index("month")


# ============================================================
# 7. STREAMLIT APPLICATION
# ============================================================

st.title("🌾 Myanmar Rice Price Forecasting")
st.subheader("Pakokku Market")

st.caption(
    "Rice (low quality) | Retail | MMK per KG"
)

try:

    monthly_prices = load_monthly_prices(
        str(DATA_PATH)
    )

    model, feature_columns, training_count = (
        train_ridge_model(monthly_prices)
    )

except Exception as error:

    st.error(str(error))
    st.stop()


# ------------------------------------------------------------
# Historical data
# ------------------------------------------------------------

st.header("Historical Rice Prices")

observed_prices = monthly_prices.dropna()

latest_price = observed_prices.iloc[-1]
latest_month = observed_prices.index[-1]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest observed price",
    f"{latest_price:,.2f} MMK/KG"
)

col2.metric(
    "Latest observation",
    str(latest_month)
)

col3.metric(
    "Training observations",
    training_count
)


st.divider()


# ------------------------------------------------------------
# Future forecasting
# ------------------------------------------------------------

st.header("Future Rice Price Forecast")

horizon = st.selectbox(
    "Select forecast horizon",
    options=[1, 2, 3, 4, 5, 6],
    format_func=lambda x: f"{x} month(s) ahead"
)

if st.button(
    "Generate Forecast",
    type="primary"
):

    try:

        forecast_df = generate_forecast(
            model=model,
            feature_columns=feature_columns,
            monthly_prices=monthly_prices,
            horizon=horizon
        )

        st.success("Forecast generated successfully.")

        st.subheader("Predicted Prices")

        st.dataframe(
            forecast_df.round(2),
            use_container_width=True
        )

        st.metric(
            "Next month's predicted price",
            f"{forecast_df['predicted_price'].iloc[0]:,.2f} MMK/KG"
        )

        st.subheader("Historical vs Forecast")

        chart_data = pd.concat([
            monthly_prices.rename("Historical"),
            forecast_df["predicted_price"].rename("Forecast")
        ])

        st.line_chart(chart_data)

        st.download_button(
            "Download Forecast CSV",
            data=forecast_df.to_csv().encode("utf-8"),
            file_name="pakokku_rice_forecast.csv",
            mime="text/csv"
        )

    except Exception as error:

        st.error(f"Forecasting failed: {error}")


# ============================================================
# 8. MODEL INFORMATION
# ============================================================

with st.expander("About the forecasting model"):

    st.write("**Model:** Ridge Regression (Percentage Return)")

    st.write("**Regularization:** alpha = 10.0")

    st.write("**Training strategy:** Sliding window (36 observations)")

    st.write("**Preprocessing:** StandardScaler")

    st.write("**Features:** 12 historical lags and 3 rolling means, transformed into relative percentage features.")

    st.write(
        "**Forecasting:** Recursive one-step predictions. "
        "Predicted prices are used as historical inputs for subsequent forecast months."
    )

    st.warning(
        "Longer-horizon forecasts are recursive estimates. "
        "Prediction errors can accumulate, and the model's "
        "one-step validation performance does not guarantee "
        "the same accuracy over multiple months."
    )