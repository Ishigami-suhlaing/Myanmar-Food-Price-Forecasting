# Myanmar Food Price Forecasting

![app demo](<images/demo.png>)

## 🌐 Live Demo

**Try the deployed application here:**

[Myanmar Food Price Forecasting – Live Application](https://myanmar-food-price-forecasting.onrender.com/)

The application is deployed using **Render** and built with **Streamlit**.

---

## 1. Project Overview

**Myanmar Rice Price Forecasting** is a machine learning project that aims to forecast the monthly retail price of rice in Myanmar.

Food prices can fluctuate due to various factors, including supply and demand, transportation costs, economic instability, seasonal changes, and other market conditions. These fluctuations make it difficult for consumers, traders, and other stakeholders to understand potential future price movements.

This project explores how historical food-price data and machine learning can be used to analyze price trends and generate future price forecasts.

The project is part of the ML Zoomcamp midterm project.

## 2. Problems We Are Trying to Address

Food-price fluctuations create uncertainty for people who depend on essential commodities.

Some of the main problems this project explores are:

- **Price uncertainty:** Consumers and traders may have difficulty anticipating future commodity prices.
- **Historical price analysis:** Understanding how prices change over time can be challenging without systematic analysis.
- **Market fluctuations:** Sudden price increases and decreases can make forecasting particularly difficult.
- **Data limitations:** Historical market data may contain missing observations, irregular changes, and limited records, which affect forecasting performance.

The goal is to investigate whether machine learning can provide useful estimates of future rice prices based on historical observations.

## 3. What Does the Model Do?

The application uses a trained machine learning model to forecast rice prices using historical monthly price information.

The forecasting workflow includes:

1. Collecting and preparing historical rice-price data.
2. Analyzing price trends and historical fluctuations.
3. Engineering features from historical price observations.
4. Training and evaluating forecasting models.
5. Selecting a model based on time-series validation.
6. Generating price forecasts through a Streamlit web application.

### Selected Model

**Ridge Regression (Percentage-based features)**

The final model was selected using a Sliding Walk-Forward validation strategy with a 36-month window.

| Evaluation metric | Result |
|---|---:|
| MSE | 131.80 |
| RMSE | 171.24 |

These results represent the selected model's evaluation under the specified validation strategy. They should not be interpreted as a guarantee of future forecasting accuracy.

## 4. Current Project Scope and Limitations

At the current development stage, the project supports only one market and one commodity.

| Configuration | Current selection |
|---|---|
| Country | Myanmar |
| Market | Pakokku |
| Market ID | 1664 |
| Commodity | Rice (low quality) |
| Price type | Retail |
| Unit | KG |
| Currency | MMK |
| Forecasting frequency | Monthly |
| Model | Ridge Regression (Pct) |

### Current Limitations

The project is currently a focused case study rather than a nationwide food-price forecasting system.

- Only Pakokku market is supported.
- Only Rice (low quality) is included.
- The model has not yet been validated for other markets or commodities.
- Historical price data contains missing observations and irregular price fluctuations.
- Forecasting performance may be affected by unexpected economic events and sudden market changes.

The model's performance on Pakokku rice prices should not be assumed to generalize to other markets or commodities.

### Future Improvements

Potential future development includes:

- Expanding forecasting coverage to additional markets across Myanmar.
- Supporting more essential food commodities.
- Incorporating external variables such as inflation, transportation costs, and other relevant economic indicators.
- Experimenting with additional forecasting algorithms.
- Improving forecasting performance during periods of sudden price fluctuations.
- Developing a more comprehensive food-price monitoring and forecasting platform.

## 5. Running the Project Locally

### Prerequisites

Make sure you have installed:

- Python
- Pipenv
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ishigami-suhlaing/Myanmar-Food-Price-Forecasting.git
```

Navigate into the project directory:

```bash
cd Myanmar-Food-Price-Forecasting
```

### Step 2: Install Dependencies

This project uses Pipenv for dependency management.

```bash
pipenv install --deploy
```

### Step 3: Activate the Virtual Environment

```bash
pipenv shell
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

Alternatively:

```bash
pipenv run streamlit run app.py
```

### Step 5: Open the Application

```text
http://localhost:8501
```

## 6. Technologies Used

- **Python** – Main programming language.
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical computation.
- **Scikit-learn** – Machine learning model development and evaluation.
- **Streamlit** – Interactive web application.
- **Pipenv** – Python dependency and virtual environment management.
- **Render** – Application deployment.

## 7. Project Status

**Current status: Deployed**

The first version of the application is deployed and focuses on monthly retail price forecasting for Rice (low quality) in Pakokku, Myanmar.

This project is an ongoing exploration of applying machine learning to real-world food-price forecasting problems.

---

*Developed as an ML Zoomcamp midterm project.*