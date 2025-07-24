# Digital Wallet & E-Commerce Dashboard

A comprehensive Streamlit dashboard for interactive EDA, time series analysis, customer segmentation, and machine learning on digital wallet, e-commerce, and financial literacy datasets from India.

## Features
- **Interactive EDA**: Visualize and explore digital wallet transactions, e-commerce orders, and UPI financial literacy survey data.
- **Time Series Analysis**: Analyze trends, seasonality, and growth in digital wallet transactions with interactive Plotly charts.
- **Customer Segmentation**: K-Means clustering with dynamic insights and PCA visualization for actionable customer segments.
- **Churn Prediction**: Machine learning models (RandomForest, XGBoost, LightGBM) with class imbalance handled using SMOTE.
- **Regional & Socio-Economic Analysis**: State-wise and category-wise payment and revenue insights.
- **Dark Theme Optimized**: All visuals styled for dark backgrounds.
- **Robust Error Handling**: Graceful handling of missing data and dependencies.

## Project Structure
```
Digital Wallet Analysis/
├── app.py
├── upi_streamlit_app_interactive.py  # Main dashboard code
├── requirements.txt
├── Details.csv
├── Orders.csv
├── digital_wallet_transactions.csv
├── upi_financial_literacy.csv
├── UPI Transactions.csv
├── india_state_geo.json
├── ... (other data files)
```

## Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/digital-wallet-dashboard.git
   cd digital-wallet-dashboard/Digital Wallet Analysis
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add data files:**
   Place all required CSV and JSON data files in the `Digital Wallet Analysis` folder.

## Usage
Run the Streamlit app:
```sh
streamlit run upi_streamlit_app_interactive.py
```

## Data Sources
- `digital_wallet_transactions.csv`: Digital wallet transaction data
- `Orders.csv`, `Details.csv`: E-commerce order and product details
- `upi_financial_literacy.csv`: UPI financial literacy survey
- `UPI Transactions.csv`: UPI transaction records
- `india_state_geo.json`: GeoJSON for Indian states

## Key Libraries
- Streamlit
- Pandas, NumPy
- Plotly
- scikit-learn, imblearn, xgboost
- statsmodels, scipy

## Author
Kiran
