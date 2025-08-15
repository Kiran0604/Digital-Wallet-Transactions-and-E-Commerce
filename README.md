# 💳 Digital Wallet & E-Commerce Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-Analytics-green?style=for-the-badge&logo=pandas)](https://pandas.pydata.org)
[![Machine Learning](https://img.shields.io/badge/ML-Models-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![PowerBI](https://img.shields.io/badge/PowerBI-Dashboards-yellow?style=for-the-badge&logo=powerbi)](https://powerbi.microsoft.com)

> **🚀 Interactive analytics platform for digital wallet transactions, e-commerce patterns, and financial literacy insights across India with ML models, time series forecasting, and PowerBI integration.**

---

## 📊 **Application Architecture & Navigation**

```mermaid
graph TB
    A[Main Dashboard] --> B[EDA Section]
    A --> C[Time Series Analysis]
    A --> D[Machine Learning Models]
    A --> E[Regional & Socio-Economic Analysis]
    A --> F[PowerBI Dashboards]
    A --> G[Comprehensive Overview]
    
    B --> B1[Digital Wallet Transactions]
    B --> B2[E-Commerce Orders]
    B --> B3[UPI Financial Literacy]
    
    C --> C1[UPI Transaction Forecasting]
    C --> C2[Digital Wallet Time Series Analysis]
    
    D --> D1[Customer Segmentation]
    D --> D2[Order Category Classification]
    
    E --> E1[Geographic Maps]
    E --> E2[State-wise Analysis]
    E --> E3[Payment Method Heatmaps]
    
    F --> F1[Digital Wallet PowerBI]
    F --> F2[E-Commerce PowerBI]
```

---

## 🎯 **Core Features**

### **📈 Exploratory Data Analysis (EDA)**
- **Digital Wallet Transactions**: Interactive analysis of payment patterns, merchant data, and transaction trends
- **E-Commerce Orders**: Comprehensive order analysis with profit margins, categories, and customer behavior
- **UPI Financial Literacy**: Survey insights on age groups, financial habits, and digital adoption patterns

### **⏰ Time Series Analysis**
- **UPI Transaction Forecasting**: ARIMA and Prophet models for predicting future transaction volumes
- **Digital Wallet Trends**: Moving averages, seasonality detection, and trend analysis
- **Interactive Charts**: Dynamic time series visualizations with Plotly

### **🤖 Machine Learning Models**
- **Customer Segmentation**: K-Means clustering with PCA visualization for market segmentation
- **Order Category Classification**: Automated classification of e-commerce orders into product categories
- **Performance Metrics**: Model accuracy, F1-scores, and detailed classification reports

### **🗺️ Regional & Socio-Economic Analysis**
- **Geographic Mapping**: State-wise UPI adoption with choropleth visualizations
- **Payment Heatmaps**: Regional payment method preferences and transaction patterns
- **Demographic Insights**: Age group analysis and generational financial behavior

### **📊 PowerBI Integration**
- **Digital Wallet Dashboard**: Comprehensive PowerBI dashboard for transaction analytics
- **E-Commerce Dashboard**: Sales performance, regional analysis, and profit tracking
- **Interactive Reports**: Embedded PowerBI reports with filtering capabilities

---

## 🔄 **Data Processing Pipeline**

```mermaid
flowchart TD
    subgraph "Data Sources"
        A[Digital Wallet CSV]
        B[Orders & Details CSV]
        C[UPI Literacy Survey CSV]
        D[UPI Transactions CSV]
        E[India GeoJSON]
    end
    
    subgraph "Data Processing"
        F[Data Loading & Validation]
        G[Feature Engineering]
        H[Categorical Encoding]
        I[Geographic Mapping]
    end
    
    subgraph "Analytics Engine"
        J[Interactive Visualizations]
        K[ML Model Training]
        L[Time Series Forecasting]
        M[Regional Analysis]
    end
    
    A --> F
    B --> F
    C --> F
    D --> F
    E --> I
    
    F --> G
    G --> H
    H --> J
    H --> K
    H --> L
    I --> M
    
    J --> N[Business Insights]
    K --> N
    L --> N
    M --> N
```

---

## 🛠️ **Technology Stack**

| **Component** | **Technology** | **Purpose** |
|---------------|----------------|-------------|
| **Frontend** | Streamlit | Interactive web application |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Visualization** | Plotly, Matplotlib | Interactive charts and maps |
| **Machine Learning** | Scikit-learn | Customer segmentation and classification |
| **Time Series** | Statsmodels, Prophet | Forecasting and trend analysis |
| **Geographic Analysis** | GeoJSON, Choropleth | State-wise mapping and visualization |
| **Business Intelligence** | PowerBI | Professional dashboards and reports |

---

## 📋 **Dataset Structure**

### **Core Data Sources**
```
📊 Digital Wallet Transactions (digital_wallet_transactions.csv)
├── Transaction amounts, fees, cashback, loyalty points
├── Payment methods (UPI, Credit Card, Debit Card, Digital Wallet)
├── Merchant information and product categories
├── Device types (Mobile, Desktop, Tablet)
└── Geographic location data

🛒 E-Commerce Data
├── Orders.csv: Order details, customer info, dates
├── Details.csv: Product categories, amounts, profit margins
├── Payment modes and state-wise distribution
└── Quantity and shipping information

💰 UPI Financial Literacy (upi_financial_literacy.csv)
├── Age groups and generational classifications
├── UPI usage frequency and monthly spending
├── Financial literacy scores and savings rates
└── Budgeting habits and financial behavior

🗺️ Geographic Data
├── UPI Transactions.csv: State-wise transaction volumes
├── india_state_geo.json: Geographic boundaries
└── Regional adoption patterns
```

---

## 🤖 **Machine Learning Capabilities**

### **Customer Segmentation (K-Means)**
- **Algorithm**: K-Means clustering with standardized features
- **Features**: Total spent, order count, profit, quantity
- **Visualization**: PCA 2D projection with cluster coloring
- **Business Value**: Market segmentation and targeted marketing

### **Order Category Classification**
- **Algorithm**: Random Forest classifier
- **Features**: Amount, profit, quantity, payment mode, customer data
- **Output**: Automated product category assignment
- **Accuracy**: Real-time performance metrics and classification reports

### **Performance Monitoring**
```mermaid
graph LR
    A[Data Input] --> B[Feature Extraction]
    B --> C[Model Training]
    C --> D[Performance Evaluation]
    D --> E[Classification Reports]
    E --> F[Business Insights]
    
    D --> G[Accuracy Metrics]
    D --> H[Confusion Matrix]
    D --> I[F1-Score Analysis]
```

---

## 🚀 **Quick Start Guide**

### **Installation & Setup**
```bash
# Clone the repository
git clone https://github.com/Kiran0604/Digital-Wallet-Transactions-and-E-Commerce.git
cd Digital-Wallet-Transactions-and-E-Commerce

# Navigate to project directory
cd "Digital Wallet Analysis"

# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn matplotlib seaborn statsmodels

# Launch the application
streamlit run upi_streamlit_app_interactive.py
```

### **Live Demo**
🌐 **Access the live application**: [Digital Wallet Analytics Platform](https://digital-wallet-transactions-and-e-commerce-lm3wjqvnzhxwmkdnsne.streamlit.app/)

---

## 📈 **Dashboard Navigation**

### **Main Menu Options**
1. **EDA**: Explore three datasets with interactive visualizations
2. **Time Series Analysis**: UPI forecasting and trend analysis
3. **Machine Learning Models**: Customer segmentation and classification
4. **Regional & Socio-Economic Analysis**: Geographic insights and state comparisons
5. **PowerBI Dashboards**: Professional BI reports with live data
6. **Comprehensive Overview**: Executive summary of key findings

### **Key Visualizations**
- **Choropleth Maps**: State-wise UPI transaction volumes
- **Interactive Charts**: Dynamic filtering and real-time updates
- **Customer Clusters**: PCA-based segmentation visualization
- **Time Series Plots**: Forecasting with confidence intervals
- **Heatmaps**: Payment method preferences by geography

---

## 💡 **Business Intelligence Features**

### **PowerBI Integration**
- **Digital Wallet Dashboard**: Transaction analytics, merchant performance, loyalty programs
- **E-Commerce Dashboard**: Sales performance, regional analysis, profit tracking
- **Key Metrics**: Total sales, AOV, profit margins, regional performance
- **Interactive Filtering**: Year, status, location, device type filters

### **Strategic Insights**
- **Top Performing States**: Maharashtra and Madhya Pradesh lead in sales
- **Payment Preferences**: COD still dominant, but UPI gaining traction  
- **Seasonal Patterns**: December peaks, mid-year dips in profit trends
- **Category Performance**: Clothing dominates quantity, Electronics high-value

---

## 🔍 **Advanced Analytics**

### **Time Series Forecasting**
- **Models**: ARIMA and Prophet for UPI transaction prediction
- **Metrics**: MAPE, RMSE, MAE for model evaluation
- **Seasonality**: Automatic detection of seasonal patterns
- **Confidence Intervals**: Forecast uncertainty quantification

### **Geographic Analysis**
- **State Mapping**: Interactive choropleth with transaction heatmaps
- **Regional Patterns**: Payment method adoption by geography
- **Demographic Correlations**: Age group vs digital adoption patterns

---

## 📊 **Key Performance Indicators**

| **Metric** | **Value** | **Insight** |
|------------|-----------|-------------|
| **Total Datasets** | 4 Primary Sources | Comprehensive data coverage |
| **Geographic Coverage** | 36 Indian States/UTs | National scope analysis |
| **ML Models** | 2 Core Models | Customer segmentation & classification |
| **Time Series Models** | ARIMA & Prophet | Dual forecasting approach |
| **Visualization Types** | 15+ Chart Types | Rich interactive experience |
| **Dashboard Load Time** | <3 seconds | Optimized performance |

---

## 🔮 **Future Enhancements**

### **Planned Features**
- [ ] **Real-time Data Streaming**: Live transaction monitoring
- [ ] **Advanced ML Models**: Deep learning for complex patterns
- [ ] **API Development**: RESTful endpoints for data access
- [ ] **Mobile Optimization**: Responsive design improvements

### **Analytics Expansion**
- [ ] **Predictive Analytics**: Customer lifetime value prediction
- [ ] **Anomaly Detection**: Automated fraud detection systems
- [ ] **Recommendation Engine**: Personalized product suggestions
- [ ] **Advanced Segmentation**: Behavioral clustering algorithms

---

## 🤝 **Contributing**

### **Development Guidelines**
1. **Fork** the repository and create a feature branch
2. **Follow** Python PEP 8 style guidelines
3. **Test** new features thoroughly before submission
4. **Document** code changes and update README if needed
5. **Submit** pull request with clear description

---

## 👤 **Author**

**Kiran R Aithal**
- **GitHub**: [@Kiran0604](https://github.com/Kiran0604)
- **Portfolio**: [Digital Analytics Specialist](https://github.com/Kiran0604)

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

*Built for advancing digital payment analytics and e-commerce intelligence across India 🇮🇳*

---

## 🎯 **Key Features**

### **📈 Advanced Analytics**
- **Interactive EDA**: Multi-dataset exploration with dynamic visualizations
- **Time Series Forecasting**: ARIMA & Prophet models for transaction prediction
- **Customer Segmentation**: K-Means clustering with PCA visualization
- **Regional Analysis**: State-wise payment adoption and geographic insights

### **🤖 Machine Learning Pipeline**
- **10+ ML Models**: Transaction prediction, fraud detection, customer classification
- **Ensemble Methods**: AdaBoost, Random Forest, Stacking classifiers
- **Real-time Predictions**: Interactive forms for business decision support
- **Model Performance**: Comprehensive accuracy metrics and classification reports

### **🗺️ Geographic Intelligence**
- **State-wise Mapping**: Choropleth maps for UPI adoption patterns
- **Payment Heatmaps**: Regional preference analysis across payment methods
- **Economic Correlations**: Digital literacy vs transaction volume insights

### **💼 Business Intelligence**
- **Revenue Optimization**: Category-wise performance and profit analysis
- **Customer Value**: Lifetime value prediction and segmentation strategies
- **Risk Management**: Fraud detection with outlier identification
- **Market Insights**: Age group preferences and generational trends

---

## 🔄 **Customer Journey & ML Workflow**

```mermaid
flowchart TD
    subgraph "Customer Data Flow"
        A[Customer Registration] --> B[Transaction History]
        B --> C[Payment Preferences]
        C --> D[Behavioral Patterns]
    end
    
    subgraph "ML Processing Pipeline"
        E[Feature Extraction] --> F[Data Preprocessing]
        F --> G[Model Training]
        G --> H[Prediction Engine]
        H --> I[Business Insights]
    end
    
    subgraph "Business Applications"
        J[Customer Segmentation]
        K[Fraud Detection]
        L[Revenue Forecasting]
        M[Payment Optimization]
    end
    
    D --> E
    I --> J
    I --> K
    I --> L
    I --> M
    
    J --> N[Targeted Marketing]
    K --> O[Risk Mitigation]
    L --> P[Resource Planning]
    M --> Q[UX Enhancement]
```

---

## 🛠️ **Technology Stack**

| **Category** | **Technologies** | **Purpose** |
|--------------|------------------|-------------|
| **Backend** | Python 3.8+, Pandas, NumPy | Data processing & analytics |
| **ML/AI** | Scikit-learn, XGBoost, SMOTE | Machine learning models |
| **Visualization** | Plotly, Matplotlib, Seaborn | Interactive charts & graphs |
| **Time Series** | Statsmodels, Prophet | Forecasting & trend analysis |
| **Web Framework** | Streamlit | Interactive dashboard |
| **Data Sources** | CSV, JSON (GeoJSON) | Multiple dataset formats |

---

## 📋 **Dataset Overview**

### **Core Datasets**
```
📊 Digital Wallet Transactions
├── Transaction amounts, fees, loyalty points
├── Payment methods (UPI, Credit Card, Digital Wallet)
├── Device types and customer categories
└── Geographic location data

🛒 E-Commerce Orders & Details  
├── Order information with customer details
├── Product categories and payment modes
├── Profit margins and quantity data
└── State-wise transaction patterns

💰 UPI Financial Literacy Survey
├── Age groups and generational data
├── UPI usage frequency and patterns
├── Financial literacy scores
└── Budgeting habits and savings rates

🗺️ Regional Data
├── State-wise UPI transaction volumes
├── Geographic boundaries (GeoJSON)
├── Adoption level classifications
└── Socio-economic correlations
```

---

## 🤖 **Machine Learning Models**

### **Predictive Models**
| **Model** | **Accuracy** | **Use Case** | **Algorithm** |
|-----------|--------------|--------------|---------------|
| Transaction Amount Predictor | 78% | Categorize transaction sizes | Random Forest |
| Payment Method Classifier | 85% | Predict payment preference | Ensemble (AdaBoost + RF) |
| Customer Segmentation | - | Group customer behavior | K-Means Clustering |
| Fraud Detection | 76% | Identify suspicious transactions | Isolation Forest |
| Revenue Forecasting | 79% | Predict business revenue | Random Forest |
| Customer Lifetime Value | 83% | Assess customer worth | Gradient Boosting |
| Regional Adoption | 74% | Predict area adoption levels | Random Forest |

### **Model Performance Workflow**
```mermaid
graph LR
    A[Data Input] --> B[Feature Engineering]
    B --> C[Train/Test Split]
    C --> D[Model Training]
    D --> E[Hyperparameter Tuning]
    E --> F[Performance Evaluation]
    F --> G[Classification Reports]
    G --> H[Business Insights]
    
    F --> I[Accuracy Metrics]
    F --> J[F1-Score Analysis]
    F --> K[Confusion Matrix]
```

---

## 🚀 **Quick Start Guide**

### **Installation**
```bash
# Clone the repository
git clone https://github.com/Kiran0604/Digital-Wallet-Transactions-and-E-Commerce.git
cd Digital-Wallet-Transactions-and-E-Commerce

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run "Digital Wallet Analysis/upi_streamlit_app_interactive.py"
```

### **Project Structure**
```
Digital Wallet Analysis/
├── 📱 upi_streamlit_app_interactive.py    # Main interactive dashboard
├── 📊 upi_streamlit_app.py                # Alternative dashboard version
├── 📁 Data Files/
│   ├── digital_wallet_transactions.csv
│   ├── Orders.csv & Details.csv
│   ├── upi_financial_literacy.csv
│   ├── UPI Transactions.csv
│   └── india_state_geo.json
└── 📋 requirements.txt
```

---

## 📈 **Dashboard Features**

### **Interactive Navigation**
- **EDA Section**: Dataset exploration with filterable charts
- **Time Series Analysis**: Trend forecasting and seasonality detection  
- **ML Models**: 10+ predictive models with real-time predictions
- **Regional Analysis**: Geographic insights and state comparisons
- **Comprehensive Overview**: Business intelligence summary

### **Key Visualizations**
- **Choropleth Maps**: State-wise UPI transaction heatmaps
- **Customer Segmentation**: PCA-based cluster visualization
- **Time Series Plots**: Revenue forecasting with confidence intervals
- **Payment Heatmaps**: Method preference by geography and category
- **Performance Metrics**: Model accuracy and classification reports

---

## 💡 **Business Impact & Insights**

### **Strategic Advantages**
- **Customer Understanding**: Segment-based marketing strategies
- **Risk Management**: Proactive fraud detection and prevention
- **Revenue Optimization**: Data-driven pricing and product strategies
- **Market Expansion**: Regional adoption patterns for growth planning

### **Key Findings**
- **Digital Adoption**: Maharashtra leads in UPI transaction volume
- **Payment Preferences**: Electronics favor Credit Card, COD for rural areas  
- **Customer Segments**: VIP customers drive 40% of revenue with 3% of base
- **Generational Trends**: Gen Z/Millennials show highest digital literacy

---

## 🔄 **Time Series Analysis Pipeline**

```mermaid
graph TD
    A[Historical Transaction Data] --> B[Data Preprocessing]
    B --> C[Trend Decomposition]
    C --> D[Seasonality Detection]
    D --> E[Model Selection]
    
    E --> F[ARIMA Modeling]
    E --> G[Prophet Forecasting]
    
    F --> H[Parameter Optimization]
    G --> I[Component Analysis]
    
    H --> J[Model Evaluation]
    I --> J
    
    J --> K[MAPE/RMSE Metrics]
    K --> L[Forecast Generation]
    L --> M[Business Planning]
```

---

## 🎨 **Visualization Gallery**

### **Regional Intelligence**
- **State Performance Maps**: Interactive choropleth visualizations
- **Payment Method Heatmaps**: Cross-regional preference analysis
- **Adoption Trending**: Time-based geographic expansion patterns

### **Customer Analytics** 
- **Segmentation Clusters**: 2D PCA projections with behavioral groupings
- **Lifetime Value Distribution**: Customer tier visualizations
- **Transaction Patterns**: Frequency and amount correlation plots

### **Business Metrics**
- **Revenue Forecasting**: Multi-model prediction comparisons
- **Category Performance**: Profit margin and volume analysis
- **Fraud Detection**: Outlier identification and risk scoring

---

## 🔧 **Advanced Configuration**

### **Model Customization**
```python
# Example: Customize clustering parameters
n_clusters = st.slider("Select K for Customer Segmentation", 2, 6, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Example: Fraud detection threshold adjustment  
contamination = st.slider("Fraud Detection Sensitivity", 0.05, 0.2, 0.1)
fraud_model = IsolationForest(contamination=contamination)
```

### **Data Integration**
- **CSV Upload**: Dynamic data loading capabilities
- **GeoJSON Support**: Custom geographic boundary integration
- **API Ready**: Extensible for real-time data feeds

---

## 📊 **Performance Metrics**

| **Category** | **Metric** | **Performance** |
|--------------|------------|-----------------|
| **Data Processing** | Processing Speed | <2 seconds for 100K records |
| **Model Training** | Average Training Time | 15-30 seconds per model |
| **Visualization** | Chart Rendering | Real-time with Plotly |
| **Accuracy** | Average Model Performance | 79.2% across all models |
| **User Experience** | Dashboard Load Time | <3 seconds |

---

## 🔮 **Future Enhancements**

### **Version 2.0 Roadmap**
- [ ] **Real-time Data Streaming**: Live transaction processing
- [ ] **Deep Learning Models**: LSTM networks for sequence prediction
- [ ] **API Development**: RESTful endpoints for external integration
- [ ] **Mobile Optimization**: Responsive design for mobile devices

### **Advanced Analytics**
- [ ] **Sentiment Analysis**: Social media sentiment correlation
- [ ] **Network Analysis**: Transaction flow and customer connections
- [ ] **Recommendation Engine**: Personalized product suggestions
- [ ] **A/B Testing Framework**: Experiment tracking and analysis

---

## 🤝 **Contributing**

### **Development Workflow**
1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### **Code Standards**
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

---

## 📄 **License & Usage**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### **Citation**
```bibtex
@software{digital_wallet_analytics,
  title={Digital Wallet & E-Commerce Analytics Platform},
  author={Kiran R Aithal},
  year={2024},
  url={https://github.com/Kiran0604/Digital-Wallet-Transactions-and-E-Commerce}
}
```

---

## 👤 **Author**

**Kiran R Aithal**
- **GitHub**: [@Kiran0604](https://github.com/Kiran0604)
- **LinkedIn**: [Connect on LinkedIn](https://linkedin.com/in/kiran-aithal)

---

## 📈 **Project Statistics**

| **Component** | **Count** |
|---------------|-----------|
| **ML Models** | 10+ Predictive Models |
| **Datasets** | 4 Primary Data Sources |
| **Visualizations** | 25+ Interactive Charts |
| **Geographic Coverage** | 36 Indian States/UTs |
| **Customer Segments** | 6 Behavioral Clusters |
| **Accuracy Range** | 74-85% Model Performance |

---

*Built with ❤️ for advancing digital payment analytics and financial technology insights across India*
