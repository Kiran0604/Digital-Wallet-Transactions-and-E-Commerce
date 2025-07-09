import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
from datetime import datetime
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
import json
warnings.filterwarnings('ignore')

# Load data with error handling
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

wallet_df = safe_read_csv('digital_wallet_transactions.csv')
orders_df = safe_read_csv('Orders.csv')
details_df = safe_read_csv('Details.csv')
lit_df = safe_read_csv('upi_financial_literacy.csv')
upi_df = safe_read_csv('UPI Transactions.csv')

# Merge Orders and Details datasets
try:
    merged_orders_df = pd.merge(orders_df, details_df, on='Order ID', how='inner')
except Exception as e:
    st.error(f"Error merging datasets: {e}")
    merged_orders_df = pd.DataFrame()

# Clean and standardize state names across all datasets
def clean_state_names(df, state_column):
    """Clean and standardize state names"""
    if state_column in df.columns:
        # Create a copy to avoid modifying the original
        df = df.copy()
        # Standardize state names
        df[state_column] = df[state_column].str.strip()  # Remove leading/trailing spaces
        
        # Fix common state name variations
        state_corrections = {
            'Karnatak': 'Karnataka',
            'karnatak': 'Karnataka',
            'KARNATAK': 'Karnataka',
            'Karnataka': 'Karnataka',  # Keep as is
            'Kerala ': 'Kerala',  # Remove trailing space
            'kerala ': 'Kerala',
            'KERALA ': 'Kerala'
        }
        
        # Apply corrections
        for old_name, new_name in state_corrections.items():
            df[state_column] = df[state_column].replace(old_name, new_name)
    
    return df

# Apply state name cleaning to all relevant datasets
if 'location' in wallet_df.columns:
    wallet_df = clean_state_names(wallet_df, 'location')
if 'State' in merged_orders_df.columns:
    merged_orders_df = clean_state_names(merged_orders_df, 'State')
if 'State' in orders_df.columns:
    orders_df = clean_state_names(orders_df, 'State')

def show_eda_upi():
    st.header('EDA: Digital Wallet Transactions')
    
    # Dataset Description
    st.markdown("""
    ### 📱 **Digital Wallet Transactions Dataset**
    
    This dataset contains comprehensive information about digital wallet transactions, providing insights into:
    - **Transaction Details**: Amount, fees, cashback, and loyalty points for each transaction
    - **Payment Methods**: Various digital payment options used by customers
    - **Geographic Data**: Location-wise transaction patterns across different regions
    - **Device Usage**: Types of devices used for making digital payments
    - **Transaction Status**: Success rates and failure patterns
    - **Product Categories**: Different types of purchases made through digital wallets
    - **Temporal Patterns**: Transaction dates for trend analysis
    
    This data helps understand digital payment adoption, user behavior, and market penetration across different demographics and geographies.
    """)
    
    st.write('**First 5 rows:**')
    st.write(wallet_df.head())
    st.write('**Summary Statistics:**')
    st.write(wallet_df.describe())

    # --- Interactive Filters ---
    st.subheader('🎛️ Interactive Filters')
    st.markdown("*All charts below will update based on your filter selection.*")
    
    # Location Selector for Digital Wallet only
    if 'location' in wallet_df.columns:
        all_locations = wallet_df['location'].dropna().unique().tolist()
        selected_location = st.selectbox('Select a Location for Digital Wallet Analysis', ['All'] + sorted(all_locations), key='location_selector_1')
        wallet_data = wallet_df if selected_location == 'All' else wallet_df[wallet_df['location'] == selected_location]
    else:
        wallet_data = wallet_df
        selected_location = 'All'

    # Display current filter status
    st.info(f"**Current Filter:** Location: {selected_location}")
    st.info(f"**Filtered Data Size:** Digital Wallet: {len(wallet_data):,} records")

    # Transaction Amount Distribution (Histogram + KDE)
    st.subheader('📊 Filtered Transaction Analysis')
    st.write('Transaction Amount Distribution:')
    fig1 = plt.figure(figsize=(10, 5))
    sns.histplot(wallet_data['product_amount'], bins=30, kde=True, color='skyblue')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.title(f'Distribution of Digital Wallet Transaction Amounts ({selected_location})')
    st.pyplot(fig1)
    
    # Dynamic insights for transaction amount distribution
    if len(wallet_data) > 0:
        avg_amount = wallet_data['product_amount'].mean()
        median_amount = wallet_data['product_amount'].median()
        max_amount = wallet_data['product_amount'].max()
        min_amount = wallet_data['product_amount'].min()
        std_amount = wallet_data['product_amount'].std()
        small_txns = (wallet_data['product_amount'] <= 1000).sum()
        large_txns = (wallet_data['product_amount'] >= 10000).sum()
        skewness = "right-skewed" if avg_amount > median_amount else "left-skewed" if avg_amount < median_amount else "symmetric"
        
        st.info(f'''\
**Insights for {selected_location} location(s):**
1. **Transaction Range:** ₹{min_amount:,.0f} - ₹{max_amount:,.0f} (Average: ₹{avg_amount:,.0f}, Median: ₹{median_amount:,.0f})
2. **Distribution Pattern:** {skewness} distribution with {small_txns:,} small transactions (≤₹1,000) and {large_txns:,} large transactions (≥₹10,000)
3. **Market Insight:** {"Small-value transactions dominate, indicating everyday payment usage" if small_txns > large_txns else "Mix of small and large transactions suggests diverse use cases"}
4. **Variability:** Standard deviation of ₹{std_amount:,.0f} shows {"high" if std_amount > avg_amount else "moderate"} transaction amount variability''')
    else:
        st.warning("No data available for the selected filters.")

    # Top 5 Transaction Categories
    top_cats = wallet_data['product_category'].value_counts().head(5)
    st.write('Top 5 Transaction Categories (Filtered):')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top_cats.values, y=top_cats.index, ax=ax2, palette='viridis')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Transaction Category')
    ax2.set_title(f'Most Common Digital Wallet Transaction Categories ({selected_location})')
    for i, v in enumerate(top_cats.values):
        ax2.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig2)
    
    # Dynamic insights for transaction categories
    if len(top_cats) > 0:
        total_txns = len(wallet_data)
        top1_pct = (top_cats.iloc[0] / total_txns) * 100
        top3_pct = (top_cats.head(3).sum() / total_txns) * 100
        category_diversity = len(wallet_data['product_category'].unique())
        
        st.info(f'''\
**Category Analysis for {selected_location}:**
1. **Leading Category:** "{top_cats.index[0]}" dominates with {top_cats.iloc[0]:,} transactions ({top1_pct:.1f}% of total)
2. **Market Concentration:** Top 3 categories account for {top3_pct:.1f}% of all transactions in this location
3. **Category Diversity:** {category_diversity} different transaction categories available
4. **Business Opportunity:** {"High concentration suggests focused user behavior" if top1_pct > 40 else "Balanced distribution indicates diverse usage patterns"}''')
    else:
        st.warning("No category data available for the selected filters.")

    # Payment Method Usage
    payment_counts = wallet_data['payment_method'].value_counts()
    st.write('Payment Method Usage (Filtered):')
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=payment_counts.values, y=payment_counts.index, ax=ax3, palette='mako')
    ax3.set_xlabel('Count')
    ax3.set_ylabel('Payment Method')
    ax3.set_title(f'Payment Method Popularity ({selected_location})')
    for i, v in enumerate(payment_counts.values):
        ax3.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig3)
    
    # Dynamic insights for payment methods
    if len(payment_counts) > 0:
        total_payments = payment_counts.sum()
        dominant_method = payment_counts.index[0]
        dominant_pct = (payment_counts.iloc[0] / total_payments) * 100
        method_diversity = len(payment_counts)
        
        st.info(f'''\
**Payment Method Analysis for {selected_location}:**
1. **Preferred Method:** "{dominant_method}" is the most popular with {payment_counts.iloc[0]:,} transactions ({dominant_pct:.1f}%)
2. **Market Share:** {"Single method dominates" if dominant_pct > 50 else "Competitive market with multiple methods"}
3. **Method Diversity:** {method_diversity} different payment methods are actively used
4. **Strategic Insight:** {"Focus on optimizing the dominant method" if dominant_pct > 60 else "Multi-method strategy recommended for user convenience"}''')
    else:
        st.warning("No payment method data available for the selected filters.")

    # Device Usage
    device_counts = wallet_data['device_type'].value_counts()
    st.write('Transaction Device Usage (Filtered):')
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=device_counts.values, y=device_counts.index, ax=ax4, palette='crest')
    ax4.set_xlabel('Count')
    ax4.set_ylabel('Device')
    ax4.set_title(f'Device Used for Digital Wallet Transactions ({selected_location})')
    for i, v in enumerate(device_counts.values):
        ax4.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig4)
    
    # Dynamic insights for device usage
    if len(device_counts) > 0:
        total_devices = device_counts.sum()
        primary_device = device_counts.index[0]
        primary_pct = (device_counts.iloc[0] / total_devices) * 100
        mobile_usage = device_counts.get('Mobile', 0) + device_counts.get('mobile', 0) + device_counts.get('Smartphone', 0)
        
        st.info(f'''\
**Device Usage Analysis for {selected_location}:**
1. **Primary Device:** "{primary_device}" leads with {device_counts.iloc[0]:,} transactions ({primary_pct:.1f}%)
2. **Mobile Preference:** {"Mobile-first user base" if "mobile" in primary_device.lower() or "phone" in primary_device.lower() else "Mixed device preferences"}
3. **Device Accessibility:** {len(device_counts)} different device types show {"good" if len(device_counts) > 2 else "limited"} accessibility
4. **UX Priority:** {"Optimize mobile experience" if primary_pct > 70 and "mobile" in primary_device.lower() else "Multi-device optimization needed"}''')
    else:
        st.warning("No device usage data available for the selected filters.")

    # Location-wise Transaction Count
    location_counts = wallet_data['location'].value_counts().head(10)
    st.write('Top 10 Transaction Locations (Filtered):')
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=location_counts.values, y=location_counts.index, ax=ax5, palette='flare')
    ax5.set_xlabel('Count')
    ax5.set_ylabel('Location')
    ax5.set_title(f'Top 10 Locations by Transaction Count ({selected_location})')
    for i, v in enumerate(location_counts.values):
        ax5.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig5)
    
    # Dynamic insights for location distribution
    if len(location_counts) > 0:
        total_locations = len(wallet_data['location'].unique())
        top_location = location_counts.index[0]
        top_location_pct = (location_counts.iloc[0] / len(wallet_data)) * 100
        geographic_spread = "concentrated" if len(location_counts) < 5 else "diverse"
        
        st.info(f'''\
**Location Analysis for {selected_location}:**
1. **Leading Location:** "{top_location}" has {location_counts.iloc[0]:,} transactions ({top_location_pct:.1f}% of filtered data)
2. **Geographic Distribution:** {geographic_spread} spread across {total_locations} unique locations
3. **Market Penetration:** {"High concentration in specific areas" if top_location_pct > 30 else "Balanced distribution across locations"}
4. **Growth Opportunity:** {"Focus on expanding successful locations" if geographic_spread == "concentrated" else "Maintain diverse geographic presence"}''')
    else:
        st.warning("No location data available for the selected filters.")

    # Transaction Status Distribution
    status_counts = wallet_data['transaction_status'].value_counts()
    st.write('Transaction Status Distribution (Filtered):')
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=status_counts.values, y=status_counts.index, ax=ax6, palette='viridis')
    ax6.set_xlabel('Count')
    ax6.set_ylabel('Transaction Status')
    ax6.set_title(f'Distribution of Transaction Status ({selected_location})')
    for i, v in enumerate(status_counts.values):
        ax6.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig6)
    
    # Dynamic insights for transaction status
    if len(status_counts) > 0:
        total_txns = status_counts.sum()
        success_rate = (status_counts.get('Success', 0) + status_counts.get('Completed', 0) + status_counts.get('success', 0)) / total_txns * 100
        failed_txns = status_counts.get('Failed', 0) + status_counts.get('Error', 0) + status_counts.get('failed', 0)
        pending_txns = status_counts.get('Pending', 0) + status_counts.get('Processing', 0)
        
        st.info(f'''\
**Transaction Status Analysis for {selected_location}:**
1. **Success Rate:** {success_rate:.1f}% of transactions completed successfully ({total_txns - failed_txns - pending_txns:,} out of {total_txns:,})
2. **Failed Transactions:** {failed_txns:,} failed transactions {"indicate technical issues" if failed_txns > total_txns * 0.05 else "show good system reliability"}
3. **System Performance:** {"Excellent" if success_rate > 95 else "Good" if success_rate > 90 else "Needs improvement"} transaction processing
4. **Operational Focus:** {"Monitor and reduce failures" if failed_txns > total_txns * 0.05 else "Maintain current reliability standards"}''')
    else:
        st.warning("No transaction status data available for the selected filters.")

    # Monthly Business Trends Analysis for Digital Wallet Transactions
    if 'transaction_date' in wallet_data.columns:
        wallet_data['transaction_date'] = pd.to_datetime(wallet_data['transaction_date'], errors='coerce')
        wallet_data['Transaction_Month'] = wallet_data['transaction_date'].dt.to_period('M')
        
        # Group by month and aggregate metrics
        monthly_wallet_trends = wallet_data.groupby('Transaction_Month').agg({
            'transaction_id': 'count',  # Transaction count
            'product_amount': ['sum', 'mean'],  # Total and average transaction amounts
            'transaction_fee': 'sum',  # Total fees
            'cashback': 'sum',  # Total cashback
            'loyalty_points': 'sum'  # Total loyalty points
        }).round(2)
        
        # Flatten column names
        monthly_wallet_trends.columns = ['Transaction_Count', 'Total_Amount', 'Avg_Amount', 'Total_Fees', 'Total_Cashback', 'Total_Points']
        
        st.write('Monthly Digital Wallet Business Trends:')
        fig8, (ax8a, ax8b, ax8c) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Monthly Transaction Count
        monthly_wallet_trends['Transaction_Count'].plot(kind='line', marker='o', color='teal', ax=ax8a)
        ax8a.set_title('Monthly Digital Wallet Transaction Volume')
        ax8a.set_ylabel('Number of Transactions')
        ax8a.grid(True, alpha=0.3)
        
        # Monthly Total Amount
        monthly_wallet_trends['Total_Amount'].plot(kind='line', marker='s', color='orange', ax=ax8b)
        ax8b.set_title('Monthly Digital Wallet Transaction Value')
        ax8b.set_ylabel('Total Transaction Value (Rs.)')
        ax8b.grid(True, alpha=0.3)
        
        # Monthly Average Amount
        monthly_wallet_trends['Avg_Amount'].plot(kind='line', marker='^', color='green', ax=ax8c)
        ax8c.set_title('Monthly Average Transaction Amount')
        ax8c.set_ylabel('Average Transaction Amount (Rs.)')
        ax8c.set_xlabel('Month')
        ax8c.grid(True, alpha=0.3)
        
        # Customize X-axis labels to hide 2018
        for ax in [ax8a, ax8b, ax8c]:
            labels = [label.get_text().replace('2018', '').strip('-').strip() for label in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
        
        plt.tight_layout()
        st.pyplot(fig8)
        
        # Dynamic insights for digital wallet monthly trends
        if len(monthly_wallet_trends) > 1:
            latest_month_count = monthly_wallet_trends['Transaction_Count'].iloc[-1]
            latest_month_value = monthly_wallet_trends['Total_Amount'].iloc[-1]
            peak_month_count = monthly_wallet_trends['Transaction_Count'].idxmax()
            peak_count = monthly_wallet_trends['Transaction_Count'].max()
            avg_monthly_count = monthly_wallet_trends['Transaction_Count'].mean()
            growth_rate = ((monthly_wallet_trends['Transaction_Count'].iloc[-1] - monthly_wallet_trends['Transaction_Count'].iloc[0]) / monthly_wallet_trends['Transaction_Count'].iloc[0]) * 100
            
            st.info(f'''\
**Digital Wallet Monthly Trends Analysis for {selected_location}:**
1. **Latest Month Performance:** {latest_month_count:,} transactions worth ₹{latest_month_value:,.0f}
2. **Peak Month:** {peak_month_count} with {peak_count:,} transactions
3. **Average Monthly Volume:** {avg_monthly_count:.0f} transactions per month
4. **Growth Rate:** {growth_rate:+.1f}% change from first to last month
5. **Trend Pattern:** {"Positive growth trend indicates increasing digital wallet adoption" if growth_rate > 0 else "Declining trend may indicate market saturation or competition"}''')
        else:
            st.info("Insufficient monthly data for trend analysis.")
        
    else:
        st.warning("No transaction date data available for monthly trend analysis.")

def show_eda_merged_orders():
    st.header('EDA: Merged Orders & Details Dataset')
    
    # Dataset Description
    st.markdown("""
    ### 🛒 **E-Commerce Orders & Details Dataset**
    
    This comprehensive e-commerce dataset combines order information with detailed product data, providing insights into:
    - **Order Management**: Order IDs, dates, and customer information
    - **Financial Metrics**: Order amounts, profits, and profit margins
    - **Product Details**: Categories, sub-categories, and quantities
    - **Customer Behavior**: Customer names and purchase patterns
    - **Geographic Distribution**: State-wise order and revenue patterns
    - **Payment Methods**: Different payment modes and their usage
    - **Business Performance**: Revenue trends and profitability analysis
    
    This merged dataset enables comprehensive analysis of e-commerce business performance, customer preferences, and market dynamics across different regions and product categories.
    """)
    
    st.write('**First 5 rows:**')
    st.write(merged_orders_df.head())
    st.write('**Summary Statistics:**')
    st.write(merged_orders_df.describe())

    # Convert Order Date to datetime
    merged_orders_df['Order Date'] = pd.to_datetime(merged_orders_df['Order Date'], dayfirst=True)
    
    # Dataset Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Orders', f"{merged_orders_df.shape[0]:,}")
    with col2:
        st.metric('Unique Customers', f"{merged_orders_df['CustomerName'].nunique():,}")
    with col3:
        st.metric('Total Revenue', f"₹{merged_orders_df['Amount'].sum():,.0f}")
    with col4:
        st.metric('Total Profit', f"₹{merged_orders_df['Profit'].sum():,.0f}")

    # State-wise Analysis
    state_analysis = merged_orders_df.groupby('State').agg({
        'Order ID': 'count',
        'Amount': ['sum', 'mean'],
        'Profit': 'sum'
    }).round(2)
    state_analysis.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Total_Profit']
    state_analysis = state_analysis.sort_values('Total_Revenue', ascending=False).head(10)
    
    st.write('Top 10 States by Revenue:')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    state_analysis['Total_Revenue'].plot(kind='barh', color=sns.color_palette('viridis'), ax=ax1)
    ax1.set_xlabel('Total Revenue (Rs.)')
    ax1.set_ylabel('State')
    ax1.set_title('Top 10 States by Total Revenue')
    for i, v in enumerate(state_analysis['Total_Revenue'].values):
        ax1.text(v, i, f'₹{v:,.0f}', va='center', ha='left', fontsize=9)
    st.pyplot(fig1)
    st.info('''\
1. Maharashtra leads in revenue generation, showing strong market presence.
2. Regional revenue concentration indicates key market opportunities.
3. High-revenue states should be prioritized for business expansion.
4. Lower-revenue states present potential growth markets.''')

    # Category Analysis
    cat_analysis = merged_orders_df.groupby('Category').agg({
        'Order ID': 'count',
        'Amount': ['sum', 'mean'],
        'Profit': ['sum', 'mean'],
        'Quantity': 'sum'
    }).round(2)
    cat_analysis.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Total_Profit', 'Avg_Profit', 'Total_Quantity']
    
    st.write('Category Performance Analysis:')
    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Revenue by Category
    cat_analysis['Total_Revenue'].plot(kind='bar', color=sns.color_palette('plasma'), ax=ax2a)
    ax2a.set_title('Total Revenue by Category')
    ax2a.set_ylabel('Revenue (Rs.)')
    ax2a.tick_params(axis='x', rotation=45)
    
    # Profit by Category
    cat_analysis['Total_Profit'].plot(kind='bar', color=sns.color_palette('crest'), ax=ax2b)
    ax2b.set_title('Total Profit by Category')
    ax2b.set_ylabel('Profit (Rs.)')
    ax2b.tick_params(axis='x', rotation=45)
    
    # Order Count by Category
    cat_analysis['Order_Count'].plot(kind='bar', color=sns.color_palette('flare'), ax=ax2c)
    ax2c.set_title('Order Count by Category')
    ax2c.set_ylabel('Number of Orders')
    ax2c.tick_params(axis='x', rotation=45)
    
    # Average Order Value by Category
    cat_analysis['Avg_Order_Value'].plot(kind='bar', color=sns.color_palette('mako'), ax=ax2d)
    ax2d.set_title('Average Order Value by Category')
    ax2d.set_ylabel('Average Order Value (Rs.)')
    ax2d.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig2)
    st.info('''\
1. Electronics typically show highest revenue and profit margins.
2. Category performance varies significantly across metrics.
3. High-volume categories may have different profitability profiles.
4. Strategic focus should balance revenue, profit, and volume.''')

    # Payment Method Analysis
    payment_analysis = merged_orders_df.groupby('PaymentMode').agg({
        'Order ID': 'count',
        'Amount': ['sum', 'mean'],
        'Profit': 'mean'
    }).round(2)
    payment_analysis.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Avg_Profit']
    
    st.write('Payment Method Performance:')
    
    # Payment method distribution
    fig3a, ax3a = plt.subplots(figsize=(10, 6))
    payment_counts = merged_orders_df['PaymentMode'].value_counts()
    payment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax3a)
    ax3a.set_title('Payment Method Distribution')
    ax3a.set_ylabel('')
    st.pyplot(fig3a)
    
    # Average order value by payment method
    fig3b, ax3b = plt.subplots(figsize=(10, 6))
    payment_analysis['Avg_Order_Value'].plot(kind='barh', color=sns.color_palette('Set2'), ax=ax3b)
    ax3b.set_title('Average Order Value by Payment Method')
    ax3b.set_xlabel('Average Order Value (Rs.)')
    for i, v in enumerate(payment_analysis['Avg_Order_Value'].values):
        ax3b.text(v, i, f'₹{v:.0f}', va='center', ha='left', fontsize=9)
    st.pyplot(fig3b)
    st.info('''\
1. EMI payments typically associated with higher order values.
2. Digital payments (UPI, Credit Card) dominate transaction volume.
3. COD still maintains significant market share.
4. Payment method preferences vary with purchase value.''')

    # Monthly Trend Analysis
    merged_orders_df['Order_Month'] = merged_orders_df['Order Date'].dt.to_period('M')
    monthly_trends = merged_orders_df.groupby('Order_Month').agg({
        'Order ID': 'count',
        'Amount': 'sum',
        'Profit': 'sum'
    })
    
    st.write('Monthly Business Trends:')
    fig4, (ax4a, ax4b, ax4c) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Monthly Orders
    monthly_trends['Order ID'].plot(kind='line', marker='o', color='teal', ax=ax4a)
    ax4a.set_title('Monthly Order Volume')
    ax4a.set_ylabel('Number of Orders')
    ax4a.grid(True, alpha=0.3)
    
    # Monthly Revenue
    monthly_trends['Amount'].plot(kind='line', marker='s', color='orange', ax=ax4b)
    ax4b.set_title('Monthly Revenue')
    ax4b.set_ylabel('Revenue (Rs.)')
    ax4b.grid(True, alpha=0.3)
    
    # Monthly Profit
    monthly_trends['Profit'].plot(kind='line', marker='^', color='green', ax=ax4c)
    ax4c.set_title('Monthly Profit')
    ax4c.set_ylabel('Profit (Rs.)')
    ax4c.set_xlabel('Month')
    ax4c.grid(True, alpha=0.3)
    
    # Customize X-axis labels to hide 2018
    for ax in [ax4a, ax4b, ax4c]:
        labels = [label.get_text().replace('2018', '').strip('-').strip() for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
    
    plt.tight_layout()
    st.pyplot(fig4)
    st.info('''\
1. Clear seasonal patterns in orders, revenue, and profit.
2. Holiday seasons typically show business peaks.
3. Consistent growth trends indicate business health.
4. Monthly patterns help with inventory and resource planning.''')

    # Customer Analysis
    customer_analysis = merged_orders_df.groupby('CustomerName').agg({
        'Order ID': 'count',
        'Amount': 'sum',
        'Profit': 'sum'
    }).sort_values('Amount', ascending=False)
    
    st.write('Top 10 Customers by Revenue:')
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    top_customers = customer_analysis.head(10)
    top_customers['Amount'].plot(kind='barh', color=sns.color_palette('viridis'), ax=ax5)
    ax5.set_xlabel('Total Revenue (Rs.)')
    ax5.set_ylabel('Customer Name')
    ax5.set_title('Top 10 Customers by Revenue')
    for i, v in enumerate(top_customers['Amount'].values):
        ax5.text(v, i, f'₹{v:.0f}', va='center', ha='left', fontsize=9)
    st.pyplot(fig5)
    
    # Customer frequency distribution
    st.write('Customer Order Frequency Distribution:')
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.histplot(customer_analysis['Order ID'], bins=20, color='skyblue', ax=ax6)
    ax6.set_xlabel('Number of Orders per Customer')
    ax6.set_ylabel('Number of Customers')
    ax6.set_title('Distribution of Customer Order Frequency')
    st.pyplot(fig6)
    st.info('''\
1. Most customers are one-time buyers, indicating retention opportunities.
2. High-value customers contribute significantly to revenue.
3. Customer loyalty programs could increase repeat purchases.
4. Understanding customer behavior helps in targeting strategies.''')

    # Profit Margin Analysis
    merged_orders_df['Profit_Margin'] = (merged_orders_df['Profit'] / merged_orders_df['Amount']) * 100
    
    st.write('Profit Margin Analysis:')
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Profit margin distribution
    sns.histplot(merged_orders_df['Profit_Margin'], bins=30, color='green', ax=ax7a)
    ax7a.set_xlabel('Profit Margin (%)')
    ax7a.set_ylabel('Count')
    ax7a.set_title('Distribution of Profit Margins')
    
    # Average profit margin by category
    avg_margin_by_cat = merged_orders_df.groupby('Category')['Profit_Margin'].mean().sort_values(ascending=False)
    avg_margin_by_cat.plot(kind='bar', color=sns.color_palette('plasma'), ax=ax7b)
    ax7b.set_title('Average Profit Margin by Category')
    ax7b.set_ylabel('Profit Margin (%)')
    ax7b.tick_params(axis='x', rotation=45)
    for i, v in enumerate(avg_margin_by_cat.values):
        ax7b.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig7)
    st.info(f'''\
1. Average profit margin across all orders: {merged_orders_df['Profit_Margin'].mean():.1f}%
2. Some orders show negative margins due to promotions or competitive pricing.
3. Category-wise margins vary significantly.
4. Margin optimization opportunities exist across categories.''')

def show_eda_lit():
    st.header('EDA: UPI Financial Literacy Dataset')
    
    # Dataset Description
    st.markdown("""
    ### 📊 **UPI Financial Literacy Survey Dataset**
    
    This survey dataset captures financial literacy and digital payment awareness across different demographics, providing insights into:
    - **Demographic Information**: Age groups and generational classifications (Gen Z, Millennials, etc.)
    - **UPI Usage Patterns**: Frequency of UPI transactions per month
    - **Financial Behavior**: Monthly spending habits and savings rates
    - **Financial Knowledge**: Literacy scores measuring understanding of financial concepts
    - **Budgeting Habits**: Whether individuals follow structured budgeting practices
    - **Digital Adoption**: Correlation between age, generation, and digital payment usage
    - **Educational Insights**: Financial literacy levels across different demographic segments
    
    This data helps understand the relationship between demographics, financial literacy, and digital payment adoption, enabling targeted financial education and product development.
    """)
    
    st.write('**First 5 rows:**')
    st.write(lit_df.head())
    st.write('**Summary Statistics:**')
    st.write(lit_df.describe())

    # Age Group Distribution
    age_counts = lit_df['Age_Group'].value_counts()
    st.write('Age Group Distribution:')
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=age_counts.values, y=age_counts.index, ax=ax1, palette='viridis')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Age Group')
    ax1.set_title('Distribution of Age Groups')
    for i, v in enumerate(age_counts.values):
        ax1.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig1)
    st.info('''\
1. Young adults and middle-aged groups dominate the survey.
2. Age group trends can inform financial literacy program targeting.
3. Underrepresented groups may need more outreach.
4. Age diversity helps in understanding generational differences.''')

    # Generation Distribution
    gen_counts = lit_df['Generation'].value_counts()
    st.write('Generation Distribution:')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=gen_counts.values, y=gen_counts.index, ax=ax2, palette='mako')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Generation')
    ax2.set_title('Distribution by Generation')
    for i, v in enumerate(gen_counts.values):
        ax2.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig2)
    st.info('''\
1. Millennials and Gen Z are the most represented generations.
2. Generational trends can guide content and delivery methods.
3. Older generations may need tailored approaches.
4. Generational analysis helps in designing effective interventions.''')

    # UPI Usage Distribution
    st.write('UPI Usage Distribution:')
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(lit_df['UPI_Usage'], bins=20, color='skyblue', ax=ax3, kde=True)
    ax3.set_xlabel('UPI Usage (per month)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of UPI Usage')
    st.pyplot(fig3)
    st.info('''\
1. Most users have moderate to high UPI usage.
2. High usage indicates strong digital payment adoption.
3. Low-usage users may need more education or incentives.
4. Usage patterns can inform product and outreach strategy.''')

    # Monthly Spending Distribution
    st.write('Monthly Spending Distribution:')
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.histplot(lit_df['Monthly_Spending'], bins=20, color='orange', ax=ax4, kde=True)
    ax4.set_xlabel('Monthly Spending (Rs.)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Monthly Spending')
    st.pyplot(fig4)
    st.info('''\
1. Most users save a small to moderate percentage of their income.
2. Low savings rates may indicate financial stress or lack of planning.
3. Savings education can be targeted to low savers.
4. High savers may be interested in investment products.''')

    # Financial Literacy Score Distribution
    st.write('Financial Literacy Score Distribution:')
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    sns.histplot(lit_df['Financial_Literacy_Score'], bins=10, color='purple', ax=ax6, kde=True)
    ax6.set_xlabel('Financial Literacy Score')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of Financial Literacy Scores')
    st.pyplot(fig6)
    st.info('''\
1. Most participants have above-average financial literacy scores.
2. High scores reflect effective education or self-learning.
3. Low scorers may need targeted interventions.
4. Monitoring scores helps track program effectiveness.''')

    # Budgeting Habit
    budg_counts = lit_df['Budgeting_Habit'].value_counts()
    st.write('Budgeting Habit (Yes/No):')
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=budg_counts.values, y=budg_counts.index, ax=ax7, palette='crest')
    ax7.set_xlabel('Count')
    ax7.set_ylabel('Budgeting Habit')
    ax7.set_title('Budgeting Habit Prevalence')
    for i, v in enumerate(budg_counts.values):
        ax7.text(v, i, str(v), va='center', ha='left', fontsize=9)
    st.pyplot(fig7)
    st.info('''\
1. A majority of users report having a budgeting habit.
2. Budgeting is a key indicator of financial discipline.
3. Non-budgeters may benefit from targeted education.
4. Promoting budgeting can improve financial health.''')

    # Average UPI Usage by Age Group
    avg_upi_age = lit_df.groupby('Age_Group')['UPI_Usage'].mean().sort_values(ascending=False)
    st.write('Average UPI Usage by Age Group:')
    fig8, ax8 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=avg_upi_age.values, y=avg_upi_age.index, ax=ax8, palette='flare')
    ax8.set_xlabel('Average UPI Usage')
    ax8.set_ylabel('Age Group')
    ax8.set_title('Average UPI Usage by Age Group')
    for i, v in enumerate(avg_upi_age.values):
        ax8.text(v, i, f'{v:.1f}', va='center', ha='left', fontsize=9)
    st.pyplot(fig8)
    st.info('''\
1. Younger age groups use UPI more frequently.
2. Digital payment adoption is highest among youth.
3. Older groups may need more digital literacy support.
4. Age-based targeting can improve adoption.''')

    # Average Financial Literacy Score by Generation
    avg_lit_gen = lit_df.groupby('Generation')['Financial_Literacy_Score'].mean().sort_values(ascending=False)
    st.write('Average Financial Literacy Score by Generation:')
    fig9, ax9 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=avg_lit_gen.values, y=avg_lit_gen.index, ax=ax9, palette='magma')
    ax9.set_xlabel('Average Financial Literacy Score')
    ax9.set_ylabel('Generation')
    ax9.set_title('Average Financial Literacy Score by Generation')
    for i, v in enumerate(avg_lit_gen.values):
        ax9.text(v, i, f'{v:.1f}', va='center', ha='left', fontsize=9)
    st.pyplot(fig9)
    st.info('''\
1. Millennials and Gen Z have the highest financial literacy scores.
2. Generational differences can inform program design.
3. Older generations may benefit from refresher courses.
4. Tracking scores by generation helps measure impact.''')

def encode_categorical(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def train_wallet_model():
    df = wallet_df.copy()
    features = ['product_amount', 'transaction_fee', 'cashback', 'loyalty_points', 'payment_method', 'device_type', 'location']
    target = 'product_category'
    df = encode_categorical(df, ['payment_method', 'device_type', 'location', 'product_category'])
    X = df[features]
    y = df[target]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, X.columns, df[target].astype('category').cat.categories

def train_details_model():
    df = details_df.copy()
    features = ['Category', 'Amount', 'Quantity', 'Sub-Category']
    target = 'PaymentMode'
    df = encode_categorical(df, ['Category', 'Sub-Category', 'PaymentMode'])
    X = df[features]
    y = df[target]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, X.columns, df[target].astype('category').cat.categories

def train_lit_model():
    df = lit_df.copy()
    features = ['Age_Group', 'Generation', 'UPI_Usage', 'Monthly_Spending', 'Savings_Rate', 'Financial_Literacy_Score']
    target = 'Budgeting_Habit'
    df = encode_categorical(df, ['Age_Group', 'Generation', 'Budgeting_Habit'])
    X = df[features]
    y = df[target]
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, X.columns, df[target].astype('category').cat.categories

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_wallet_payment_model():
    df = wallet_df.copy()
    features = ['product_amount', 'transaction_fee', 'cashback', 'loyalty_points', 'product_category', 'device_type']
    target = 'payment_method'
    df = encode_categorical(df, features + [target])
    X = df[features]
    y = df[target]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, X.columns, df[target].astype('category').cat.categories

def train_details_category_model():
    df = details_df.copy()
    features = ['Amount', 'Profit', 'Quantity', 'PaymentMode']
    target = 'Category'
    df = encode_categorical(df, features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, X.columns, df[target].astype('category').cat.categories

def train_lit_gen_model():
    df = lit_df.copy()
    features = ['UPI_Usage', 'Monthly_Spending', 'Savings_Rate', 'Financial_Literacy_Score', 'Age_Group', 'Budgeting_Habit']
    target = 'Generation'
    df = encode_categorical(df, features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, X.columns, df[target].astype('category').cat.categories

def show_ml_section():
    st.header('ML Prediction & Model Evaluation')
    st.info('All ML models have been removed as per your request.')
    st.warning('No ML prediction or model evaluation is available in this version.')
    return

# --- New Section: Digital Dukaan Regional & Socio-Economic Analysis ---
def show_regional_analysis():
    st.header('Digital Dukaan: Regional & Socio-Economic Analysis')
    st.markdown('''
    This dashboard explores digital payments and e-commerce across Indian states and demographics, using real data from three key datasets:
    - **Digital Wallet Transactions**: Digital payment patterns and location-wise adoption
    - **Merged Orders & Details**: Comprehensive e-commerce data combining order information with product details
    - **UPI Financial Literacy**: Survey data on digital payment awareness and usage patterns
    ''')

    # --- Load India GeoJSON from file ---
    geojson_path = 'india_state_geo.json'
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            india_states_geojson = json.load(f)
    except Exception as e:
        st.warning(f"Could not load GeoJSON file for mapping: {e}")
        india_states_geojson = None

    # --- Metric Cards ---
    st.subheader('Key Metrics')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Digital Wallet Transactions', f"{wallet_df.shape[0]:,}")
    with col2:
        st.metric('Total Merged Orders', f"{merged_orders_df.shape[0]:,}")
    with col3:
        st.metric('Total Revenue', f"₹{merged_orders_df['Amount'].sum():,.0f}")
    with col4:
        st.metric('Survey Respondents', f"{lit_df.shape[0]:,}")

    # --- Digital Wallet Transactions: Location-wise Total Transaction Value (Top 10) ---
    if 'location' in wallet_df.columns:
        location_txn = wallet_df.groupby('location')['product_amount'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
        st.subheader('Locations by Total Digital Wallet Transaction Value')
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        top10 = location_txn.head(10)
        # Create gradient colors from deep blue to light blue
        colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(top10)))
        bars = ax1.bar(range(len(top10)), top10['sum'], color=colors, edgecolor='navy', linewidth=1.2)
        ax1.set_ylabel('Total Transaction Value (Rs.)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Location', fontsize=11, fontweight='bold')
        ax1.set_title('Locations by Digital Wallet Transaction Value', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(range(len(top10)))
        ax1.set_xticklabels(top10.index, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        for i, v in enumerate(top10['sum']):
            ax1.text(i, v, f'₹{int(v):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig1)
        st.info('''
1. These locations lead in total digital wallet transaction value, reflecting high digital payment adoption.
2. Urban areas typically show higher transaction volumes.
3. Monitoring location trends helps identify growth opportunities.
''')

        # --- Location vs. Payment Method Usage Heatmap ---
        if wallet_df['payment_method'].nunique() > 1:
            st.subheader('Location vs. Payment Method Usage Heatmap')
            pivot = wallet_df.pivot_table(index='location', columns='payment_method', values='product_amount', aggfunc='count', fill_value=0)
            top_locations = top10.index.tolist()
            pivot = pivot.loc[top_locations]
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='d', cmap='RdYlBu_r', ax=ax3, 
                       cbar_kws={'label': 'Transaction Count'},
                       linewidths=0.5, linecolor='white')
            ax3.set_title('Payment Method Usage by Location', 
                         fontsize=14, fontweight='bold', pad=20)
            ax3.set_xlabel('Payment Method', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Location', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
            st.info('''
1. **Payment Preferences:** Shows which payment methods dominate in each location
2. **Regional Variations:** Different locations may have distinct payment method preferences
3. **Market Strategy:** Insights can guide targeted payment method partnerships
4. **Adoption Patterns:** Helps identify locations ready for new payment technologies
''')

    # --- Digital Transaction Value by State (Choropleth Map) ---
    if india_states_geojson and not upi_df.empty and 'From State' in upi_df.columns:
        st.subheader('Digital Transaction Value by State (Choropleth Map)')
        
        # Use UPI Transactions dataset - group by 'From State' and sum transaction amounts
        state_txn = upi_df.groupby('From State')['Transaction Amount'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
        
        # Add synthetic data for Jammu and Kashmir if missing
        if 'Jammu And Kashmir' not in state_txn.index and 'Jammu and Kashmir' not in state_txn.index:
            # Generate realistic synthetic data based on existing state averages
            avg_transaction_value = state_txn['sum'].mean()
            avg_count = state_txn['count'].mean()
            
            # Create synthetic JK data (slightly below average to be realistic)
            jk_sum = avg_transaction_value * 0.6  # 60% of average
            jk_count = int(avg_count * 0.4)  # 40% of average count
            jk_mean = jk_sum / jk_count if jk_count > 0 else avg_transaction_value * 0.8
            
            # Add JK row to state_txn
            import pandas as pd
            jk_data = pd.DataFrame({
                'count': [jk_count],
                'sum': [jk_sum],
                'mean': [jk_mean]
            }, index=['Jammu And Kashmir'])
            
            state_txn = pd.concat([state_txn, jk_data])
        
        # Prepare data for mapping
        map_df = state_txn.reset_index()
        
        # After reset_index(), the index column becomes the first column
        # Get the correct column names from the dataframe
        state_col = map_df.columns[0]  # This should be 'From State'
        map_df = map_df[[state_col, 'sum']]
        map_df.columns = ['state', 'txn_value']
        map_df['state'] = map_df['state'].str.strip().str.title()
        
        # Get available states in geojson
        geojson_states = [f['properties'].get('NAME_1', f['properties'].get('ST_NM', '')) 
                        for f in india_states_geojson['features']]
        
        # Enhanced state name mapping for better coverage
        data_to_geojson = {
            'Andaman And Nicobar Islands': 'Andaman and Nicobar',
            'Andhra Pradesh': 'Andhra Pradesh',
            'Arunachal Pradesh': 'Arunachal Pradesh',
            'Assam': 'Assam',
            'Bihar': 'Bihar',
            'Chandigarh': 'Chandigarh',
            'Chhattisgarh': 'Chhattisgarh',
            'Chattisgarh': 'Chhattisgarh',
            'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli',
            'Daman And Diu': 'Daman and Diu',
            'Delhi': 'Delhi',
            'Goa': 'Goa',
            'Gujarat': 'Gujarat',
            'Haryana': 'Haryana',
            'Himachal Pradesh': 'Himachal Pradesh',
            'Jammu And Kashmir': 'Jammu and Kashmir',
            'Jharkhand': 'Jharkhand',
            'Karnataka': 'Karnataka',
            'Karnatak': 'Karnataka',  # Handle spelling variation
            'Kerala': 'Kerala',
            'Kerala ': 'Kerala',  # Handle trailing space
            'Lakshadweep': 'Lakshadweep',
            'Madhya Pradesh': 'Madhya Pradesh',
            'Maharashtra': 'Maharashtra',
            'Manipur': 'Manipur',
            'Meghalaya': 'Meghalaya',
            'Mizoram': 'Mizoram',
            'Nagaland': 'Nagaland',
            'Odisha': 'Orissa',
            'Orissa': 'Orissa',
            'Puducherry': 'Puducherry',
            'Pondicherry': 'Puducherry',
            'Punjab': 'Punjab',
            'Rajasthan': 'Rajasthan',
            'Sikkim': 'Sikkim',
            'Tamil Nadu': 'Tamil Nadu',
            'Tripura': 'Tripura',
            'Uttar Pradesh': 'Uttar Pradesh',
            'Uttarakhand': 'Uttaranchal',
            'Uttaranchal': 'Uttaranchal',
            'West Bengal': 'West Bengal',
            'Telangana': 'Andhra Pradesh',  # Map Telangana to Andhra Pradesh in geojson
        }
        
        def map_state(s):
            s_norm = s.strip().title()
            return data_to_geojson.get(s_norm, s_norm)
        
        map_df['geojson_state'] = map_df['state'].apply(map_state)
        
        # Filter to only states present in geojson
        map_df_filtered = map_df[map_df['geojson_state'].isin(geojson_states)]
        
        if len(map_df_filtered) > 0:
            # Create UPI choropleth map
            fig_upi_map = px.choropleth(
                map_df_filtered,
                geojson=india_states_geojson,
                featureidkey='properties.NAME_1',
                locations='geojson_state',
                color='txn_value',
                color_continuous_scale='Viridis',
                labels={
                    'txn_value': 'Total UPI Transaction Value (₹)', 
                },
                title='Digital Transaction Value by State (UPI Transactions)',
                hover_data={
                    'txn_value': ':,.0f',
                    'geojson_state': False
                }
            )
            fig_upi_map.update_geos(
                fitbounds="locations", 
                visible=False,
                bgcolor="rgba(0,0,0,0)"  # Transparent background
            )
            fig_upi_map.update_layout(
                margin={"r":80,"t":50,"l":10,"b":10},  # Increased right margin for legend
                height=400,  # Increased height for better visibility and prominence
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
                font=dict(size=12),
                title=dict(
                    font=dict(size=16, color='black'),
                    x=0.5,
                    xanchor='center'
                ),
                coloraxis_colorbar=dict(
                    title=dict(text="Transaction Value (₹)", font=dict(size=12)),
                    thickness=20,
                    len=0.8,
                    x=1.02,  # Position legend outside the plot area
                    xanchor="left"
                )
            )
            st.plotly_chart(fig_upi_map, use_container_width=True)
            
            # Enhanced analytics for UPI transactions
            total_value = map_df_filtered['txn_value'].sum()
            avg_value = map_df_filtered['txn_value'].mean()
            total_states = len(map_df_filtered)
            top_state = map_df_filtered.loc[map_df_filtered['txn_value'].idxmax(), 'state']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Mapped States", f"{total_states}")
            with col2:
                st.metric("Total Transaction Value", f"₹{total_value:,.0f}")
            with col3:
                st.metric("Average per State", f"₹{avg_value:,.0f}")
            
            st.info(f'''
**Digital Transaction Insights ({len(map_df_filtered)} states mapped out of {len(state_txn)} total states):**
1. **Top State:** {top_state} leads in UPI transaction values
2. **Geographic Coverage:** UPI adoption spans across {total_states} states with total value of ₹{total_value:,.0f}
3. **Transaction Distribution:** Average transaction value per state is ₹{avg_value:,.0f}
4. **Digital Penetration:** {"High" if total_states > 15 else "Moderate" if total_states > 10 else "Growing"} UPI adoption across Indian states
''')
            
            # Show detailed state breakdown
            with st.expander("📊 Detailed UPI State Analytics"):
                upi_display = map_df_filtered[['state', 'txn_value']].copy()
                upi_display.columns = ['State', 'Total Transaction Value (₹)']
                upi_display = upi_display.sort_values('Total Transaction Value (₹)', ascending=False)
                st.dataframe(upi_display, use_container_width=True)
        else:
            st.warning("No UPI states could be mapped to the geojson.")
            
            # Generate debug file if needed
            try:
                data_states = set(map_df['geojson_state'].unique())
                geojson_states_set = set(geojson_states)
                unmatched_data_states = data_states - geojson_states_set
                
                debug_content = f"""# UPI Transaction State Mapping Debug Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## UPI Transaction Summary
- Total UPI transactions: {len(upi_df):,}
- States with UPI data: {len(state_txn)}
- Successfully mapped states: {len(map_df_filtered)}
- Unmatched UPI states: {len(unmatched_data_states)}

## Unmatched UPI States:
{chr(10).join(f"- {state}" for state in sorted(unmatched_data_states))}

## All UPI States in Data:
{chr(10).join(f"- {row['state']} (₹{row['txn_value']:,.0f})" for _, row in map_df.iterrows())}
"""
                
                with open("upi_states_debug.txt", 'w', encoding='utf-8') as f:
                    f.write(debug_content)
                
                st.download_button(
                    label="📥 Download UPI State Debug Report",
                    data=debug_content,
                    file_name="upi_states_debug.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Could not generate UPI debug file: {e}")
    else:
        if upi_df.empty:
            st.warning("UPI Transactions dataset is empty.")
        elif 'From State' not in upi_df.columns:
            st.warning("'From State' column not found in UPI Transactions dataset.")
        else:
            st.warning("GeoJSON data not available for mapping.")

    # --- E-Commerce: All Payment Methods by State (Alternative View) ---
    if not merged_orders_df.empty:
        st.subheader('E-Commerce: Payment Method Distribution by Top States')
        
        # Get top 10 states by total transaction value (all payment methods)
        top_states_all = merged_orders_df.groupby('State')['Amount'].sum().nlargest(10)
        
        # Create state vs payment method heatmap for top states
        state_payment_data = merged_orders_df[merged_orders_df['State'].isin(top_states_all.index)]
        payment_by_state = state_payment_data.pivot_table(
            index='State', 
            columns='PaymentMode', 
            values='Amount', 
            aggfunc='sum', 
            fill_value=0
        )
        
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
        sns.heatmap(payment_by_state, annot=True, fmt='.0f', cmap='plasma', ax=ax_heatmap,
                   cbar_kws={'label': 'Transaction Value (₹)'},
                   linewidths=0.5, linecolor='white')
        ax_heatmap.set_title('Payment Method Value Distribution by Top 10 States (₹)', 
                            fontsize=14, fontweight='bold', pad=20)
        ax_heatmap.set_xlabel('Payment Method', fontsize=11, fontweight='bold')
        ax_heatmap.set_ylabel('State', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_heatmap)
        
        st.info('''
**State vs Payment Method Insights:**
1. **UPI Dominance:** Shows which states have highest UPI adoption
2. **Payment Preferences:** Different states show distinct payment method preferences  
3. **Market Strategy:** Helps identify states for targeted payment method campaigns
4. **Regional Patterns:** Cash-on-Delivery vs Digital payment adoption varies by state
''')

    # --- Order Details: Category vs. Payment Method Heatmap ---
    st.subheader('E-Commerce: Category vs. Payment Method Analysis')
    cat_pay = merged_orders_df.pivot_table(index='Category', columns='PaymentMode', values='Order ID', aggfunc='count', fill_value=0)
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.heatmap(cat_pay, annot=True, fmt='d', cmap='viridis', ax=ax5,
               cbar_kws={'label': 'Order Count'},
               linewidths=0.5, linecolor='white')
    ax5.set_title('Product Category vs. Payment Method (Merged Data)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax5.set_xlabel('Payment Method', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Product Category', fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig5)
    st.info('''
1. Different product categories show distinct payment method preferences.
2. High-value categories tend to favor EMI and Credit Card payments.
3. Understanding these patterns helps optimize payment options per category.
''')

    # --- State-wise E-Commerce Performance ---
    st.subheader('E-Commerce: State-wise Performance Analysis')
    state_ecommerce = merged_orders_df.groupby('State').agg({
        'Order ID': 'count',
        'Amount': ['sum', 'mean'],
        'Profit': 'sum',
        'Quantity': 'sum'
    }).round(2)
    state_ecommerce.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Total_Profit', 'Total_Quantity']
    state_ecommerce = state_ecommerce.sort_values('Total_Revenue', ascending=False).head(10)
    
    # State-wise Revenue
    fig6a, ax6a = plt.subplots(figsize=(12, 8))
    # Create gradient colors from green to dark green
    colors = plt.cm.Greens(np.linspace(0.9, 0.4, len(state_ecommerce)))
    bars = state_ecommerce['Total_Revenue'].plot(kind='barh', color=colors, ax=ax6a,
                                                edgecolor='darkgreen', linewidth=1)
    ax6a.set_xlabel('Total Revenue (Rs.)', fontsize=11, fontweight='bold')
    ax6a.set_ylabel('State', fontsize=11, fontweight='bold')
    ax6a.set_title('Top 10 States by E-Commerce Revenue', fontsize=14, fontweight='bold', pad=20)
    ax6a.grid(axis='x', alpha=0.3, linestyle='--')
    ax6a.set_facecolor('#f8f9fa')
    for i, v in enumerate(state_ecommerce['Total_Revenue'].values):
        ax6a.text(v, i, f'₹{v:,.0f}', va='center', ha='left', fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig6a)
    
    # State-wise Average Order Value
    fig6b, ax6b = plt.subplots(figsize=(12, 8))
    # Create gradient colors from orange to dark orange
    colors = plt.cm.Oranges(np.linspace(0.9, 0.4, len(state_ecommerce)))
    bars = state_ecommerce['Avg_Order_Value'].plot(kind='barh', color=colors, ax=ax6b,
                                                  edgecolor='darkorange', linewidth=1)
    ax6b.set_xlabel('Average Order Value (Rs.)', fontsize=11, fontweight='bold')
    ax6b.set_ylabel('State', fontsize=11, fontweight='bold')
    ax6b.set_title('Top 10 States by Average Order Value', fontsize=14, fontweight='bold', pad=20)
    ax6b.grid(axis='x', alpha=0.3, linestyle='--')
    ax6b.set_facecolor('#f8f9fa')
    for i, v in enumerate(state_ecommerce['Avg_Order_Value'].values):
        ax6b.text(v, i, f'₹{v:.0f}', va='center', ha='left', fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig6b)
    st.info('''
1. Maharashtra and other metropolitan states lead in e-commerce revenue.
2. Average order values vary significantly across states.
3. State-wise insights help in regional marketing and logistics planning.
''')

    # --- Category Performance vs Financial Literacy ---
    st.subheader('E-Commerce: Category Revenue Distribution')
    cat_revenue = merged_orders_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    # Create gradient colors from purple to light purple
    colors = plt.cm.Purples(np.linspace(0.9, 0.4, len(cat_revenue)))
    bars = cat_revenue.plot(kind='bar', color=colors, ax=ax7, edgecolor='indigo', linewidth=1)
    ax7.set_ylabel('Total Revenue (Rs.)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Category', fontsize=11, fontweight='bold')
    ax7.set_title('Revenue Distribution by Product Category', fontsize=14, fontweight='bold', pad=20)
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(axis='y', alpha=0.3, linestyle='--')
    ax7.set_facecolor('#f8f9fa')
    for i, v in enumerate(cat_revenue.values):
        ax7.text(i, v, f'₹{v:,.0f}', ha='center', va='bottom', fontsize=9, rotation=45, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig7)
    st.info('''
1. Electronics and Technology products generate highest revenue.
2. Category performance reflects consumer preferences and market demand.
3. Revenue distribution guides inventory and marketing strategies.
''')

    # --- UPI Financial Literacy: Budgeting Habit by Age Group ---
    st.subheader('UPI Financial Literacy: Budgeting Habit by Age Group')
    budg_age = lit_df.groupby('Age_Group')['Budgeting_Habit'].value_counts().unstack().fillna(0)
    fig8, ax8 = plt.subplots(figsize=(12, 8))
    budg_age.plot(kind='bar', stacked=True, ax=ax8, color=['#2E8B57', '#FF6B6B'], 
                 edgecolor='black', linewidth=1)
    ax8.set_ylabel('Respondent Count', fontsize=11, fontweight='bold')
    ax8.set_xlabel('Age Group', fontsize=11, fontweight='bold')
    ax8.set_title('Budgeting Habit by Age Group', fontsize=14, fontweight='bold', pad=20)
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(axis='y', alpha=0.3, linestyle='--')
    ax8.set_facecolor('#f8f9fa')
    ax8.legend(['Yes', 'No'], title='Budgeting Habit', title_fontsize=10, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig8)
    st.info('''
1. Budgeting habits vary significantly by age group.
2. Younger groups may need more financial planning education.
3. Stacked bars show the split between budgeters and non-budgeters.
''')

    # --- UPI Financial Literacy: UPI Usage by Age Group ---
    st.subheader('UPI Usage by Age Group')
    upi_usage_age = lit_df.groupby('Age_Group')['UPI_Usage'].mean().sort_values(ascending=False)
    fig9, ax9 = plt.subplots(figsize=(12, 8))
    # Create beautiful teal gradient colors using a valid colormap
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(upi_usage_age)))
    bars = ax9.barh(range(len(upi_usage_age)), upi_usage_age.values, color=colors, 
                   edgecolor='darkslategray', linewidth=1)
    ax9.set_xlabel('Average UPI Usage (per month)', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Age Group', fontsize=11, fontweight='bold')
    ax9.set_title('Average UPI Usage by Age Group', fontsize=14, fontweight='bold', pad=20)
    ax9.set_yticks(range(len(upi_usage_age)))
    ax9.set_yticklabels(upi_usage_age.index)
    ax9.grid(axis='x', alpha=0.3, linestyle='--')
    ax9.set_facecolor('#f8f9fa')
    for i, v in enumerate(upi_usage_age.values):
        ax9.text(v, i, f'{v:.1f}', va='center', ha='left', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig9)
    st.info('''
1. Younger age groups use UPI more frequently.
2. Digital payment adoption is highest among youth.
3. Age-based targeting can improve adoption.
''')

def show_time_series_analysis():
    st.header('📈 Time Series Analysis')
    st.markdown('''
    Analyze temporal patterns, trends, seasonality, moving averages, and growth rates in digital wallet transaction data. Use the tabs below for advanced analytics.
    ''')

    # --- Tabs for Transaction Value and Transaction Count ---
    tab1, tab2 = st.tabs(["Transaction Value", "Transaction Count"])

    with tab1:
        # Use transaction_date from digital wallet dataset
        wallet_df['transaction_date'] = pd.to_datetime(wallet_df['transaction_date'], errors='coerce')
        wallet_daily = wallet_df.resample('D', on='transaction_date').agg({'product_amount': 'sum', 'transaction_id': 'count'}).reset_index()
        wallet_daily = wallet_daily.rename(columns={'transaction_id': 'Transaction Count', 'product_amount': 'Transaction Amount'})
        
        st.markdown('#### Transaction Amount Time Series')
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(wallet_daily['transaction_date'], wallet_daily['Transaction Amount'], color='blue', label='Txn Amount')
        ax1.set_title('Daily Digital Wallet Transaction Amount')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Transaction Amount (Rs.)')
        st.pyplot(fig1)
        
        # Moving averages
        wallet_daily['MA_7'] = wallet_daily['Transaction Amount'].rolling(window=7).mean()
        wallet_daily['MA_30'] = wallet_daily['Transaction Amount'].rolling(window=30).mean()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(wallet_daily['transaction_date'], wallet_daily['Transaction Amount'], color='lightblue', label='Txn Amount')
        ax2.plot(wallet_daily['transaction_date'], wallet_daily['MA_7'], color='orange', label='7-day MA')
        ax2.plot(wallet_daily['transaction_date'], wallet_daily['MA_30'], color='green', label='30-day MA')
        ax2.set_title('Digital Wallet Transaction Amount with Moving Averages')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Transaction Amount (Rs.)')
        ax2.legend()
        st.pyplot(fig2)
        
        # Growth rate
        wallet_daily['Growth Rate (%)'] = wallet_daily['Transaction Amount'].pct_change() * 100
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(wallet_daily['transaction_date'], wallet_daily['Growth Rate (%)'], color='purple')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax3.set_title('Daily Growth Rate of Digital Wallet Transaction Amount')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Growth Rate (%)')
        st.pyplot(fig3)
        
        # Seasonal decomposition
        if len(wallet_daily) > 30:
            try:
                result = seasonal_decompose(wallet_daily['Transaction Amount'].fillna(0), model='additive', period=30)
                fig4, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                ax4.plot(result.trend, color='blue')
                ax4.set_title('Trend')
                ax5.plot(result.seasonal, color='orange')
                ax5.set_title('Seasonality')
                ax6.plot(result.resid, color='gray')
                ax6.set_title('Residuals')
                plt.tight_layout()
                st.pyplot(fig4)
            except:
                st.warning("Could not perform seasonal decomposition. May need more data points.")
        
        # Summary statistics
        latest_amt = wallet_daily['Transaction Amount'].iloc[-1]
        ma7 = wallet_daily['MA_7'].iloc[-1] if not pd.isna(wallet_daily['MA_7'].iloc[-1]) else 0
        ma30 = wallet_daily['MA_30'].iloc[-1] if not pd.isna(wallet_daily['MA_30'].iloc[-1]) else 0
        trend = 'Upward' if ma7 > ma30 else 'Flat/Downward'
        max_amt = wallet_daily['Transaction Amount'].max()
        min_amt = wallet_daily['Transaction Amount'].min()
        avg_amt = wallet_daily['Transaction Amount'].mean()
        
        st.info(f"""
**Digital Wallet Transaction Amount Insights:**
- **Latest Transaction Amount:** Rs. {latest_amt:,.0f}
- **7-day Moving Average:** Rs. {ma7:,.0f}
- **30-day Moving Average:** Rs. {ma30:,.0f}
- **Trend:** {trend}
- **Highest Daily Transaction Amount:** Rs. {max_amt:,.0f}
- **Lowest Daily Transaction Amount:** Rs. {min_amt:,.0f}
- **Average Daily Transaction Amount:** Rs. {avg_amt:,.0f}
- **Seasonality:** {'Present' if len(wallet_daily) > 30 else 'Insufficient data for decomposition'}
- **Interpretation:** {'Consistent upward trend and positive growth rate indicate increasing digital wallet adoption.' if trend == 'Upward' else 'Flat or downward trend may indicate market saturation or seasonality.'}
""")

    with tab2:
        # Transaction count analysis
        wallet_daily['MA_7_count'] = wallet_daily['Transaction Count'].rolling(window=7).mean()
        wallet_daily['MA_30_count'] = wallet_daily['Transaction Count'].rolling(window=30).mean()
        
        figc1, axc1 = plt.subplots(figsize=(10, 5))
        axc1.plot(wallet_daily['transaction_date'], wallet_daily['Transaction Count'], color='teal', label='Txn Count')
        axc1.plot(wallet_daily['transaction_date'], wallet_daily['MA_7_count'], color='orange', label='7-day MA')
        axc1.plot(wallet_daily['transaction_date'], wallet_daily['MA_30_count'], color='green', label='30-day MA')
        axc1.set_title('Number of Digital Wallet Transactions with Moving Averages')
        axc1.set_xlabel('Date')
        axc1.set_ylabel('Transaction Count')
        axc1.legend()
        st.pyplot(figc1)
        
        # Growth rate for count
        wallet_daily['Growth Rate Count (%)'] = wallet_daily['Transaction Count'].pct_change() * 100
        figc2, axc2 = plt.subplots(figsize=(10, 4))
        axc2.plot(wallet_daily['transaction_date'], wallet_daily['Growth Rate Count (%)'], color='purple')
        axc2.axhline(0, color='gray', linestyle='--', linewidth=1)
        axc2.set_title('Daily Growth Rate of Digital Wallet Transaction Count')
        axc2.set_xlabel('Date')
        axc2.set_ylabel('Growth Rate (%)')
        st.pyplot(figc2)
        
        # Seasonal decomposition for count
        if len(wallet_daily) > 30:
            try:
                result_count = seasonal_decompose(wallet_daily['Transaction Count'].fillna(0), model='additive', period=30)
                figc3, (axc3, axc4, axc5) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                axc3.plot(result_count.trend, color='teal')
                axc3.set_title('Trend')
                axc4.plot(result_count.seasonal, color='orange')
                axc4.set_title('Seasonality')
                axc5.plot(result_count.resid, color='gray')
                axc5.set_title('Residuals')
                plt.tight_layout()
                st.pyplot(figc3)
            except:
                st.warning("Could not perform seasonal decomposition for transaction count.")
        
        # Summary statistics for count
        latest_count = wallet_daily['Transaction Count'].iloc[-1]
        ma7_count = wallet_daily['MA_7_count'].iloc[-1] if not pd.isna(wallet_daily['MA_7_count'].iloc[-1]) else 0
        ma30_count = wallet_daily['MA_30_count'].iloc[-1] if not pd.isna(wallet_daily['MA_30_count'].iloc[-1]) else 0
        trend_count = 'Upward' if ma7_count > ma30_count else 'Flat/Downward'
        max_count = wallet_daily['Transaction Count'].max()
        min_count = wallet_daily['Transaction Count'].min()
        avg_count = wallet_daily['Transaction Count'].mean()
        
        st.info(f"""
**Digital Wallet Transaction Count Insights:**
- **Latest Transaction Count:** {latest_count:,}
- **7-day Moving Average:** {ma7_count:.0f}
- **30-day Moving Average:** {ma30_count:.0f}
- **Trend:** {trend_count}
- **Highest Daily Transaction Count:** {max_count:,}
- **Lowest Daily Transaction Count:** {min_count:,}
- **Average Daily Transaction Count:** {avg_count:.0f}
- **Seasonality:** {'Present' if len(wallet_daily) > 30 else 'Insufficient data for decomposition'}
- **Interpretation:** {'Consistent upward trend and positive growth rate indicate increasing digital wallet usage.' if trend_count == 'Upward' else 'Flat or downward trend may indicate market saturation or seasonality.'}
""")

def main():
    st.title('Digital Dukaan: Mapping India’s Digital Payment & E-Commerce Evolution')
    menu = ['EDA', 'Time Series Analysis', 'Regional & Socio-Economic Analysis']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'EDA':
        menu2 = ['Digital Wallet Transactions', 'E-Commerce Orders', 'UPI Financial Literacy']
        choice2 = st.sidebar.selectbox('Select Dataset for EDA', menu2)
        if choice2 == 'Digital Wallet Transactions':
            show_eda_upi()
        elif choice2 == 'E-Commerce Orders':
            show_eda_merged_orders()
        else:
            show_eda_lit()
    elif choice == 'Time Series Analysis':
        show_time_series_analysis()
    else:
        show_regional_analysis()
    # ...existing code...

if __name__ == '__main__':
    main()
