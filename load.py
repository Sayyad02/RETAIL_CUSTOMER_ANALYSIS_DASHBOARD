# Core Libraries
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st # Import Streamlit
import io # For capturing df.info() and excel export
import warnings # Import warnings module

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Statistics & RFM
from scipy import stats

# Market Basket Analysis
from mlxtend.frequent_patterns import apriori, association_rules

# Settings
warnings.filterwarnings('ignore') # Now this line should work

# --- Helper function for Excel download ---
@st.cache_data # Cache the conversion to avoid re-running
def to_excel(df):
    """Converts a DataFrame to an Excel file in memory."""
    output = io.BytesIO()
    # Use openpyxl engine explicitly if needed, pandas usually defaults well
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    processed_data = output.getvalue()
    return processed_data

# --- Page Configuration (Set Title and Layout) ---
st.set_page_config(
    page_title="Retail Customer Analysis Dashboard",
    page_icon="🛒", # Add a page icon
    layout="wide"
)

# --- App Title and Introduction ---
st.title("🛒 Retail Customer Analysis Dashboard")
st.markdown("""
Welcome! This interactive dashboard analyzes customer purchasing patterns from an online retail dataset.
Upload your data (CSV format) to explore:
* **Data Overview & Cleaning:** Understand and prepare your data.
* **Exploratory Data Analysis (EDA):** Discover sales trends, top customers, products, and countries.
* **Purchase Behavior:** Analyze key metrics like Average Order Value (AOV) and purchase frequency.
* **RFM Segmentation:** Group customers based on Recency, Frequency, and Monetary value.
* **Market Basket Analysis:** Uncover frequently co-purchased items.
* **A/B Test Simulation:** Conceptually test the potential impact of promotions.

**Upload your CSV file using the sidebar to begin!** 👇
""")
st.divider() # Add a divider after intro

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from the uploaded CSV file with caching."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

@st.cache_data
def clean_data(df_raw):
    """Cleans the raw dataframe with caching."""
    if df_raw is None:
        return None
    df = df_raw.copy()
    start_rows = df.shape[0]
    st.write(f"Cleaning data... Initial rows: {start_rows}")

    # Drop rows with missing CustomerID
    rows_before_dropna = df.shape[0]
    df.dropna(subset=['CustomerID'], inplace=True)
    rows_after_dropna = df.shape[0]
    if rows_before_dropna > rows_after_dropna:
         st.write(f"- Dropped {rows_before_dropna - rows_after_dropna} rows with missing CustomerID.")

    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Convert InvoiceDate to datetime
    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
    except ValueError:
         try:
             df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
         except Exception as e:
             st.error(f"Error parsing InvoiceDate. Please ensure format is consistent (e.g., MM/DD/YYYY HH:MM). Error: {e}")
             return None

    # Remove cancelled orders
    rows_before_cancel = df.shape[0]
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    rows_after_cancel = df.shape[0]
    if rows_before_cancel > rows_after_cancel:
        st.write(f"- Removed {rows_before_cancel - rows_after_cancel} cancelled orders (InvoiceNo starting with 'C').")


    # Remove rows with negative or zero quantity
    rows_before_qty = df.shape[0]
    df = df[df['Quantity'] > 0]
    rows_after_qty = df.shape[0]
    if rows_before_qty > rows_after_qty:
        st.write(f"- Removed {rows_before_qty - rows_after_qty} rows with Quantity <= 0.")

    # Remove rows with zero or negative unit price
    rows_before_price = df.shape[0]
    df = df[df['UnitPrice'] > 0]
    rows_after_price = df.shape[0]
    if rows_before_price > rows_after_price:
         st.write(f"- Removed {rows_before_price - rows_after_price} rows with UnitPrice <= 0.")

    # Create TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Drop duplicates
    rows_before_dupes = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after_dupes = df.shape[0]
    if rows_before_dupes > rows_after_dupes:
        st.write(f"- Removed {rows_before_dupes - rows_after_dupes} duplicate rows.")

    st.write(f"Cleaning finished. Final rows: {df.shape[0]}")
    return df

@st.cache_data
def calculate_rfm(df):
    """Calculates RFM metrics from the cleaned dataframe."""
    if df is None or 'InvoiceDate' not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
         st.error("InvoiceDate column is not in datetime format for RFM calculation.")
         return None, None
    snapshot_dt = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_data = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_dt - date.max()).days, # Recency
        'InvoiceNo': 'nunique',                                       # Frequency
        'TotalPrice': 'sum'                                           # Monetary
    }).reset_index()
    rfm_data.rename(columns={'InvoiceDate': 'Recency',
                           'InvoiceNo': 'Frequency',
                           'TotalPrice': 'MonetaryValue'}, inplace=True)
    return rfm_data, snapshot_dt

@st.cache_data
def perform_mba(_df_cleaned, min_support=0.02): # Use different variable name to avoid confusion with cached df
    """Performs Market Basket Analysis using Apriori."""
    # Add spinner for long operation
    with st.spinner("Performing Market Basket Analysis... This may take a moment."):
        if _df_cleaned is None or 'InvoiceNo' not in _df_cleaned.columns or 'Description' not in _df_cleaned.columns:
            st.warning("MBA requires 'InvoiceNo' and 'Description' columns.")
            return None, None

        # Prepare data: Create transaction-item matrix (basket)
        try:
            # Clean Description column slightly
            _df_cleaned['Description'] = _df_cleaned['Description'].str.strip()
            basket = (_df_cleaned
                      .groupby(['InvoiceNo', 'Description'])['Quantity']
                      .sum().unstack().reset_index().fillna(0)
                      .set_index('InvoiceNo'))
        except Exception as e:
            st.error(f"Error creating basket for MBA: {e}")
            return None, None

        # Convert quantities to 0 or 1
        def encode_units(x):
            return 1 if x >= 1 else 0
        basket_sets = basket.map(encode_units)

        if 'POSTAGE' in basket_sets.columns:
            basket_sets.drop('POSTAGE', inplace=True, axis=1)
        if basket_sets.empty:
            st.warning("Basket data is empty after processing. Cannot perform MBA.")
            return None, None

        # Apply Apriori
        try:
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        except Exception as e:
            st.error(f"Error during Apriori calculation: {e}")
            return None, None
        if frequent_itemsets.empty:
            st.warning(f"No frequent itemsets found with min_support={min_support}. Try lowering the threshold.")
            return None, None

        # Generate association rules
        try:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        except Exception as e:
            st.error(f"Error generating association rules: {e}")
            return frequent_itemsets, None

        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    st.success("Market Basket Analysis completed.")
    return frequent_itemsets, rules


# --- File Uploader ---
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", help="Upload your online retail transaction data.")

# --- Main App Logic ---
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        st.header("1. Data Overview & Cleaning 🧹")
        st.markdown("Inspecting the raw data and applying cleaning steps.")
        st.write("Shape of raw data:", df_raw.shape)
        st.dataframe(df_raw.head())

        with st.expander("Raw Data Details"):
            st.subheader("Initial Data Info")
            buffer = io.StringIO()
            df_raw.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.subheader("Missing Values")
            st.dataframe(df_raw.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))
            st.subheader("Descriptive Stats")
            st.dataframe(df_raw.describe())

        # --- Data Cleaning ---
        df_cleaned = clean_data(df_raw)

        if df_cleaned is not None:
            if not pd.api.types.is_datetime64_any_dtype(df_cleaned['InvoiceDate']):
                 st.error("Data cleaning failed to produce a datetime 'InvoiceDate' column. Cannot proceed.")
                 st.stop()

            st.success("Data cleaning completed successfully!")
            st.write("Shape of cleaned data:", df_cleaned.shape)

            with st.expander("View Cleaned Data Details"):
                 st.subheader("Cleaned Data Head")
                 st.dataframe(df_cleaned.head())
                 st.subheader("Missing Values After Cleaning")
                 st.dataframe(df_cleaned.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))

            # --- Add Excel Download Button ---
            st.subheader("⬇️ Download Cleaned Data")
            st.markdown("Download the processed data used for the analysis below.")
            excel_data = to_excel(df_cleaned)
            st.download_button(
                label="Download Cleaned Data (.xlsx)",
                data=excel_data,
                file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.divider() # Divider after cleaning section

            # --- Initialize Report String ---
            report_content = "# Online Retail Analysis Report\n\n"
            report_content += f"Analysis performed on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_content += f"Uploaded Filename: {uploaded_file.name}\n\n"
            report_content += f"## 1. Data Summary\n"
            report_content += f"- Raw data shape: {df_raw.shape}\n"
            report_content += f"- Cleaned data shape: {df_cleaned.shape}\n\n"

            # --- Exploratory Data Analysis (EDA) ---
            st.header("2. Exploratory Data Analysis (EDA) 📊")
            st.markdown("Visualizing key trends and patterns in the cleaned data.")
            report_content += "## 2. Exploratory Data Analysis\n"

            st.subheader("Sales Trends")
            col1, col2 = st.columns(2)
            # ... (Rest of EDA, RFM, MBA, A/B Test, Monitoring, Report sections remain largely the same) ...
            # --- Sales Trends Over Time ---
            with col1:
                st.markdown("##### Total Monthly Sales")
                try:
                    monthly_sales = df_cleaned.resample('M', on='InvoiceDate')['TotalPrice'].sum().reset_index()
                    monthly_sales.rename(columns={'InvoiceDate': 'InvoiceMonth'}, inplace=True)
                    fig_monthly = px.line(monthly_sales, x='InvoiceMonth', y='TotalPrice', markers=True,
                                          labels={'TotalPrice': 'Total Sales ($)', 'InvoiceMonth': 'Month'})
                    fig_monthly.update_layout(title='Total Monthly Sales Over Time', title_x=0.1)
                    st.plotly_chart(fig_monthly, use_container_width=True)
                    if not monthly_sales.empty:
                         report_content += f"- Peak sales month observed around: {monthly_sales.loc[monthly_sales['TotalPrice'].idxmax()]['InvoiceMonth'].strftime('%Y-%m')}\n"
                    else: report_content += "- Monthly sales data could not be generated.\n"
                except Exception as e:
                    st.error(f"Error generating monthly sales trend: {e}")
                    report_content += "- Error generating monthly sales trend.\n"

            with col2:
                st.markdown("##### Total Sales by Day of Week")
                try:
                    df_cleaned['DayOfWeek'] = df_cleaned['InvoiceDate'].dt.day_name()
                    daily_sales = df_cleaned.groupby('DayOfWeek')['TotalPrice'].sum().reset_index()
                    daily_sales = daily_sales.set_index('DayOfWeek').reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        ).reset_index()
                    fig_daily = px.bar(daily_sales, x='DayOfWeek', y='TotalPrice',
                                       labels={'TotalPrice': 'Total Sales ($)', 'DayOfWeek': 'Day of Week'})
                    fig_daily.update_layout(title='Total Sales by Day of the Week', title_x=0.1)
                    st.plotly_chart(fig_daily, use_container_width=True)
                    if not daily_sales.empty:
                        report_content += f"- Busiest day for sales: {daily_sales.loc[daily_sales['TotalPrice'].idxmax()]['DayOfWeek']}\n"
                    else: report_content += "- Daily sales data could not be generated.\n"
                except Exception as e:
                    st.error(f"Error generating daily sales trend: {e}")
                    report_content += "- Error generating daily sales trend.\n"

            # --- Top Customers & Countries ---
            st.subheader("Top Customers & Countries")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### Top 10 Customers by Total Spend")
                top_customers = df_cleaned.groupby('CustomerID')['TotalPrice'].sum().nlargest(10).reset_index()
                fig_cust = px.bar(top_customers, x='CustomerID', y='TotalPrice', text_auto='.2s',
                                  labels={'TotalPrice': 'Total Sales ($)'},
                                  category_orders={'CustomerID': top_customers['CustomerID'].astype(str).tolist()})
                fig_cust.update_layout(title='Top 10 Customers by Total Spend', title_x=0.1, xaxis_title="Customer ID", yaxis_title="Total Sales ($)")
                st.plotly_chart(fig_cust, use_container_width=True)
                if not top_customers.empty:
                    report_content += f"- Top spending customer ID: {top_customers.iloc[0]['CustomerID']} (${top_customers.iloc[0]['TotalPrice']:.2f})\n"

            with col4:
                st.markdown("##### Top 15 Countries by Total Sales")
                country_sales = df_cleaned.groupby('Country')['TotalPrice'].sum().nlargest(15).reset_index()
                fig_country = px.bar(country_sales, x='Country', y='TotalPrice', text_auto='.2s',
                                     labels={'TotalPrice': 'Total Sales ($)'})
                fig_country.update_layout(title='Top 15 Countries by Total Sales', title_x=0.1, yaxis_title="Total Sales ($)")
                st.plotly_chart(fig_country, use_container_width=True)
                if not country_sales.empty:
                    report_content += f"- Top country by sales: {country_sales.iloc[0]['Country']} (${country_sales.iloc[0]['TotalPrice']:.2f})\n"

            # --- Top Selling Products ---
            st.subheader("Top Selling Products")
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("##### Top 10 Products by Quantity Sold")
                if 'Description' in df_cleaned.columns:
                    top_products_quantity = df_cleaned.dropna(subset=['Description']).groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
                    if not top_products_quantity.empty:
                        fig_prod_q = px.bar(top_products_quantity, x='Description', y='Quantity')
                        fig_prod_q.update_layout(title='Top 10 Products by Quantity Sold', title_x=0.1, yaxis_title="Total Quantity")
                        st.plotly_chart(fig_prod_q, use_container_width=True)
                        report_content += f"- Top product by quantity: {top_products_quantity.iloc[0]['Description']}\n"
                    else:
                        st.warning("Could not determine top products by quantity.")
                        report_content += "- Could not determine top products by quantity.\n"
                else:
                    st.warning("'Description' column not found for product analysis.")
                    report_content += "- 'Description' column not found for product analysis.\n"
            with col6:
                st.markdown("##### Top 10 Products by Total Revenue")
                if 'Description' in df_cleaned.columns:
                    top_products_revenue = df_cleaned.dropna(subset=['Description']).groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
                    if not top_products_revenue.empty:
                        fig_prod_r = px.bar(top_products_revenue, x='Description', y='TotalPrice', text_auto='.2s',
                                            labels={'TotalPrice': 'Total Sales ($)'})
                        fig_prod_r.update_layout(title='Top 10 Products by Total Revenue', title_x=0.1, yaxis_title="Total Sales ($)")
                        st.plotly_chart(fig_prod_r, use_container_width=True)
                        report_content += f"- Top product by revenue: {top_products_revenue.iloc[0]['Description']}\n\n"
                    else:
                        st.warning("Could not determine top products by revenue.")
                        report_content += "- Could not determine top products by revenue.\n\n"
                else:
                    report_content += "- 'Description' column not found for product analysis.\n\n"

            st.divider() # Divider after EDA

            # --- Purchase Behavior Analysis ---
            st.header("3. Purchase Behavior Analysis 📈")
            st.markdown("Analyzing key metrics related to customer purchase patterns.")
            report_content += "## 3. Purchase Behavior Analysis\n"

            col7, col8, col9 = st.columns(3)
            total_customers = df_cleaned['CustomerID'].nunique()
            total_invoices = df_cleaned['InvoiceNo'].nunique()
            invoice_totals = df_cleaned.groupby('InvoiceNo')['TotalPrice'].sum()
            aov = invoice_totals.mean() if not invoice_totals.empty else 0
            orders_per_customer = df_cleaned.groupby('CustomerID')['InvoiceNo'].nunique()
            avg_purchase_frequency = orders_per_customer.mean() if not orders_per_customer.empty else 0

            col7.metric("Total Unique Customers 👥", f"{total_customers:,}")
            col8.metric("Total Unique Invoices 🧾", f"{total_invoices:,}")
            col9.metric("Average Order Value (AOV) 💰", f"${aov:.2f}")
            st.metric("Average Purchase Frequency (Orders/Customer) 🔄", f"{avg_purchase_frequency:.2f}")

            report_content += f"- Total Unique Customers: {total_customers:,}\n"
            report_content += f"- Total Unique Invoices: {total_invoices:,}\n"
            report_content += f"- Average Order Value (AOV): ${aov:.2f}\n"
            report_content += f"- Average Purchase Frequency: {avg_purchase_frequency:.2f} orders/customer\n\n"

            if not orders_per_customer.empty:
                st.subheader("Distribution of Orders per Customer")
                fig_freq_dist, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(orders_per_customer, bins=50, kde=False, ax=ax)
                ax.set_title('Distribution of Number of Orders per Customer (Log Scale)')
                ax.set_xlabel('Number of Orders')
                ax.set_ylabel('Number of Customers')
                ax.set_yscale('log')
                xlim_upper = orders_per_customer.quantile(0.95) if not orders_per_customer.empty else 10
                ax.set_xlim(0, xlim_upper)
                st.pyplot(fig_freq_dist)
            else:
                st.warning("Could not generate distribution of orders per customer.")

            st.divider() # Divider after Purchase Behavior

            # --- User Segmentation (RFM Analysis) ---
            st.header("4. User Segmentation (RFM Analysis) 🧑‍🤝‍🧑")
            st.markdown("Segmenting customers based on **R**ecency, **F**requency, and **M**onetary value.")
            report_content += "## 4. RFM Customer Segmentation\n"

            rfm_data, snapshot_date = calculate_rfm(df_cleaned)

            if rfm_data is not None:
                st.success(f"RFM analysis based on data up to: **{snapshot_date.date()}**")
                report_content += f"- RFM Snapshot Date: {snapshot_date.date()}\n"

                with st.expander("View RFM Metrics Data & Stats"):
                    st.subheader("RFM Metrics (Sample)")
                    st.dataframe(rfm_data.head())
                    st.subheader("RFM Descriptive Stats")
                    st.dataframe(rfm_data[['Recency', 'Frequency', 'MonetaryValue']].describe())

                # --- RFM Scores ---
                st.subheader("Assigning RFM Scores and Segments")
                r_labels = range(4, 0, -1); f_labels = range(1, 5); m_labels = range(1, 5)
                try:
                    rfm_data['Recency'] = pd.to_numeric(rfm_data['Recency'])
                    rfm_data['Frequency'] = pd.to_numeric(rfm_data['Frequency'])
                    rfm_data['MonetaryValue'] = pd.to_numeric(rfm_data['MonetaryValue'])
                    rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], q=4, labels=r_labels, duplicates='drop')
                    rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), q=4, labels=f_labels, duplicates='drop')
                    rfm_data['M_Score'] = pd.qcut(rfm_data['MonetaryValue'], q=4, labels=m_labels, duplicates='drop')
                except Exception as e:
                    st.error(f"Error assigning RFM scores using qcut: {e}. Check data distribution and types.")
                    st.stop()

                rfm_data['RFM_Score_Str'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)
                simplified_segment_map = {
                    r'^[1-2][1-2]': 'Hibernating', r'^[1-2][3-4]': 'At Risk',
                    r'^3[1-2]': 'Need Attention', r'^3[3-4]': 'Potential Loyalists',
                    r'^4[1-2]': 'New/Promising', r'^4[3-4]': 'Champions/Loyal'
                }
                rfm_data['Segment'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str)
                rfm_data['Segment'] = rfm_data['Segment'].replace(simplified_segment_map, regex=True)
                rfm_data['Segment'] = rfm_data['Segment'].astype('category')
                unsegmented_mask = ~rfm_data['Segment'].isin(simplified_segment_map.values())
                if unsegmented_mask.any(): rfm_data.loc[unsegmented_mask, 'Segment'] = 'Other'

                st.success("RFM Scores and Segments Assigned.")
                st.dataframe(rfm_data[['CustomerID', 'Recency', 'Frequency', 'MonetaryValue', 'Segment']].head())

                # --- Visualize Segments ---
                st.subheader("RFM Segment Visualization")
                col10, col11 = st.columns(2)
                with col10:
                     st.markdown("##### Customer Count by RFM Segment")
                     segment_counts = rfm_data['Segment'].value_counts().reset_index()
                     segment_counts.columns = ['Segment', 'Count']
                     fig_segment = px.bar(segment_counts, x='Segment', y='Count', text_auto=True,
                                          color='Segment', # Color bars by segment
                                          labels={'Count': 'Number of Customers'})
                     fig_segment.update_layout(title='Customer Count by RFM Segment', title_x=0.1)
                     st.plotly_chart(fig_segment, use_container_width=True)
                     report_content += "\n### RFM Segment Counts:\n"
                     report_content += segment_counts.to_markdown(index=False) + "\n\n"
                with col11:
                    st.markdown("##### Recency vs Frequency")
                    fig_rf_scatter = px.scatter(rfm_data, x='Recency', y='Frequency', color='Segment',
                                                hover_data=['MonetaryValue', 'CustomerID'],
                                                size='MonetaryValue', # Size bubbles by Monetary value
                                                size_max=30) # Control max bubble size
                    fig_rf_scatter.update_layout(title='Recency vs Frequency by Segment', title_x=0.1)
                    st.plotly_chart(fig_rf_scatter, use_container_width=True)

                st.markdown("##### Recency vs Monetary Value")
                fig_rm_scatter = px.scatter(rfm_data, x='Recency', y='MonetaryValue', color='Segment',
                                            log_y=True, hover_data=['Frequency', 'CustomerID'],
                                            size='Frequency', # Size bubbles by Frequency
                                            size_max=30)
                fig_rm_scatter.update_layout(title='Recency vs Monetary Value by Segment (Log Scale)', title_x=0.1,
                                             yaxis_title="Monetary Value (Log Scale)")
                st.plotly_chart(fig_rm_scatter, use_container_width=True)
            else:
                st.warning("Could not calculate RFM data.")
                report_content += "- RFM analysis could not be performed.\n"

            st.divider() # Divider after RFM

            # --- Market Basket Analysis (MBA) ---
            st.header("5. Market Basket Analysis (MBA) 🧺")
            st.markdown("Identifying frequently co-purchased items using the Apriori algorithm. This helps understand product associations.")
            report_content += "## 5. Market Basket Analysis\n"

            min_support_threshold = st.slider("Select Minimum Support for MBA:", min_value=0.01, max_value=0.1, value=0.02, step=0.005, format="%.3f",
                                              help="Lower support finds more (potentially less relevant) rules; higher finds fewer (potentially stronger) rules.")

            frequent_itemsets, rules = perform_mba(df_cleaned, min_support=min_support_threshold) # Pass cleaned df

            if frequent_itemsets is not None:
                with st.expander("View Frequent Itemsets (Sample)"):
                    st.dataframe(frequent_itemsets.head())

                if rules is not None and not rules.empty:
                    st.subheader("Association Rules (Top 20 by Lift)")
                    st.markdown("`Lift > 1` suggests items are bought together more often than expected by chance.")
                    rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.dataframe(rules_display.head(20).style.format({'support': '{:.3f}', 'confidence': '{:.3f}', 'lift': '{:.2f}'})) # Format numbers

                    report_content += f"\n### Top 5 Association Rules (min_support={min_support_threshold}):\n"
                    report_content += rules_display.head(5).to_markdown(index=False) + "\n\n"
                elif rules is not None and rules.empty:
                     st.warning(f"No association rules found with lift >= 1 for min_support={min_support_threshold}. Try adjusting thresholds.")
                     report_content += f"- No association rules found (lift >= 1, min_support={min_support_threshold}).\n\n"
                else:
                    report_content += "- Could not generate association rules.\n\n"
            else:
                 report_content += "- Could not generate frequent itemsets.\n\n"

            st.divider() # Divider after MBA

            # --- A/B Test Simulation ---
            st.header("6. A/B Test Simulation (Conceptual) 🧪")
            report_content += "## 6. A/B Test Simulation\n"
            st.markdown("""
            Simulating an A/B test targeting a specific customer segment (e.g., 'Need Attention') to estimate the potential impact of a promotion (e.g., a discount increasing average spend).
            **Disclaimer:** This is a *simulation* using existing data and assumptions. Real A/B tests require implementing the change and collecting new data.
            """)

            if rfm_data is not None:
                # Allow segment selection for simulation
                segment_list = rfm_data['Segment'].unique().tolist()
                target_segment = st.selectbox("Select Target Segment for A/B Test Simulation:", segment_list, index=segment_list.index('Need Attention') if 'Need Attention' in segment_list else 0)

                # Check if selected segment exists
                if target_segment in rfm_data['Segment'].cat.categories:
                    segment_customers = rfm_data[rfm_data['Segment'] == target_segment].copy()
                    st.info(f"Simulating test on **{target_segment}** segment ({segment_customers.shape[0]} customers)")
                    report_content += f"- A/B Test Target Segment: {target_segment} ({segment_customers.shape[0]} customers)\n"

                    if segment_customers.shape[0] > 10:
                        group_a = segment_customers.sample(frac=0.5, random_state=42)
                        group_b_indices = segment_customers.index.difference(group_a.index)
                        group_b = segment_customers.loc[group_b_indices]

                        st.write(f"Group A (Control): {group_a.shape[0]} customers")
                        st.write(f"Group B (Variant - Simulated Treatment): {group_b.shape[0]} customers")

                        simulated_lift_pct = st.slider("Simulated Lift (%) in Avg. Spend for Group B:", min_value=5, max_value=50, value=15, step=5,
                                                       help="Assume the promotion increases the average spend of customers in Group B by this percentage.")
                        simulated_lift = simulated_lift_pct / 100.0
                        group_b_simulated_monetary = group_b['MonetaryValue'] * (1 + simulated_lift)

                        avg_monetary_a = group_a['MonetaryValue'].mean()
                        avg_monetary_b_simulated_mean = group_b_simulated_monetary.mean()

                        col12, col13 = st.columns(2)
                        col12.metric("Avg Monetary Value (Group A - Control)", f"${avg_monetary_a:.2f}")
                        col13.metric(f"Avg Monetary Value (Group B - Simulated +{simulated_lift:.0%})", f"${avg_monetary_b_simulated_mean:.2f}")
                        report_content += f"- Simulated Lift Applied to Group B: +{simulated_lift:.0%}\n"
                        report_content += f"- Group A Avg Monetary: ${avg_monetary_a:.2f}\n"
                        report_content += f"- Group B Simulated Avg Monetary: ${avg_monetary_b_simulated_mean:.2f}\n"

                        st.subheader("Statistical Test (Simulated Outcome)")
                        st.markdown("Comparing Group A's actual spending with Group B's *simulated* spending after applying the lift using Welch's t-test.")
                        if group_a['MonetaryValue'].isnull().any() or group_b_simulated_monetary.isnull().any() or group_a.empty or group_b.empty:
                             st.warning("Cannot perform t-test due to missing or empty data in groups.")
                             report_content += "- T-test could not be performed due to data issues.\n\n"
                        else:
                            t_stat, p_value = stats.ttest_ind(group_a['MonetaryValue'], group_b_simulated_monetary, equal_var=False, nan_policy='omit')
                            st.write(f"T-statistic: {t_stat:.3f}")
                            st.write(f"P-value: {p_value:.6f}")
                            report_content += f"- T-statistic (A vs B Simulated): {t_stat:.3f}\n"
                            report_content += f"- P-value (A vs B Simulated): {p_value:.6f}\n"
                            alpha = 0.05
                            if p_value < alpha:
                                st.success(f"**Result:** The simulated difference is statistically significant (p < {alpha}). The simulated treatment had a detectable effect.")
                                report_content += f"- Result: Simulated difference is statistically significant (p < {alpha}).\n\n"
                            else:
                                st.warning(f"**Result:** The simulated difference is NOT statistically significant (p >= {alpha}). The simulated treatment did not have a large enough effect to be detected with this sample size.")
                                report_content += f"- Result: Simulated difference is NOT statistically significant (p >= {alpha}).\n\n"
                    else:
                        st.warning(f"Not enough customers in '{target_segment}' to perform a meaningful simulation (need > 10).")
                        report_content += f"- A/B Test: Not enough customers in '{target_segment}' for simulation.\n\n"
                else:
                     st.warning(f"Selected target segment '{target_segment}' not found in data. Skipping A/B Test Simulation.")
                     report_content += "- A/B Test: Target segment not found.\n\n"
            else:
                 st.warning("RFM data not available, skipping A/B Test Simulation.")
                 report_content += "- A/B Test: RFM data not available.\n\n"

            st.divider() # Divider after A/B Test

            # --- Monitoring Trends ---
            st.header("7. Monitoring Trends ⏳")
            report_content += "## 7. Monitoring Trends\n"
            st.markdown("""
            Continuously tracking KPIs and segment shifts over time is crucial for understanding business health and campaign effectiveness.
            Below is an example showing the trend of average monetary value for key RFM segments.
            """)

            if rfm_data is not None:
                try:
                    if 'Segment' in rfm_data.columns:
                        df_merged = pd.merge(df_cleaned, rfm_data[['CustomerID', 'Segment']], on='CustomerID', how='left')
                        if 'InvoiceDate' in df_merged.columns and pd.api.types.is_datetime64_any_dtype(df_merged['InvoiceDate']) and 'Segment' in df_merged.columns:
                            # Resample to get monthly average spend per segment
                            segment_trends = df_merged.set_index('InvoiceDate').groupby('Segment')['TotalPrice'].resample('M').mean().reset_index()
                            segment_trends.rename(columns={'InvoiceDate': 'InvoiceMonth'}, inplace=True) # Rename for clarity

                            key_segments = ['Champions/Loyal', 'Potential Loyalists', 'At Risk', 'Need Attention']
                            segment_trends_filtered = segment_trends[segment_trends['Segment'].isin(key_segments)]

                            if not segment_trends_filtered.empty:
                                fig_segment_trend = px.line(segment_trends_filtered, x='InvoiceMonth', y='TotalPrice', color='Segment',
                                                            title='Average Monthly Spend by Key RFM Segment',
                                                            labels={'TotalPrice': 'Average Sales ($)', 'InvoiceMonth': 'Month'})
                                fig_segment_trend.update_layout(yaxis_title="Average Sales ($)")
                                st.plotly_chart(fig_segment_trend, use_container_width=True)
                                report_content += "- Trend plot generated for average monthly spend by key segments.\n"
                            else:
                                st.warning("Could not generate segment trend data (possibly due to lack of data points for key segments).")
                                report_content += "- Could not generate segment trend plot.\n"
                        else:
                            st.warning("Required columns ('InvoiceDate', 'Segment') not available or in correct format for trend analysis.")
                            report_content += "- Required columns not available for trend analysis.\n"
                    else:
                        st.warning("'Segment' column not found in RFM data for merging.")
                        report_content += "- 'Segment' column not found in RFM data.\n"

                    st.info("""
                    **Monitoring Best Practices:**
                    * **Regular Reporting:** Automate this analysis to run periodically (e.g., weekly/monthly).
                    * **Dashboarding:** Utilize BI tools (like Tableau, Power BI, Looker) for real-time KPI tracking.
                    * **Alerting:** Implement alerts for significant metric deviations or segment shifts.
                    """)
                    report_content += "- Recommended monitoring: Regular reporting, dashboarding, alerting.\n\n"
                except Exception as e:
                    st.error(f"Error generating monitoring trends: {e}")
                    report_content += f"- Error generating monitoring trends: {e}\n\n"
            else:
                st.warning("RFM data not available, cannot show segment trends.")
                report_content += "- RFM data not available for trend monitoring.\n\n"

            st.divider() # Divider before final report

            # --- Download Report ---
            st.header("8. Download Summary Report 📝")
            st.markdown("Click the button below to download a text file summarizing the key findings, metrics, and recommendations from this analysis.")

            st.download_button(
                label="Download Analysis Report (.txt)",
                data=report_content,
                file_name=f"retail_analysis_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        else: # df_cleaned is None
            st.error("Data cleaning failed. Cannot proceed with analysis.")
    else: # df_raw is None
        # Error handled in load_data
        pass

else: # uploaded_file is None
    st.info("👋 Welcome! Please upload a CSV file using the sidebar to start the analysis.")

st.sidebar.divider()
st.sidebar.info(" Streamlit App for Online Retail Analysis")
st.sidebar.markdown("Developed for Customer Behavior Analysis.")
