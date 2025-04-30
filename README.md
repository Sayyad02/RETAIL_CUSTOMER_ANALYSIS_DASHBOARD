\# 🛒 Retail Customer Analysis Dashboard

This interactive web application, built with Streamlit, performs a comprehensive analysis of customer purchasing behavior using online retail transaction data. It allows users to upload their own data (in CSV format) and explore key insights through various analytical techniques.

\#\# ✨ Features

\* \*\*📤 Data Upload:\*\* Upload your own CSV transaction data via a simple interface.  
\* \*\*🧹 Data Cleaning:\*\* Automated preprocessing steps to handle missing values, duplicates, negative quantities/prices, and cancelled orders.  
\* \*\*📊 Exploratory Data Analysis (EDA):\*\*  
    \* Visualize monthly and daily sales trends.  
    \* Identify top customers by spending.  
    \* Discover top countries by sales volume.  
    \* Find top-selling products by quantity and revenue.  
\* \*\*📈 Purchase Behavior Analysis:\*\*  
    \* Calculate and display key metrics like Average Order Value (AOV) and Average Purchase Frequency.  
    \* Visualize the distribution of orders per customer.  
\* \*\*🧑‍🤝‍🧑 RFM Segmentation:\*\*  
    \* Segment customers based on Recency, Frequency, and Monetary value.  
    \* Visualize segment distribution and characteristics using interactive plots.  
\* \*\*🧺 Market Basket Analysis (MBA):\*\*  
    \* Identify frequently co-purchased items using the Apriori algorithm.  
    \* Display association rules (e.g., "Customers who buy X also tend to buy Y") sorted by lift and confidence.  
    \* Adjustable minimum support threshold.  
\* \*\*🧪 A/B Test Simulation (Conceptual):\*\*  
    \* Simulate the potential impact of a promotional strategy on a selected customer segment.  
    \* Perform a statistical test (t-test) on the simulated outcome.  
    \* Adjustable simulated lift percentage.  
\* \*\*⏳ Trend Monitoring:\*\*  
    \* Visualize the trend of average monthly spend for key RFM segments over time.  
\* \*\*⬇️ Download Reports:\*\*  
    \* Download the cleaned dataset as an Excel (\`.xlsx\`) file.  
    \* Download a summary text report (\`.txt\`) containing key findings and metrics from the analysis.

\#\# 🚀 How to Run

1\.  \*\*Clone the Repository (Optional):\*\*  
    \`\`\`bash  
    git clone \<your-repository-url\>  
    cd \<repository-directory\>  
    \`\`\`  
    Or simply download the Python script (\`retail\_app.py\` or your chosen name).

2\.  \*\*Create a Virtual Environment (Recommended):\*\*  
    \`\`\`bash  
    python \-m venv venv  
    source venv/bin/activate  \# On Windows use \`venv\\Scripts\\activate\`  
    \`\`\`

3\.  \*\*Install Requirements:\*\*  
    \`\`\`bash  
    pip install \-r requirements.txt  
    \`\`\`  
    \*(See \`requirements.txt\` section below if you don't have one yet)\*

4\.  \*\*Prepare Data:\*\*  
    \* Ensure you have your retail transaction data in a CSV file.  
    \* The expected columns are typically: \`InvoiceNo\`, \`StockCode\`, \`Description\`, \`Quantity\`, \`InvoiceDate\`, \`UnitPrice\`, \`CustomerID\`, \`Country\`. The cleaning process handles common issues, but the core columns should be present.

5\.  \*\*Run the Streamlit App:\*\*  
    \`\`\`bash  
    streamlit run \<your\_script\_name\>.py  
    \`\`\`  
    (Replace \`\<your\_script\_name\>.py\` with the actual name of your Python file, e.g., \`retail\_app.py\`).

6\.  \*\*Access the App:\*\* The application will automatically open in your default web browser. Upload your CSV file using the sidebar to start the analysis.

\#\# 📋 Requirements

The application requires the following Python libraries:

\* streamlit  
\* pandas  
\* numpy  
\* plotly  
\* matplotlib  
\* seaborn  
\* scipy  
\* mlxtend  
\* openpyxl

You can create a \`requirements.txt\` file with the following content:

\`\`\`text  
streamlit  
pandas  
numpy  
plotly  
matplotlib  
seaborn  
scipy  
mlxtend  
openpyxl

And install them using pip install \-r requirements.txt.

## **💾 Data Format**

The application expects a CSV file containing transaction data with columns similar to the standard "Online Retail" dataset. Key columns used in the analysis include:

* InvoiceNo: Unique identifier for each transaction.  
* Description: Name or description of the product.  
* Quantity: Number of units of the product purchased.  
* InvoiceDate: Date and time of the transaction (e.g., format MM/DD/YYYY HH:MM).  
* UnitPrice: Price per unit of the product.  
* CustomerID: Unique identifier for each customer.  
* Country: Country where the customer resides.

The data cleaning steps attempt to handle common issues, but the presence and correct format of these columns are important for the analysis to run successfully.

## **🏗️ Application Structure**

The Streamlit application is divided into the following main sections:

1. **Data Overview & Cleaning:** Initial data inspection and automated cleaning process. Option to download cleaned data.  
2. **Exploratory Data Analysis (EDA):** Visualizations of sales trends, top performers (customers, products, countries).  
3. **Purchase Behavior Analysis:** Calculation and display of AOV and purchase frequency metrics.  
4. **RFM Segmentation:** Calculation of RFM scores and assignment of customers to segments like 'Champions', 'At Risk', 'Need Attention', etc.  
5. **Market Basket Analysis (MBA):** Finding product associations using Apriori.  
6. **A/B Test Simulation:** Conceptual testing of promotional impact on a chosen segment.  
7. **Monitoring Trends:** Example visualization of segment performance over time.  
8. **Download Summary Report:** Option to download a text report summarizing the analysis.

\*(Optional: Add a License section