# 🚗 Craigslist Used Car Market Analysis

An interactive **Streamlit dashboard** for exploring used-vehicle listing prices on Craigslist. It continues the analysis in `Craigslist_Used_Car_Market_Analysis.ipynb`: the notebook cleans and explores the data, and the dashboard lets anyone filter that cleaned dataset and see every chart, statistic and insight recalculate live.

---

## Project overview

Used car prices depend on many vehicle characteristics. This project asks:

1. How does manufacturing year relate to listing price?
2. What is the relationship between mileage and price?
3. Which manufacturers have the highest average listing prices?
4. What are the most common vehicle characteristics?
5. Are there unusual values or outliers that affect the analysis?

The notebook answers these for the full dataset; the dashboard makes the same questions explorable for any subset (a manufacturer, a price band, a body type, and so on).

## Dataset

- **Source:** [Kaggle — Craigslist Cars and Trucks Data (austinreese)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- **Raw data:** 426,880 listings, 26 columns
- **Cleaned data (`clean_data.csv`):** 374,762 records, 9 features, 0 missing values

| Column | Description |
|---|---|
| `price` | Listing (asking) price in USD |
| `year` | Manufacturing (model) year |
| `manufacturer` | Vehicle make |
| `condition` | Seller-reported condition (`unknown` if not given) |
| `cylinders` | Engine cylinders (missing values filled with the mode) |
| `fuel` | Fuel type |
| `odometer` | Mileage in miles |
| `transmission` | Transmission type |
| `type` | Body type |

**Cleaning steps (from the notebook):** dropped the empty `county` column; kept prices between $1,000 and $200,000; kept mileage ≤ 500,000; removed extreme log-price values (|z| ≥ 3); dropped rows missing price, year or odometer; filled missing categories with `unknown` and missing cylinders with the mode; converted year to integer. Remaining IQR price outliers (7,821) were kept as plausible luxury/specialty listings.

## Objectives

- Clean and prepare the dataset for analysis.
- Explore numerical and categorical variables.
- Identify pricing patterns and relationships.
- Visualize findings in Python.
- Deliver an interactive dashboard for exploring the cleaned data.

## Dashboard features

**Sidebar**
- Dataset description and source link
- Filters: manufacturer, manufacturing year, price, mileage, condition, fuel type, transmission, vehicle type (an empty multiselect means "all")
- Live count of matching listings (with share of the full dataset) and a **Reset filters** button

**Main dashboard**
- **Market Overview** — KPI cards: total listings, average and median price, average mileage, average model year (each compared with the full dataset)
- **Overview tab** — data preview plus summary statistics for numeric and categorical features
- **Price Distribution** — histogram with adjustable bins, optional log scale, and average/median reference lines
- **Price Drivers** — price vs. manufacturing year and price vs. mileage scatter plots, coloured by condition or top manufacturers, with a linear trend line and the Pearson correlation for the current selection
- **Manufacturers** — top-N ranking by average price, median price or listing count, with a minimum-listings threshold and listing counts in the tooltip
- **Vehicle Characteristics** — condition, fuel, transmission and body-type distributions
- **Correlation** — heatmap of price, year and odometer with plain-language interpretation
- **Key Insights & Findings** — sentences generated from the filtered data, plus a reference panel with the full-dataset results
- **Data Explorer** — searchable, sortable table with CSV download of the filtered records

**Data integrity & performance**
- The CSV is loaded once with `@st.cache_data` and never modified; filters are vectorised boolean masks.
- KPIs, correlations, rankings and insights always use the **full** filtered data. Scatter plots show a reproducible random sample (10,000 points by default, `random_state=42`) for speed, and say so on the chart.
- Histogram bins are computed server-side, so only the bars are sent to the browser.
- Clear messages for a missing/invalid CSV, empty filter results, and correlations that cannot be computed (e.g. a single model year).

## Technology stack

Python · Streamlit · Pandas · NumPy · Plotly (Express and Graph Objects)

## Project structure

```text
Craigslist-Used-Car-Market-Analysis/
├── app.py                                    # Streamlit dashboard
├── clean_data.csv                            # Cleaned dataset exported by the notebook
├── requirements.txt
├── README.md
├── .streamlit/config.toml                    # Light theme with blue accent
└── Craigslist_Used_Car_Market_Analysis.ipynb # Cleaning, EDA and visualization
```

## Installation

Requires Python 3.9–3.13.

```bash
cd Craigslist-Used-Car-Market-Analysis
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the dashboard

```bash
streamlit run app.py
```

Streamlit opens the dashboard at `http://localhost:8501`. Make sure `clean_data.csv` is in the same folder as `app.py`.

## Main analytical findings (full cleaned dataset)

- **Manufacturing year vs price:** r = **+0.353**, a moderate positive relationship — newer vehicles tend to be listed higher. Model years 2017+ have a median price of about $29,991 vs about $6,495 for 2008 and older.
- **Mileage vs price:** r = **−0.517**, a moderate negative relationship and the stronger of the two linear relationships.
- **Manufacturers:** Ferrari has the highest average price (≈ $98,108, 64 listings), followed by Aston Martin (≈ $61,137, 21 listings) and Tesla (≈ $41,124, 809 listings). Ford is the most-listed manufacturer (62,619 listings).
- **Price distribution:** right-skewed — mean ≈ $19,522 vs median $15,995; the middle 50% of listings sit between $7,995 and $27,990.
- **Characteristics:** most listings run on gas with automatic transmissions; condition is unspecified for about 37.5% of listings.

These figures describe the full dataset. In the dashboard, all values are recomputed for whatever filters are applied.

## Dataset limitations

- Prices are **asking prices**, not final sale prices.
- Craigslist data is user-entered: some odometer readings, years and conditions are inaccurate (e.g. 0-mile used cars, very old "classic" model years).
- Many categories are `unknown` (condition ~37.5%, body type ~21%), and missing cylinders were filled with the most common value, which over-represents "6 cylinders".
- The price ($1,000–$200,000) and mileage (≤ 500,000) cut-offs are analytical choices that may exclude some legitimate listings.
- 159,494 rows share identical values across the 9 selected features; they were kept because identical specs do not necessarily mean duplicate listings (listing IDs are unique).
- Manufacturer averages based on few listings (e.g. Ferrari, Aston Martin) are sensitive to individual listings — use the minimum-listings control to compare like with like.
- Location, vehicle history and model-level detail are not part of the cleaned dataset.
