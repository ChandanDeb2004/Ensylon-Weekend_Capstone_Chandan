## Introduction

This week's Capstone project contains ETL processing of two different datasets:

1. Airline Flights ETL Pipeline  
2. Stock Market ETL Pipeline  

---


# 1. Airline Flights ETL Pipeline

**Author:** Chandan Deb

------------------------------------------------------------------------

## Project Overview

This project implements an **end-to-end ETL (Extract, Transform, Load)
pipeline** for airline flight operations data using Python, Pandas, and
SQLite.

Main stages:

1.  Data Extraction
2.  Data Validation
3.  Data Cleaning
4.  Data Transformation
5.  Aggregation & Analysis
6.  SQLite Data Loading
7.  Relational Schema Design

------------------------------------------------------------------------

## Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   SQLite
-   Jupyter Notebook

------------------------------------------------------------------------

## Project Structure

    Airline-ETL-Pipeline
    │
    ├── ETL_Project_Chandan_Airline.ipynb
    ├── data/
    │   ├── airline_flights.csv
    │   └── cleaned/
    │       └── airline_flights_cleaned.csv
    ├── database/
    │   └── airline_flights.db
    └── README.txt

------------------------------------------------------------------------

## Dataset Columns

| Column | Description |
|------|-------------|
| Date | Flight date |
| Departure Time | Scheduled departure |
| Flight Number | Unique identifier |
| Airline | Airline company |
| Aircraft Type | Aircraft model |
| Origin Airport | Departure airport |
| Destination Airport | Arrival airport |
| Distance (km) | Flight distance |
| Passengers (First Class) | First class passengers |
| Passengers (Business Class) | Business class passengers |
| Passengers (Economy Class) | Economy passengers |
| Revenue ($) | Revenue generated |
| Arrival Gate | Arrival gate |
| Departure Gate | Departure gate |
------------------------------------------------------------------------

## Example ETL Code

``` python
import pandas as pd

df = pd.read_csv("airline_flights.csv")
df["Date"] = pd.to_datetime(df["Date"])
```

------------------------------------------------------------------------

## Example Aggregation

``` python
df.groupby("Airline")["Revenue ($)"].sum()
```

------------------------------------------------------------------------

## Example SQL

``` sql
SELECT airline, SUM(revenue)
FROM flights
GROUP BY airline;
```

------------------------------------------------------------------------

## Future Improvements

-   Airflow pipeline orchestration
-   Incremental ETL
-   Data warehouse architecture
-   SQL index optimization
-   BI dashboards

------------------------------------------------------------------------

# Big Tech Stock Data ETL Pipeline

## Goal
To perform end-to-end data preprocessing, transformation, and loading for an ETL project focused on historical big tech stock prices.

------------------------------------------------------------------------
## Dataset
* **Source**: [Big Tech Stock Prices (GitHub)](https://github.com/cgre23/etl-dataset/blob/main/big_tech_stock_prices.txt)
* **Contents**: Historical data including `open`, `high`, `low`, `close`, `adj_close`, and `volume` for 14 major tech companies.
-----------------------------------------------------------------------

## Tech Stack
* **Language**: Python
* **Libraries**: 
    * `pandas`, `numpy`: Data manipulation and analysis.
    * `matplotlib`, `seaborn`: Data quality visualization.
    * `sqlite3`: Relational database storage.
    * `logging`: Pipeline monitoring and error tracking.
----------------------------------------------------------------------

## Pipeline Architecture

### 1. Extraction & Data Quality
* **Data Extraction**: Reusable functions with error handling to load raw text/CSV files.
* **Data Profiling**: Generates summaries of shapes, data types, and memory usage.
* **Quality Assessment**: Visualizes null percentages and data completeness.
* **Validation**: Checks for missing columns, row count thresholds, and high null ratios.

### 2. Transformation & Feature Engineering
* **Cleaning**: Handles missing values using time-series forward filling (`ffill`).
* **Feature Creation**:
    * **Daily Returns %**: `(close - open) / open`
    * **Daily Price Range**: `high - low`
    * **Quarterly/Yearly Labels**: Derived from date-time objects.
* **Advanced Validation**:
    * Identifies illegal weekend trades.
    * Flags extreme price moves (>40% daily return).
    * Checks for negative prices or duplicate records.

### 3. Loading (Storage)
* **File Storage**: Exports the fully cleaned dataset and summarized reports (Quarterly/Yearly) to an `output/` directory in CSV format.
* **Database Storage**:
    * Loads data into an SQLite database (`stock_market_etl.db`).
    * Creates a `stock_prices_fact` table and summary tables.
    * Implements **Database Indexing** on `stock_symbol` and `date` for high-performance querying.

### 4. Logging and Error Handling
* Includes a logging framework to track the ETL process.
* Implements `try-except` blocks to ensure the pipeline doesn't crash on missing files or malformed data.

---------------------------------------------------------------

## Usage
1.  Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```
2.  Place the raw data file in the expected directory (default: `/content/Data/`).
3.  Run the notebook cells sequentially to execute the full ETL pipeline.
4.  Check the `output/` folder for CSV results and the local directory for the `.db` file.

---------------------------------------------------------------------
## Author
**Chandan**
