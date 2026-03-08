#Introduction
This weeks Capstone project contains ETL processing of two different datasets 
1.Airline Flights ETL Pipeline
2.Stock Market ETL Pipeline

Airline Flights ETL Pipeline

Author: Chandan Deb

  ------------------
  PROJECT OVERVIEW
  ------------------

This project implements an end‑to‑end ETL (Extract, Transform, Load)
data engineering pipeline for airline flight operations data.

The pipeline performs the following steps:

1.  Data Extraction from CSV files
2.  Data Validation and Profiling
3.  Data Cleaning and Missing Value Handling
4.  Data Type Standardization
5.  Exploratory Aggregations
6.  Loading Clean Data into SQLite Database
7.  Database Normalization for relational analysis

The goal of the project is to simulate a real-world data engineering
workflow that prepares raw operational data for analytics and querying.

  -------------------
  TECHNOLOGIES USED
  -------------------

Python Pandas NumPy Matplotlib Seaborn SQLite Jupyter Notebook

  -------------------
  PROJECT STRUCTURE
  -------------------

Airline-ETL-Pipeline │ ├── ETL_Project_Chandan_Airline.ipynb │ ├── data
│ ├── airline_flights.csv │ └── cleaned │ └──
airline_flights_cleaned.csv │ ├── database │ └── airline_flights.db │
└── README.txt

  ---------------------
  DATASET DESCRIPTION
  ---------------------

The dataset contains airline operational flight records.

Important columns include:

Date Departure Time Flight Number Airline Aircraft Type Origin Airport
Destination Airport Distance (km) Passengers (First Class) Passengers
(Business Class) Passengers (Economy Class) Revenue ($) Arrival Gate
Departure Gate

  --------------
  ETL PIPELINE
  --------------

1.  DATA EXTRACTION

The dataset is loaded using pandas.read_csv(). Basic checks are
performed to ensure the file exists and the dataset loads correctly.

2.  DATA VALIDATION

Validation checks include:

-   Expected column verification
-   Minimum row count checks
-   Missing column detection
-   Data structure validation

3.  DATA PROFILING

The pipeline inspects the dataset using:

-   Dataset shape
-   Column datatypes
-   Memory usage
-   Statistical summary
-   Null value analysis

4.  DATA CLEANING

Missing values are handled using domain‑specific strategies:

-   Mode imputation for categorical columns
-   Removal of rows with excessive missing values
-   Replacement of missing gate values

5.  DATA TYPE STANDARDIZATION

Columns are converted to appropriate types:

Date -> datetime Passenger counts -> numeric Revenue -> numeric

6.  DATA AGGREGATION

Basic analytical insights are generated such as:

-   Total revenue by airline
-   Most popular destinations
-   Most frequently used aircraft types

7.  DATA EXPORT

The cleaned dataset is saved to:

data/cleaned/airline_flights_cleaned.csv

8.  DATABASE LOADING

The cleaned dataset is loaded into an SQLite database:

airline_flights.db

9.  DATABASE NORMALIZATION

The dataset can be further structured into relational tables such as:

Airline Aircraft Airports Flights Flight_Instance Passengers Revenue

This supports more efficient querying and relational analysis.

  ---------------------
  EXAMPLE SQL QUERIES
  ---------------------

Total revenue by airline:

SELECT airline, SUM(revenue) FROM flights GROUP BY airline;

Most popular routes:

SELECT origin, destination, COUNT(*) FROM flights GROUP BY origin,
destination;

  ---------------------
  FUTURE IMPROVEMENTS
  ---------------------

Possible improvements include:

-   Airflow pipeline orchestration
-   Incremental ETL processing
-   Data warehouse integration
-   Query optimization with indexes
-   Visualization dashboards (PowerBI/Tableau)

------------------------------------------------------------------------
