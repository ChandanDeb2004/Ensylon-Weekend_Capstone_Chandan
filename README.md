## Introduction
---------------
This weeks Capstone project contains ETL processing of two different datasets 
1.Airline Flights ETL Pipeline
2.Stock Market ETL Pipeline
----------------

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

  Column                        Description
  ----------------------------- ---------------------------
  Date                          Flight date
  Departure Time                Scheduled departure
  Flight Number                 Unique identifier
  Airline                       Airline company
  Aircraft Type                 Aircraft model
  Origin Airport                Departure airport
  Destination Airport           Arrival airport
  Distance (km)                 Flight distance
  Passengers (First Class)      First class passengers
  Passengers (Business Class)   Business class passengers
  Passengers (Economy Class)    Economy passengers
  Revenue (\$)                  Revenue generated
  Arrival Gate                  Arrival gate
  Departure Gate                Departure gate

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

## Author

Chandan Deb
