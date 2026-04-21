# COVID-19 Clinical Trials Data Analysis (EDA)

Welcome to the **COVID-19 Clinical Trials Data Analysis** repository! This project conducts an indepth Exploratory Data Analysis (EDA) on a dataset consisting of COVID-19 clinical trial records to uncover trends in research focus, sponsorships, conditions, trial phases, and more.

## Overview
The primary goal of this project is to analyze clinical trial dynamics surrounding the COVID-19 pandemic. By parsing the recorded studies, it reveals:
- **Top Conditions**: The primary focus areas alongside baseline COVID-19 research based on a word-based visual study.
- **Phases Breakdown**: The distribution of trial phases across all experiments.
- **Sponsors Analysis**: Finding the top corporate and educational entities that fund or collaborate on multiple trials.
- **Execution Pipelines**: Using relational schemas (SQLite) to formulate queries extracting deep trial comparisons.

## Tech Stack
- **Python**: Primary analysis language
- **Pandas / Numpy**: Dataset cleaning, missing value imputation, and correlation mapping.
- **Matplotlib / Seaborn**: Aesthetic static charting with structured styling layouts.
- **Plotly Express**: Interactive dynamic visualizations (donut and bar charts) for fluid data consumption.
- **WordCloud**: Mapping abstract conditions studied across tens of thousands of subjects.
- **SQLite3**: Rapid local tabular database querying.

## Key Features
- **Dynamic Visuals**: Re-rendered graphics that guarantee NO overlapping labels or cut-off legends.
- **Structured Outputs**: Extracts structured `.csv` reports highlighting categorical/numeric insights immediately upon execution.
- **Reproducible Storage**: Seamless connection string mappings that output direct database frames for remote fetching.

## Usage
1. Open the primary exploratory notebook: `unified covid prj.ipynb` locally using Jupyter Notebooks or JupyterLab.
2. The initial codebase expects `COVID clinical trials (2).csv` to be in the identically mounted root direction. 
3. Run all cells from `Cell 1` to process the exploratory graphs and re-generate the SQL query statistics.

Enjoy analyzing!
