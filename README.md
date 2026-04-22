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
- **Interactive Dashboard**: A fully featured Streamlit app that provides exploratory visualizations.
- **Machine Learning**: An embedded predictive model (Random Forest Classifier) to estimate the completion rate of clinical trials given specific trial features.

## Usage

### 1. Jupyter Notebook (Exploratory Data Analysis)
1. Open the primary exploratory notebook: `unified covid prj.ipynb` locally using Jupyter Notebooks or JupyterLab.
2. The initial codebase expects `COVID clinical trials (2).csv` to be present in the root directory. 
3. Run all cells to process the exploratory graphs and re-generate the SQL query statistics.

### 2. Streamlit Application (Dashboard & ML)
To run the interactive web application containing the **Dashboard, ML Model, and Insights** sections:
1. Make sure you have the required Python packages installed:
   ```bash
   pip install streamlit pandas scikit-learn plotly
   ```
2. Navigate to your project folder in the terminal and launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open the provided `localhost` link (default is http://localhost:8501) in your web browser.

Enjoy analyzing!
