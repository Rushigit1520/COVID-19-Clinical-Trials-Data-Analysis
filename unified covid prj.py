#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from wordcloud import WordCloud
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")


# In[20]:


df = pd.read_csv("COVID clinical trials (2).csv")


# In[21]:


df.head()


# In[22]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# In[23]:


print(df.dtypes)


# In[24]:


print(df.isnull().sum())


# In[25]:


plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Overview")
plt.tight_layout()
plt.show()


# In[26]:


print(df.describe(include="all"))


# In[27]:


for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")


# In[37]:


status_counts = df['Status'].value_counts()


# In[50]:


fig = px.bar(x=status_counts.index, y=status_counts.values, 
             labels={'x': 'Status', 'y': 'Number of Studies'},
             title="Number of COVID-19 Clinical Trials by Status",
             color=status_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()


# In[39]:


phase_counts = df['Phases'].value_counts()


# In[51]:


fig = px.pie(values=phase_counts.values, names=phase_counts.index, 
             title="Distribution of COVID-19 Clinical Trials by Phase", hole=0.3,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[42]:


condition_counts = df['Conditions'].value_counts().head(10)


# In[52]:


plt.figure(figsize=(12,6))
sns.barplot(x=condition_counts.index, y=condition_counts.values, palette="magma")
plt.title("Top 10 Conditions in COVID-19 Clinical Trials")
plt.xlabel("Condition")
plt.ylabel("Number of Studies")
plt.xticks(rotation=90)  
plt.tight_layout()
plt.tight_layout()
plt.show()


# Also display a Word Cloud for a better aesthetic
text = " ".join(df['Conditions'].dropna().astype(str).tolist())
if len(text) > 0:
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Clinical Trial Conditions", fontsize=16)
    plt.tight_layout()
    plt.show()


# In[45]:


sponsor_counts = df['Sponsor/Collaborators'].value_counts().head(10)


# In[49]:


plt.figure(figsize=(12,6))
sns.barplot(x=sponsor_counts.index, y=sponsor_counts.values, palette="cool")
plt.title("Top 10 Sponsors in COVID-19 Clinical Trials")
plt.xlabel("Sponsor/Collaborators")
plt.ylabel("Number of Studies")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[53]:


gender_counts = df['Gender'].value_counts()


# In[54]:


plt.figure(figsize=(8,5))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="pastel")
plt.title("Gender Distribution in COVID-19 Clinical Trials")
plt.xlabel("Gender")
plt.ylabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[63]:


import textwrap 


# In[64]:


age_counts = df['Age'].value_counts()


# In[71]:


def clean_age(age):
    if pd.isna(age):
        return "Not Provided"
    age = age.lower()
    if "child" in age and "adult" in age and "older" in age:
        return "Ch+Ad+OA"
    elif "child" in age and "adult" in age:
        return "Ch+Ad"
    elif "adult" in age and "older" in age:
        return "Ad+OA"
    elif "child" in age:
        return "Child"
    elif "adult" in age:
        return "Adult"
    elif "older" in age:
        return "OA"
    elif "all" in age:
        return "All Ages"
    else:
        return "Not Provided"


# In[72]:


df['Age_Clean'] = df['Age'].apply(clean_age)


# In[74]:


age_counts = df['Age_Clean'].value_counts()


# In[75]:


plt.figure(figsize=(8,5))
sns.barplot(x=age_counts.index, y=age_counts.values, palette="spring")
plt.title("Distribution of Age Groups in COVID-19 Clinical Trials", fontsize=16)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Number of Studies", fontsize=12)
plt.tight_layout()
plt.show()


# In[76]:


df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
enrollment_counts = df['Enrollment'].dropna()


# In[77]:


plt.figure(figsize=(10,6))
sns.histplot(enrollment_counts, bins=20, kde=False, color='skyblue')
plt.title("Distribution of Enrollment in COVID-19 Clinical Trials", fontsize=16)
plt.xlabel("Number of Participants", fontsize=12)
plt.ylabel("Number of Studies", fontsize=12)
plt.tight_layout()
plt.show()


# In[91]:


top_outcomes = df['Outcome Measures'].value_counts().head(10)


# In[92]:


short_labels = {
    'Time to clinical improvement': 'Time to Improvement',
    'Overall survival': 'Survival',
    'Mortality rate': 'Mortality',
    'Viral load reduction': 'Viral Load',
    'Adverse events': 'Adverse Events',
    'Hospitalization duration': 'Hosp. Duration',
    'ICU admission': 'ICU Admission',
    'Safety and tolerability': 'Safety',
    'Symptom resolution': 'Symptom Res.',
    'Seroconversion': 'Seroconversion',
    'Unknown': 'Unknown'
}


# In[93]:


top_outcomes.index = [short_labels.get(x, x) for x in top_outcomes.index]


# In[94]:


top_outcomes = top_outcomes.sort_values(ascending=True)  # For horizontal bar, smallest on top


# In[95]:


plt.figure(figsize=(10,6))
sns.barplot(x=top_outcomes.values, y=top_outcomes.index, palette="cool")
plt.title("Top 10 Outcome Measures in COVID-19 Clinical Trials", fontsize=16)
plt.xlabel("Number of Studies", fontsize=12)
plt.ylabel("Outcome Measure", fontsize=12)
plt.tight_layout()
plt.show()


# In[96]:


missing_values = df.isnull().sum()
print(missing_values)


# In[97]:


num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())


# In[98]:


obj_cols = df.select_dtypes(include=['object']).columns
for col in obj_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[99]:


df.describe(include='all')


# In[101]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Status', order=df['Status'].value_counts().index)
plt.title("Number of Studies by Status")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[102]:


top_conditions = df['Conditions'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(y=top_conditions.index, x=top_conditions.values, palette='viridis')
plt.title("Top 10 Conditions Studied in COVID Trials")
plt.xlabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[103]:


top_interventions = df['Interventions'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(y=top_interventions.index, x=top_interventions.values, palette='magma')
plt.title("Top 10 Interventions in COVID Trials")
plt.xlabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[104]:


top_sponsors = df['Sponsor/Collaborators'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(y=top_sponsors.index, x=top_sponsors.values, palette='cubehelix')
plt.title("Top 10 Sponsors in COVID Trials")
plt.xlabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[106]:


plt.figure(figsize=(8,5))
sns.histplot(df['Enrollment'], bins=30, kde=True)
plt.title("Distribution of Study Enrollment")
plt.xlabel("Number of Participants")
plt.tight_layout()
plt.show()


# In[107]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Phases', order=df['Phases'].value_counts().index)
plt.title("Studies by Phase")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[108]:


plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender')
plt.title("Distribution of Studies by Gender")
plt.tight_layout()
plt.show()


# In[111]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Study Type', order=df['Study Type'].value_counts().index)
plt.title("Studies by Study Type")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[113]:


plt.figure(figsize=(10,6))
top_funders = df['Funded Bys'].value_counts().head(10)
sns.barplot(y=top_funders.index, x=top_funders.values, palette='coolwarm')
plt.title("Top 10 Funding Agencies for COVID Trials")
plt.xlabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[114]:


top_outcomes = df['Outcome Measures'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(y=top_outcomes.index, x=top_outcomes.values, palette='spring')
plt.title("Top 10 Outcome Measures in COVID Trials")
plt.xlabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[115]:


plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# In[116]:


import sqlite3

# Create a SQLite database connection
conn = sqlite3.connect('covid_trials.db')
df.to_sql('covid_trials', conn, if_exists='replace', index=False)

# Confirm tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print(tables)


# In[117]:


query = "SELECT COUNT(*) as total_studies FROM covid_trials"
total_studies = pd.read_sql(query, conn)
print(total_studies)


# In[118]:


query = """
SELECT [Sponsor/Collaborators], COUNT(*) as study_count
FROM covid_trials
GROUP BY [Sponsor/Collaborators]
ORDER BY study_count DESC
LIMIT 5
"""
top_sponsors_sql = pd.read_sql(query, conn)
print(top_sponsors_sql)


# In[119]:


query = """
SELECT [Study Type], AVG(Enrollment) as avg_enrollment
FROM covid_trials
GROUP BY [Study Type]
ORDER BY avg_enrollment DESC
"""
avg_enrollment_sql = pd.read_sql(query, conn)
print(avg_enrollment_sql)


# In[120]:


query = """
SELECT Phases, Status, COUNT(*) as study_count
FROM covid_trials
GROUP BY Phases, Status
ORDER BY Phases
"""
phase_status = pd.read_sql(query, conn)
print(phase_status)


# In[121]:


plt.figure(figsize=(8,5))
sns.barplot(y=top_sponsors_sql['Sponsor/Collaborators'], x=top_sponsors_sql['study_count'], palette='cool')
plt.title("Top 5 Sponsors from SQL Query")
plt.xlabel("Number of Studies")
plt.tight_layout()
plt.show()


# In[122]:


query = "SELECT * FROM covid_trials WHERE Enrollment > 1000 ORDER BY Enrollment DESC"
large_studies = pd.read_sql(query, conn)
print(large_studies.head())


# In[123]:


query = """
SELECT Gender, Phases, COUNT(*) as count_studies
FROM covid_trials
GROUP BY Gender, Phases
ORDER BY Gender
"""
gender_phase = pd.read_sql(query, conn)
print(gender_phase)


# In[125]:


query = """
SELECT [Funded Bys], COUNT(DISTINCT Phases) as phases_count
FROM covid_trials
GROUP BY [Funded Bys]
HAVING phases_count > 1
ORDER BY phases_count DESC
"""
multi_phase_funders = pd.read_sql(query, conn)
print(multi_phase_funders)


# In[126]:


# Save the cleaned & processed dataset
df.to_csv('covid_trials_cleaned.csv', index=False)
print("Cleaned dataset saved as covid_trials_cleaned.csv")


# In[127]:


conn = sqlite3.connect('covid_trials_final.db')
df.to_sql('covid_trials', conn, if_exists='replace', index=False)
print("Cleaned dataset saved to SQL database: covid_trials_final.db")


# In[128]:


# Get numeric summary
num_summary = df.describe()
num_summary.to_csv('numeric_summary.csv')
print("Numeric summary saved as numeric_summary.csv")

# Get categorical summary
cat_summary = df.describe(include='object')
cat_summary.to_csv('categorical_summary.csv')
print("Categorical summary saved as categorical_summary.csv")


# In[129]:


# Example: Top 5 Sponsors
query = """
SELECT [Sponsor/Collaborators], COUNT(*) as study_count
FROM covid_trials
GROUP BY [Sponsor/Collaborators]
ORDER BY study_count DESC
LIMIT 5
"""
top_sponsors_sql = pd.read_sql(query, conn)
top_sponsors_sql.to_csv('top_sponsors.csv', index=False)
print("Top 5 sponsors saved as top_sponsors.csv")


# In[130]:


plt.figure(figsize=(8,5))
sns.histplot(df['Enrollment'], bins=30, kde=True)
plt.title("Distribution of Study Enrollment")
plt.xlabel("Number of Participants")
plt.savefig('enrollment_distribution.png')
plt.close()
print("Enrollment distribution plot saved as enrollment_distribution.png")


# In[ ]:




