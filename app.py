import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(
    page_title="COVID-19 Clinical Trials",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("covid_trials_cleaned.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Insights", "ML Model"])

# --- Helper logic for ML ---
@st.cache_resource
def train_model(data):
    # Prepare data
    # Target: Is Completed?
    df_ml = data.copy()
    df_ml['Is_Completed'] = df_ml['Status'].apply(lambda x: 1 if x == 'Completed' else 0)
    
    # Features
    features = ['Enrollment', 'Phases', 'Study Type', 'Age_Clean']
    df_ml = df_ml[features + ['Is_Completed']].dropna()
    
    # Encode categorical variables
    encoders = {}
    for col in ['Phases', 'Study Type', 'Age_Clean']:
        encoders[col] = LabelEncoder()
        df_ml[col] = encoders[col].fit_transform(df_ml[col].astype(str))
        
    X = df_ml[features]
    y = df_ml['Is_Completed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, encoders, acc, df_ml

# --- Pages ---

if page == "Dashboard":
    st.title("🦠 COVID-19 Clinical Trials Dashboard")
    st.markdown("Explore the landscape of COVID-19 clinical trials worldwide.")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trials", f"{len(df):,}")
    col2.metric("Total Enrollment", f"{df['Enrollment'].sum():,.0f}")
    
    # Active vs Completed
    active_trials = len(df[df['Status'].str.contains('Recruiting|Active', na=False, case=False)])
    completed_trials = len(df[df['Status'] == 'Completed'])
    col3.metric("Active/Recruiting Trials", f"{active_trials:,}")
    col4.metric("Completed Trials", f"{completed_trials:,}")
    
    st.markdown("---")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Trials by Status")
        status_counts = df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_status = px.bar(status_counts, x='Count', y='Status', orientation='h', 
                            title="Count of Trials per Status", color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig_status, use_container_width=True)
        
    with row1_col2:
        st.subheader("Trials by Phase")
        phase_counts = df['Phases'].value_counts().reset_index()
        phase_counts.columns = ['Phase', 'Count']
        # filter out 'Not Applicable' or overwhelming values to see better
        phase_counts_visual = phase_counts[phase_counts['Phase'] != 'Not Applicable']
        fig_phase = px.pie(phase_counts_visual, values='Count', names='Phase', 
                           title="Distribution of Trial Phases (Excl. N/A)", hole=0.4)
        st.plotly_chart(fig_phase, use_container_width=True)
        
    st.markdown("---")
    
    st.subheader("Top 10 Locations by Trial Count")
    # A bit of transformation since locations can be multiple separated by pipeline
    # For simplicity, we just use the first location or raw count if it's already cleaned
    if 'Locations' in df.columns:
        # Simple extraction of country/main location
        locations_split = df['Locations'].str.split('|').explode().str.split(',').str[-1].str.strip()
        loc_counts = locations_split.value_counts().head(10).reset_index()
        loc_counts.columns = ['Location', 'Count']
        fig_loc = px.bar(loc_counts, x='Location', y='Count', title="Top 10 Locations", color='Count', color_continuous_scale='Teal')
        st.plotly_chart(fig_loc, use_container_width=True)

elif page == "Insights":
    st.title("💡 Key Insights")
    st.markdown("Summary of findings from the COVID-19 Clinical Trials dataset.")
    
    st.info("""
    **Insight 1: Massive Global Effort**  
    The dataset encompasses over thousands of trials and millions in total targeted enrollment, highlighting a massive, rapid global response to the pandemic.
    """)
    
    st.success("""
    **Insight 2: Observational vs Interventional**  
    A large portion of the studies are Observational, helping understand the disease's natural history, but there is also a very significant push for Interventional trials focusing on vaccines and therapeutics.
    """)
    
    st.warning("""
    **Insight 3: Attrition and Status**  
    Many trials remain in the "Recruiting" or "Active, not recruiting" phases. The completion rate will gradually increase over time as long-term follow-ups conclude.
    """)
    
    # Calculate some dynamic insights
    st.markdown("### Data-Driven Highlights")
    top_sponsor = df['Sponsor/Collaborators'].value_counts().idxmax()
    top_phase = df['Phases'].value_counts().idxmax()
    avg_enrollment = df['Enrollment'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Sponsor", top_sponsor.split('|')[0])
    with col2:
        st.metric("Most Common Phase", top_phase)
    with col3:
        st.metric("Avg Enrollment", f"{avg_enrollment:,.0f}")
        
    st.markdown("### Raw Data Preview")
    st.dataframe(df.head(50))

elif page == "ML Model":
    st.title("🤖 Predict Trial Completion")
    st.markdown("A simple Machine Learning model (Random Forest Classifier) to predict whether a clinical trial is likely to be **Completed** based on its attributes.")
    
    with st.spinner("Training model..."):
        model, encoders, acc, df_ml = train_model(df)
        
    st.success(f"Model trained successfully! Accuracy on test set: **{acc:.2%}**")
    
    st.markdown("### Make a Prediction")
    st.write("Adjust the parameters below to see if hypothetical trial settings are associated with being Completed.")
    
    # Form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            enrollment_val = st.number_input("Expected Enrollment", min_value=1, max_value=1000000, value=500, step=50)
            phase_val = st.selectbox("Trial Phase", sorted(df['Phases'].dropna().unique()))
        
        with col2:
            study_type_val = st.selectbox("Study Type", sorted(df['Study Type'].dropna().unique()))
            age_val = st.selectbox("Age Group Targeted (Age_Clean)", sorted(df['Age_Clean'].dropna().unique()))
            
        submit = st.form_submit_button("Predict Completion")
        
    if submit:
        # Prepare the input for prediction using the saved encoders
        # Handle cases where the selected value wasn't in the training set (fallback to 0 or mode)
        try:
            phase_encoded = encoders['Phases'].transform([phase_val])[0]
        except:
            phase_encoded = 0
            
        try:
            type_encoded = encoders['Study Type'].transform([study_type_val])[0]
        except:
            type_encoded = 0
            
        try:
            age_encoded = encoders['Age_Clean'].transform([age_val])[0]
        except:
            age_encoded = 0
            
        input_data = pd.DataFrame({
            'Enrollment': [enrollment_val],
            'Phases': [phase_encoded],
            'Study Type': [type_encoded],
            'Age_Clean': [age_encoded]
        })
        
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        st.markdown("### Result")
        if prediction == 1:
            st.success(f"**Likely to Complete** (Probability: {prob[1]:.2%})")
        else:
            st.warning(f"**Less Likely to Complete / Still Active** (Probability of *not* completing: {prob[0]:.2%})")

