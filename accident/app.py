import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Road Traffic Accident Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Dataset URL
DATASET_URL = 'https://www.dropbox.com/scl/fi/vy73zd0hcwpvh1gnsp2nn/RTA-Dataset.csv?rlkey=dfol7q59inukp57i2u7br4zsw&st=49kcannf&dl=1'

# Helper functions
@st.cache_data
def load_data_from_url(url):
    """Load and cache data from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save content to a temporary file-like object
        from io import StringIO
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {str(e)}")
        return None

@st.cache_data
def load_data_from_file(uploaded_file):
    """Load and cache the uploaded data"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data for analysis"""
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.dropna()
    
    return df_processed

def create_cause_mapping(df):
    """Create accident cause mapping"""
    # Get unique causes and create mapping
    unique_causes = df['Cause_of_accident'].unique()
    cause_mapping = {}
    
    # Simple mapping logic based on keywords
    for i, cause in enumerate(unique_causes[:5]):  # Take first 5 causes
        if 'speed' in str(cause).lower():
            cause_mapping[i] = 'Speeding'
        elif 'drunk' in str(cause).lower() or 'alcohol' in str(cause).lower():
            cause_mapping[i] = 'Drunk Driving'
        elif 'distract' in str(cause).lower():
            cause_mapping[i] = 'Distracted Driving'
        elif 'weather' in str(cause).lower():
            cause_mapping[i] = 'Weather'
        else:
            cause_mapping[i] = 'Other'
    
    return cause_mapping

# Auto-load dataset function
def auto_load_dataset():
    """Automatically load the dataset from the provided URL"""
    if not st.session_state.data_loaded:
        with st.spinner("Loading Road Traffic Accident dataset..."):
            df = load_data_from_url(DATASET_URL)
            if df is not None:
                st.session_state.df = preprocess_data(df)
                st.session_state.data_loaded = True
                st.success(f"✅ Dataset loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")
                return True
            else:
                st.error("❌ Failed to load dataset from URL")
                return False
    return True

# Main App
def main():
    st.markdown('<h1 class="main-header">🚗 AI-Based Road Traffic Accident Analysis & Prediction</h1>', unsafe_allow_html=True)
    
    # Auto-load dataset
    dataset_loaded = auto_load_dataset()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📊 Data Overview",
        "🚗 Primary Accident Causes", 
        "🌦️ Environmental Factors",
        "⏰ Temporal Patterns",
        "📍 High-Risk Areas",
        "🔮 Predictive Analytics"
    ])
    
    # Optional file upload (if user wants to use their own data)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Optional: Upload Your Own Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your accident dataset (CSV)", 
        type=['csv'],
        help="Upload a CSV file with road accident data to override the default dataset"
    )
    
    # Handle file upload or use auto-loaded data
    if uploaded_file is not None:
        # Load user's own data
        with st.spinner("Loading your uploaded data..."):
            df = load_data_from_file(uploaded_file)
        
        if df is not None:
            st.session_state.df = preprocess_data(df)
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ Your data loaded: {df.shape[0]} records, {df.shape[1]} columns")
            df = st.session_state.df
        else:
            df = st.session_state.df if st.session_state.data_loaded else None
    elif dataset_loaded and st.session_state.data_loaded:
        df = st.session_state.df
        st.sidebar.info(f"📊 Using default dataset: {df.shape[0]} records, {df.shape[1]} columns")
    else:
        df = None
    
    # Show analysis if data is available
    if df is not None:
        # Show selected page
        if page == "📊 Data Overview":
            show_data_overview(df)
        elif page == "🚗 Primary Accident Causes":
            show_accident_causes(df)
        elif page == "🌦️ Environmental Factors":
            show_environmental_factors(df)
        elif page == "⏰ Temporal Patterns":
            show_temporal_patterns(df)
        elif page == "📍 High-Risk Areas":
            show_high_risk_areas(df)
        elif page == "🔮 Predictive Analytics":
            show_predictive_analytics(df)
    else:
        st.info("🔄 Loading dataset... Please wait.")
        
        # Show sample data structure
        st.markdown("### Expected Data Structure")
        sample_columns = [
            'Time', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
            'Type_of_vehicle', 'Area_accident_occured', 'Weather_conditions',
            'Light_conditions', 'Road_surface_type', 'Cause_of_accident',
            'Accident_severity'
        ]
        st.code(", ".join(sample_columns))

def show_data_overview(df):
    """Display data overview and statistics"""
    st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        if 'Area_accident_occured' in df.columns:
            st.metric("Unique Areas", df['Area_accident_occured'].nunique())
        else:
            st.metric("Unique Areas", "N/A")
    with col4:
        if 'Accident_severity' in df.columns:
            avg_severity = df['Accident_severity'].mean()
            st.metric("Avg Severity", f"{avg_severity:.2f}")
        else:
            st.metric("Avg Severity", "N/A")
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Types")
        data_types = pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values
        })
        st.dataframe(data_types, use_container_width=True)
    
    with col2:
        st.markdown("### Missing Values")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(missing_data, use_container_width=True)

def show_accident_causes(df):
    """Show primary accident causes analysis"""
    st.markdown('<h2 class="sub-header">🚗 Primary Accident Causes</h2>', unsafe_allow_html=True)
    
    if 'Cause_of_accident' not in df.columns:
        st.error("Column 'Cause_of_accident' not found in the dataset")
        return
    
    # Create cause mapping
    cause_mapping = create_cause_mapping(df)
    
    # Apply mapping (simplified version)
    cause_counts = df['Cause_of_accident'].value_counts().head(10)
    
    # Create visualization
    fig = px.bar(
        x=cause_counts.values,
        y=cause_counts.index,
        orientation='h',
        title="Top 10 Primary Causes of Road Accidents",
        labels={'x': 'Number of Accidents', 'y': 'Cause of Accident'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("### Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Most Common Cause:** {cause_counts.index[0]} ({cause_counts.values[0]} accidents)")
    
    with col2:
        total_top5 = cause_counts.head(5).sum()
        percent_top5 = (total_top5 / len(df)) * 100
        st.info(f"**Top 5 causes account for:** {percent_top5:.1f}% of all accidents")

def show_environmental_factors(df):
    """Show environmental factors analysis"""
    st.markdown('<h2 class="sub-header">🌦️ Environmental Factors</h2>', unsafe_allow_html=True)
    
    # Check for required columns
    env_columns = ['Weather_conditions', 'Light_conditions', 'Road_surface_type']
    missing_cols = [col for col in env_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weather Conditions', 'Light Conditions', 'Road Surface Type', 'Combined Analysis'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Weather conditions
    weather_counts = df['Weather_conditions'].value_counts().head(8)
    fig.add_trace(
        go.Bar(x=weather_counts.index, y=weather_counts.values, name="Weather"),
        row=1, col=1
    )
    
    # Light conditions  
    light_counts = df['Light_conditions'].value_counts().head(8)
    fig.add_trace(
        go.Bar(x=light_counts.index, y=light_counts.values, name="Light"),
        row=1, col=2
    )
    
    # Road surface type
    road_counts = df['Road_surface_type'].value_counts().head(8)
    fig.add_trace(
        go.Bar(x=road_counts.index, y=road_counts.values, name="Road Surface"),
        row=2, col=1
    )
    
    # Combined heatmap data
    if len(df) > 0:
        cross_tab = pd.crosstab(df['Weather_conditions'], df['Light_conditions'])
        fig.add_trace(
            go.Heatmap(z=cross_tab.values, x=cross_tab.columns, y=cross_tab.index,
                      colorscale='Blues', name="Weather vs Light"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Environmental insights
    st.markdown("### Environmental Impact Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Weather Conditions**")
        for condition, count in weather_counts.head(3).items():
            st.write(f"• {condition}: {count} accidents")
    
    with col2:
        st.markdown("**Light Conditions**")
        for condition, count in light_counts.head(3).items():
            st.write(f"• {condition}: {count} accidents")
    
    with col3:
        st.markdown("**Road Surface**")
        for condition, count in road_counts.head(3).items():
            st.write(f"• {condition}: {count} accidents")

def show_temporal_patterns(df):
    """Show temporal patterns analysis"""
    st.markdown('<h2 class="sub-header">⏰ Temporal Patterns</h2>', unsafe_allow_html=True)
    
    if 'Time' not in df.columns or 'Day_of_week' not in df.columns:
        st.error("Required columns 'Time' or 'Day_of_week' not found")
        return
    
    # Process time data
    df['Hour_of_day'] = (df['Time'] * 24).round().astype(int)
    df['Day_of_week_Int'] = (df['Day_of_week'] * 6).round().astype(int)
    
    # Day mapping
    day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
               4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    df['Day_of_week_Label'] = df['Day_of_week_Int'].map(day_map)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution
        st.markdown("### Accidents by Hour of Day")
        hourly_counts = df['Hour_of_day'].value_counts().sort_index()
        
        fig_hour = px.line(
            x=hourly_counts.index, 
            y=hourly_counts.values,
            title="Accident Distribution Throughout the Day"
        )
        fig_hour.update_traces(mode='lines+markers')
        fig_hour.update_xaxis(title="Hour of Day", dtick=2)
        fig_hour.update_yaxis(title="Number of Accidents")
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col2:
        # Daily distribution
        st.markdown("### Accidents by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = df['Day_of_week_Label'].value_counts().reindex(day_order)
        
        fig_day = px.bar(
            x=daily_counts.index,
            y=daily_counts.values,
            title="Accident Distribution by Day of Week"
        )
        fig_day.update_xaxis(title="Day of Week")
        fig_day.update_yaxis(title="Number of Accidents")
        st.plotly_chart(fig_day, use_container_width=True)
    
    # Heatmap
    st.markdown("### Hourly-Daily Accident Heatmap")
    pivot_table = df.pivot_table(
        values='Time', 
        index='Hour_of_day', 
        columns='Day_of_week_Label', 
        aggfunc='count', 
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        aspect="auto",
        title="Accident Frequency Heatmap (Hour vs Day)"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Temporal insights
    peak_hour = hourly_counts.idxmax()
    peak_day = daily_counts.idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Peak Hour:** {peak_hour}:00 ({hourly_counts[peak_hour]} accidents)")
    with col2:
        st.success(f"**Peak Day:** {peak_day} ({daily_counts[peak_day]} accidents)")

def show_high_risk_areas(df):
    """Show high-risk areas analysis"""
    st.markdown('<h2 class="sub-header">📍 High-Risk Areas Analysis</h2>', unsafe_allow_html=True)
    
    if 'Area_accident_occured' not in df.columns:
        st.error("Column 'Area_accident_occured' not found")
        return
    
    # Area analysis
    area_counts = df['Area_accident_occured'].value_counts().head(20)
    
    # Top risk areas visualization
    fig = px.bar(
        x=area_counts.values,
        y=area_counts.index,
        orientation='h',
        title="Top 20 High-Risk Areas",
        labels={'x': 'Number of Accidents', 'y': 'Area'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Train clustering model
    st.markdown("### High-Risk Area Prediction Model")
    
    with st.spinner("Training high-risk area prediction model..."):
        # Prepare features for clustering/classification
        features_for_area = ['Time', 'Day_of_week']
        
        # Add categorical features if available
        categorical_features = ['Sex_of_driver', 'Type_of_vehicle', 'Weather_conditions']
        available_features = [f for f in categorical_features if f in df.columns]
        
        if available_features:
            # Encode categorical variables
            df_encoded = df.copy()
            le_dict = {}
            
            for col in available_features:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le
                features_for_area.append(col + '_encoded')
        
        # Create high-risk labels
        high_risk_threshold = area_counts.quantile(0.7)
        high_risk_areas = area_counts[area_counts >= high_risk_threshold].index.tolist()
        df_encoded = df_encoded if 'df_encoded' in locals() else df.copy()
        df_encoded['High_Risk_Area'] = df_encoded['Area_accident_occured'].isin(high_risk_areas).astype(int)
        
        # Train model
        X = df_encoded[features_for_area]
        y = df_encoded['High_Risk_Area']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_area = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_area.fit(X_train, y_train)
        
        y_pred = rf_area.predict(X_test)
        
        # Model performance
        st.success("✅ High-Risk Area Prediction Model Trained!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = report['accuracy']
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': features_for_area,
                'Importance': rf_area.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance for Area Risk Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # High-risk area stats
            st.markdown("### High-Risk Area Statistics")
            st.write(f"**Total Areas:** {df['Area_accident_occured'].nunique()}")
            st.write(f"**High-Risk Areas:** {len(high_risk_areas)}")
            st.write(f"**Risk Threshold:** {high_risk_threshold:.0f} accidents")
            
            # Top 5 high-risk areas
            st.markdown("**Top 5 High-Risk Areas:**")
            for i, (area, count) in enumerate(area_counts.head(5).items(), 1):
                st.write(f"{i}. {area}: {count} accidents")

def show_predictive_analytics(df):
    """Show predictive analytics for accident severity"""
    st.markdown('<h2 class="sub-header">🔮 Predictive Analytics</h2>', unsafe_allow_html=True)
    
    if 'Accident_severity' not in df.columns:
        st.error("Column 'Accident_severity' not found")
        return
    
    # Prepare data for severity prediction
    with st.spinner("Training accident severity prediction model..."):
        df_pred = df.copy()
        
        # Create binary target
        df_pred['Accident_severity_binary'] = (df_pred['Accident_severity'] >= 0.5).astype(int)
        
        # Feature selection
        prediction_features = ['Time', 'Day_of_week']
        categorical_features = ['Sex_of_driver', 'Type_of_vehicle', 'Weather_conditions', 
                              'Light_conditions', 'Road_surface_type']
        
        # Encode categorical variables
        le_dict_severity = {}
        for col in categorical_features:
            if col in df_pred.columns:
                le = LabelEncoder()
                df_pred[col + '_encoded'] = le.fit_transform(df_pred[col].astype(str))
                le_dict_severity[col] = le
                prediction_features.append(col + '_encoded')
        
        # Prepare training data
        X = df_pred[prediction_features]
        y = df_pred['Accident_severity_binary']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_severity = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_severity.fit(X_train, y_train)
        
        y_pred_severity = rf_severity.predict(X_test)
        
        # Store models in session state
        st.session_state.rf_severity = rf_severity
        st.session_state.le_dict_severity = le_dict_severity
        st.session_state.prediction_features = prediction_features
        st.session_state.models_trained = True
    
    st.success("✅ Accident Severity Prediction Model Trained!")
    
    # Model performance
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics
        report = classification_report(y_test, y_pred_severity, output_dict=True)
        accuracy = report['accuracy']
        precision = report['1']['precision']
        recall = report['1']['recall']
        
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.metric("Precision", f"{precision:.2%}")
        st.metric("Recall", f"{recall:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_severity)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': prediction_features,
            'Importance': rf_severity.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Severity Prediction"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction interface
    st.markdown("### 🎯 Make Predictions")
    st.info("Use the trained model to predict accident severity for new scenarios")
    
    if st.session_state.models_trained:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_time = st.slider("Time of Day", 0.0, 1.0, 0.5, 0.01)
                pred_day = st.slider("Day of Week", 0.0, 1.0, 0.5, 0.01)
            
            with col2:
                if 'Sex_of_driver' in df.columns:
                    pred_gender = st.selectbox("Driver Gender", df['Sex_of_driver'].unique())
                if 'Type_of_vehicle' in df.columns:
                    pred_vehicle = st.selectbox("Vehicle Type", df['Type_of_vehicle'].unique())
            
            with col3:
                if 'Weather_conditions' in df.columns:
                    pred_weather = st.selectbox("Weather", df['Weather_conditions'].unique())
                if 'Light_conditions' in df.columns:
                    pred_light = st.selectbox("Light Conditions", df['Light_conditions'].unique())
            
            submit_prediction = st.form_submit_button("🔮 Predict Accident Severity")
            
            if submit_prediction:
                # Prepare prediction data
                pred_data = [pred_time, pred_day]
                
                # Encode categorical variables
                for col in categorical_features:
                    if col in df.columns and col in st.session_state.le_dict_severity:
                        if col == 'Sex_of_driver':
                            encoded_val = st.session_state.le_dict_severity[col].transform([pred_gender])[0]
                        elif col == 'Type_of_vehicle':
                            encoded_val = st.session_state.le_dict_severity[col].transform([pred_vehicle])[0]
                        elif col == 'Weather_conditions':
                            encoded_val = st.session_state.le_dict_severity[col].transform([pred_weather])[0]
                        elif col == 'Light_conditions':
                            encoded_val = st.session_state.le_dict_severity[col].transform([pred_light])[0]
                        else:
                            encoded_val = 0  # Default value
                        pred_data.append(encoded_val)
                
                # Make prediction
                prediction = st.session_state.rf_severity.predict([pred_data])[0]
                probability = st.session_state.rf_severity.predict_proba([pred_data])[0]
                
                # Display results
                if prediction == 1:
                    st.error(f"⚠️ **High Severity Accident Predicted** (Probability: {probability[1]:.2%})")
                else:
                    st.success(f"✅ **Low Severity Accident Predicted** (Probability: {probability[0]:.2%})")

if __name__ == "__main__":
    main()