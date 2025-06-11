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

# Dataset URL
DATASET_URL = 'https://www.dropbox.com/scl/fi/vy73zd0hcwpvh1gnsp2nn/RTA-Dataset.csv?rlkey=dfol7q59inukp57i2u7br4zsw&st=49kcannf&dl=1'

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Helper functions
@st.cache_data
def load_data_from_url(url):
    """Load and cache the data from URL"""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {str(e)}")
        return None

@st.cache_data
def load_data_from_upload(uploaded_file):
    """Load and cache the uploaded data"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading uploaded data: {str(e)}")
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

# Main App
def main():
    st.markdown('<h1 class="main-header">🚗 AI-Based Road Traffic Accident Analysis & Prediction</h1>', unsafe_allow_html=True)
    
    # Auto-load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading Road Traffic Accident dataset..."):
            df = load_data_from_url(DATASET_URL)
            if df is not None:
                st.session_state.df = preprocess_data(df)
                st.session_state.data_loaded = True
                st.success(f"✅ Dataset loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")
            else:
                st.error("❌ Failed to load dataset from URL. Please try uploading your own file.")
    
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
    
    # Optional file upload for custom dataset
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload Custom Dataset (Optional)")
    uploaded_file = st.sidebar.file_uploader(
        "Replace default dataset with your own CSV", 
        type=['csv'],
        help="Upload a CSV file with road accident data to override the default dataset"
    )
    
    # Handle custom file upload
    if uploaded_file is not None:
        with st.spinner("Loading custom dataset..."):
            custom_df = load_data_from_upload(uploaded_file)
            if custom_df is not None:
                st.session_state.df = preprocess_data(custom_df)
                st.session_state.data_loaded = True
                st.session_state.models_trained = False  # Reset models for new data
                st.sidebar.success(f"✅ Custom data loaded: {custom_df.shape[0]} records, {custom_df.shape[1]} columns")
    
    # Show data status
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Dataset info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.write(f"**Records:** {df.shape[0]:,}")
        st.sidebar.write(f"**Features:** {df.shape[1]}")
        st.sidebar.write(f"**Size:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
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
        st.info("⏳ Loading dataset... Please wait.")
        if st.button("🔄 Retry Loading Dataset"):
            st.session_state.data_loaded = False
            st.rerun()

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
            st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        if 'Accident_severity' in df.columns:
            avg_severity = df['Accident_severity'].mean()
            st.metric("Avg Severity", f"{avg_severity:.2f}")
        else:
            st.metric("Data Types", df.select_dtypes(include=[np.number]).shape[1])
    
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
    
    # Summary statistics for numerical columns
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

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
        labels={'x': 'Number of Accidents', 'y': 'Cause of Accident'},
        color=cause_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart for top 5 causes
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=cause_counts.head(5).values,
            names=cause_counts.head(5).index,
            title="Top 5 Accident Causes Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Summary statistics
        st.markdown("### Key Insights")
        st.info(f"**Most Common Cause:** {cause_counts.index[0]} ({cause_counts.values[0]} accidents)")
        
        total_top5 = cause_counts.head(5).sum()
        percent_top5 = (total_top5 / len(df)) * 100
        st.info(f"**Top 5 causes account for:** {percent_top5:.1f}% of all accidents")
        
        st.markdown("**Top 5 Causes:**")
        for i, (cause, count) in enumerate(cause_counts.head(5).items(), 1):
            percentage = (count / len(df)) * 100
            st.write(f"{i}. {cause}: {count} ({percentage:.1f}%)")

def show_environmental_factors(df):
    """Show environmental factors analysis"""
    st.markdown('<h2 class="sub-header">🌦️ Environmental Factors</h2>', unsafe_allow_html=True)
    
    # Check for required columns
    env_columns = ['Weather_conditions', 'Light_conditions', 'Road_surface_type']
    available_cols = [col for col in env_columns if col in df.columns]
    missing_cols = [col for col in env_columns if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Some environmental columns are missing: {', '.join(missing_cols)}")
        st.info(f"Analysis will proceed with available columns: {', '.join(available_cols)}")
    
    if not available_cols:
        st.error("No environmental factor columns found in the dataset")
        return
    
    # Create subplots based on available columns
    n_cols = len(available_cols)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=available_cols + ['Combined Analysis'],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Plot available environmental factors
    for i, col in enumerate(available_cols[:3]):
        counts = df[col].value_counts().head(8)
        row = 1 if i < 2 else 2
        col_pos = (i % 2) + 1
        
        fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, name=col),
            row=row, col=col_pos
        )
    
    # Combined analysis if we have at least 2 columns
    if len(available_cols) >= 2:
        cross_tab = pd.crosstab(df[available_cols[0]], df[available_cols[1]])
        fig.add_trace(
            go.Heatmap(z=cross_tab.values, x=cross_tab.columns, y=cross_tab.index,
                      colorscale='Blues', name="Combined"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Environmental insights
    st.markdown("### Environmental Impact Analysis")
    cols = st.columns(len(available_cols))
    
    for i, col in enumerate(available_cols):
        with cols[i]:
            counts = df[col].value_counts().head(3)
            st.markdown(f"**{col.replace('_', ' ').title()}**")
            for condition, count in counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {condition}: {count} ({percentage:.1f}%)")

def show_temporal_patterns(df):
    """Show temporal patterns analysis"""
    st.markdown('<h2 class="sub-header">⏰ Temporal Patterns</h2>', unsafe_allow_html=True)
    
    time_cols = ['Time', 'Day_of_week']
    available_time_cols = [col for col in time_cols if col in df.columns]
    
    if not available_time_cols:
        st.error("Required temporal columns ('Time' or 'Day_of_week') not found")
        return
    
    # Process time data
    if 'Time' in df.columns:
        df['Hour_of_day'] = (df['Time'] * 24).round().astype(int)
        df['Hour_of_day'] = df['Hour_of_day'].clip(0, 23)  # Ensure valid hours
    
    if 'Day_of_week' in df.columns:
        df['Day_of_week_Int'] = (df['Day_of_week'] * 6).round().astype(int)
        df['Day_of_week_Int'] = df['Day_of_week_Int'].clip(0, 6)  # Ensure valid days
        
        # Day mapping
        day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                   4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        df['Day_of_week_Label'] = df['Day_of_week_Int'].map(day_map)
    
    col1, col2 = st.columns(2)
    
    if 'Hour_of_day' in df.columns:
        with col1:
            # Hourly distribution
            st.markdown("### Accidents by Hour of Day")
            hourly_counts = df['Hour_of_day'].value_counts().sort_index()
            
            fig_hour = px.line(
                x=hourly_counts.index, 
                y=hourly_counts.values,
                title="Accident Distribution Throughout the Day",
                markers=True
            )
            fig_hour.update_traces(mode='lines+markers', line=dict(width=3))
            fig_hour.update_xaxis(title="Hour of Day", dtick=2)
            fig_hour.update_yaxis(title="Number of Accidents")
            st.plotly_chart(fig_hour, use_container_width=True)
    
    if 'Day_of_week_Label' in df.columns:
        with col2:
            # Daily distribution
            st.markdown("### Accidents by Day of Week")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_counts = df['Day_of_week_Label'].value_counts().reindex(day_order)
            
            fig_day = px.bar(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Accident Distribution by Day of Week",
                color=daily_counts.values,
                color_continuous_scale='Blues'
            )
            fig_day.update_xaxis(title="Day of Week")
            fig_day.update_yaxis(title="Number of Accidents")
            st.plotly_chart(fig_day, use_container_width=True)
    
    # Heatmap if both columns are available
    if 'Hour_of_day' in df.columns and 'Day_of_week_Label' in df.columns:
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
            title="Accident Frequency Heatmap (Hour vs Day)",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Temporal insights
    col1, col2 = st.columns(2)
    
    if 'Hour_of_day' in df.columns:
        hourly_counts = df['Hour_of_day'].value_counts().sort_index()
        peak_hour = hourly_counts.idxmax()
        with col1:
            st.success(f"**Peak Hour:** {peak_hour}:00 ({hourly_counts[peak_hour]} accidents)")
    
    if 'Day_of_week_Label' in df.columns:
        daily_counts = df['Day_of_week_Label'].value_counts()
        peak_day = daily_counts.idxmax()
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
        labels={'x': 'Number of Accidents', 'y': 'Area'},
        color=area_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Area statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for top 10 areas
        fig_pie = px.pie(
            values=area_counts.head(10).values,
            names=area_counts.head(10).index,
            title="Top 10 High-Risk Areas Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Area Risk Statistics")
        st.write(f"**Total Areas:** {df['Area_accident_occured'].nunique()}")
        
        high_risk_threshold = area_counts.quantile(0.7)
        high_risk_areas = area_counts[area_counts >= high_risk_threshold].index.tolist()
        st.write(f"**High-Risk Areas:** {len(high_risk_areas)}")
        st.write(f"**Risk Threshold:** {high_risk_threshold:.0f} accidents")
        
        st.markdown("**Top 5 High-Risk Areas:**")
        for i, (area, count) in enumerate(area_counts.head(5).items(), 1):
            percentage = (count / len(df)) * 100
            st.write(f"{i}. {area}: {count} ({percentage:.1f}%)")
    
    # Train high-risk area prediction model
    st.markdown("### High-Risk Area Prediction Model")
    
    if st.button("🎯 Train High-Risk Area Prediction Model"):
        with st.spinner("Training high-risk area prediction model..."):
            try:
                # Prepare features for area prediction
                features_for_area = []
                
                # Add numerical features
                if 'Time' in df.columns:
                    features_for_area.append('Time')
                if 'Day_of_week' in df.columns:
                    features_for_area.append('Day_of_week')
                
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
                
                if not features_for_area:
                    st.error("No suitable features found for training the model")
                    return
                
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
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                                      title="Confusion Matrix",
                                      labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': features_for_area,
                        'Importance': rf_area.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance for Area Risk Prediction"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

def show_predictive_analytics(df):
    """Show predictive analytics for accident severity"""
    st.markdown('<h2 class="sub-header">🔮 Predictive Analytics</h2>', unsafe_allow_html=True)
    
    if 'Accident_severity' not in df.columns:
        st.error("Column 'Accident_severity' not found")
        return
    
    # Display severity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        severity_counts = df['Accident_severity'].value_counts()
        fig_severity = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            title="Accident Severity Distribution",
            labels={'x': 'Severity Level', 'y': 'Count'}
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        fig_severity_pie = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Accident Severity Breakdown"
        )
        st.plotly_chart(fig_severity_pie, use_container_width=True)
    
    # Train prediction model
    if st.button("🤖 Train Accident Severity Prediction Model"):
        with st.spinner("Training accident severity prediction model..."):
            try:
                df_pred = df.copy()
                
                # Create binary target (high vs low severity)
                severity_threshold = df_pred['Accident_severity'].median()
                df_pred['High_Severity'] = (df_pred['Accident_severity'] >= severity_threshold).astype(int)
                
                # Feature selection
                prediction_features = []
                
                # Add numerical features
                if 'Time' in df_pred.columns:
                    prediction_features.append('Time')
                if 'Day_of_week' in df_pred.columns:
                    prediction_features.append('Day_of_week')
                
                # Categorical features
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
                
                if not prediction_features:
                    st.error("No suitable features found for training the prediction model")
                    return
                
                # Prepare training data
                X = df_pred[prediction_features]
                y = df_pred['High_Severity']
                
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
                st.session_state.severity_threshold = severity_threshold
                
                st.success("✅ Accident Severity Prediction Model Trained!")
                
                # Model performance
                col1, col2 = st.columns(2)
                
                with col1:
                    # Performance metrics
                    report = classification_report(y_test, y_pred_severity, output_dict=True)
                    accuracy = report['accuracy']
                    precision = report['1']['precision'] if '1' in report else 0
                    recall = report['1']['recall'] if '1' in report else 0
                    
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Precision", f"{precision:.2%}")
                    st.metric("Recall", f"{recall:.2%}")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred_severity)
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                                      title="Confusion Matrix",
                                      labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig_cm,
