import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os

# Import custom components
from components.data_collector import SystemMetricsCollector
from components.ml_predictor import PerformancePredictor
from components.ai_assistant import AIAssistant
from components.dashboard import Dashboard
from components.alerts import AlertManager
from utils.data_storage import DataStorage
from utils.helpers import format_bytes, get_color_for_usage

# Page configuration
st.set_page_config(
    page_title="System Performance Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components"""
    collector = SystemMetricsCollector()
    predictor = PerformancePredictor()
    ai_assistant = AIAssistant()
    alert_manager = AlertManager()
    data_storage = DataStorage()
    dashboard = Dashboard()
    
    return collector, predictor, ai_assistant, alert_manager, data_storage, dashboard

def main():
    # Initialize components
    collector, predictor, ai_assistant, alert_manager, data_storage, dashboard = initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("🖥️ System Monitor")
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Real-time Dashboard", "Performance Predictions", "AI Assistant", "Alerts & Settings", "Data Export"]
    )
    
    # Theme toggle
    if st.sidebar.button("🌙 Toggle Dark Mode"):
        st.rerun()
    
    # Auto-refresh settings
    st.sidebar.subheader("Auto-refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)
    
    # Main content area
    if page == "Real-time Dashboard":
        render_dashboard(collector, data_storage, dashboard, auto_refresh, refresh_interval)
    
    elif page == "Performance Predictions":
        render_predictions(collector, predictor, data_storage)
    
    elif page == "AI Assistant":
        render_ai_assistant(ai_assistant, data_storage)
    
    elif page == "Alerts & Settings":
        render_alerts_settings(alert_manager, collector)
    
    elif page == "Data Export":
        render_data_export(data_storage)

def render_dashboard(collector, data_storage, dashboard, auto_refresh, refresh_interval):
    """Render the main dashboard page"""
    st.title("📊 Real-time System Performance Dashboard")
    
    # Collect current metrics
    current_metrics = collector.collect_metrics()
    
    # Store metrics
    data_storage.store_metrics(current_metrics)
    
    # Get historical data
    historical_data = data_storage.get_historical_data(hours=24)
    
    # Display current metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = current_metrics['cpu_percent']
        st.metric(
            label="CPU Usage",
            value=f"{cpu_usage:.1f}%",
            delta=f"{cpu_usage - 50:.1f}%" if len(historical_data) > 1 else None
        )
        st.progress(cpu_usage / 100)
    
    with col2:
        memory_usage = current_metrics['memory_percent']
        st.metric(
            label="Memory Usage",
            value=f"{memory_usage:.1f}%",
            delta=f"{memory_usage - 60:.1f}%" if len(historical_data) > 1 else None
        )
        st.progress(memory_usage / 100)
    
    with col3:
        disk_usage = current_metrics['disk_percent']
        st.metric(
            label="Disk Usage",
            value=f"{disk_usage:.1f}%",
            delta=f"{disk_usage - 40:.1f}%" if len(historical_data) > 1 else None
        )
        st.progress(disk_usage / 100)
    
    with col4:
        uptime_hours = current_metrics['uptime'] / 3600
        st.metric(
            label="System Uptime",
            value=f"{uptime_hours:.1f}h",
            delta=None
        )
    
    # Historical charts
    if len(historical_data) > 0:
        st.subheader("📈 Performance Trends")
        
        # Create time series charts
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # CPU Chart
        fig_cpu = px.line(df, x='timestamp', y='cpu_percent', 
                         title='CPU Usage Over Time',
                         labels={'cpu_percent': 'CPU Usage (%)', 'timestamp': 'Time'})
        fig_cpu.update_traces(line_color='#FF6B6B')
        st.plotly_chart(fig_cpu, use_container_width=True)
        
        # Memory Chart
        fig_memory = px.line(df, x='timestamp', y='memory_percent', 
                           title='Memory Usage Over Time',
                           labels={'memory_percent': 'Memory Usage (%)', 'timestamp': 'Time'})
        fig_memory.update_traces(line_color='#4ECDC4')
        st.plotly_chart(fig_memory, use_container_width=True)
        
        # Disk Chart
        fig_disk = px.line(df, x='timestamp', y='disk_percent', 
                          title='Disk Usage Over Time',
                          labels={'disk_percent': 'Disk Usage (%)', 'timestamp': 'Time'})
        fig_disk.update_traces(line_color='#45B7D1')
        st.plotly_chart(fig_disk, use_container_width=True)
    
    # System Information
    st.subheader("🔧 System Information")
    sys_info = collector.get_system_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Hardware Information:**")
        st.write(f"- CPU Cores: {sys_info['cpu_cores']}")
        st.write(f"- CPU Frequency: {sys_info['cpu_freq']:.1f} MHz")
        st.write(f"- Total Memory: {format_bytes(sys_info['total_memory'])}")
    
    with col2:
        st.write("**Storage Information:**")
        st.write(f"- Total Disk Space: {format_bytes(sys_info['total_disk'])}")
        st.write(f"- Available Disk Space: {format_bytes(sys_info['available_disk'])}")
        st.write(f"- Boot Time: {sys_info['boot_time']}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def render_predictions(collector, predictor, data_storage):
    """Render the predictions page"""
    st.title("🔮 Performance Predictions")
    
    # Get historical data for predictions
    historical_data = data_storage.get_historical_data(hours=168)  # 1 week
    
    if len(historical_data) < 10:
        st.warning("⚠️ Insufficient historical data for predictions. Need at least 10 data points.")
        st.info("The system is currently collecting data. Please check back later.")
        return
    
    # Prediction settings
    col1, col2 = st.columns(2)
    with col1:
        prediction_hours = st.slider("Prediction timeframe (hours)", 1, 72, 24)
    with col2:
        metric_to_predict = st.selectbox("Metric to predict", ["cpu_percent", "memory_percent", "disk_percent"])
    
    if st.button("Generate Predictions"):
        with st.spinner("Training model and generating predictions..."):
            try:
                # Prepare data
                df = pd.DataFrame(historical_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Generate predictions
                predictions, model_performance = predictor.predict_future_usage(
                    df, metric_to_predict, prediction_hours
                )
                
                # Display model performance
                st.subheader("📊 Model Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{model_performance.get('r2_score', 0):.3f}")
                with col2:
                    st.metric("MAE", f"{model_performance.get('mae', 0):.3f}")
                with col3:
                    st.metric("RMSE", f"{model_performance.get('rmse', 0):.3f}")
                
                # Plot predictions
                st.subheader(f"🔮 {metric_to_predict.replace('_', ' ').title()} Predictions")
                
                # Combine historical and predicted data
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric_to_predict],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Predicted data
                prediction_timestamps = pd.date_range(
                    start=df['timestamp'].iloc[-1],
                    periods=len(predictions) + 1,
                    freq='5T'
                )[1:]  # Exclude the first timestamp to avoid overlap
                
                fig.add_trace(go.Scatter(
                    x=prediction_timestamps,
                    y=predictions,
                    mode='lines',
                    name='Predictions',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{metric_to_predict.replace("_", " ").title()} - Historical vs Predicted',
                    xaxis_title='Time',
                    yaxis_title='Usage (%)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction insights
                st.subheader("💡 Insights")
                avg_predicted = sum(predictions) / len(predictions)
                max_predicted = max(predictions)
                min_predicted = min(predictions)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Predicted", f"{avg_predicted:.1f}%")
                with col2:
                    st.metric("Maximum Predicted", f"{max_predicted:.1f}%")
                with col3:
                    st.metric("Minimum Predicted", f"{min_predicted:.1f}%")
                
                # Alerts for high predictions
                if max_predicted > 90:
                    st.error(f"⚠️ High usage predicted! Maximum {metric_to_predict.replace('_', ' ')} may reach {max_predicted:.1f}%")
                elif max_predicted > 80:
                    st.warning(f"⚠️ Moderate usage predicted. Maximum {metric_to_predict.replace('_', ' ')} may reach {max_predicted:.1f}%")
                else:
                    st.success(f"✅ Usage levels appear normal. Maximum predicted: {max_predicted:.1f}%")
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

def render_ai_assistant(ai_assistant, data_storage):
    """Render the AI assistant page"""
    st.title("🤖 AI Performance Assistant")
    
    # Get recent data for context
    recent_data = data_storage.get_historical_data(hours=24)
    
    if len(recent_data) == 0:
        st.warning("⚠️ No system data available for analysis.")
        return
    
    # Chat interface
    st.subheader("💬 Ask about your system performance")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask me anything about your system performance...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your system data..."):
                try:
                    response = ai_assistant.get_response(user_question, recent_data)
                    st.write(response)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Suggested questions
    st.subheader("💡 Suggested Questions")
    suggestions = [
        "What's my current system performance status?",
        "Are there any concerning trends in my CPU usage?",
        "How is my memory usage compared to yesterday?",
        "What time of day does my system perform best?",
        "Should I be worried about my disk usage?",
        "Can you summarize my system's performance today?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{suggestion}"):
            # Add suggestion as user message
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            
            # Generate response
            try:
                response = ai_assistant.get_response(suggestion, recent_data)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def render_alerts_settings(alert_manager, collector):
    """Render the alerts and settings page"""
    st.title("⚠️ Alerts & Settings")
    
    # Alert thresholds
    st.subheader("🎚️ Alert Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**CPU Usage Alerts**")
        cpu_warning = st.slider("CPU Warning Threshold (%)", 0, 100, 70)
        cpu_critical = st.slider("CPU Critical Threshold (%)", 0, 100, 90)
    
    with col2:
        st.write("**Memory Usage Alerts**")
        memory_warning = st.slider("Memory Warning Threshold (%)", 0, 100, 75)
        memory_critical = st.slider("Memory Critical Threshold (%)", 0, 100, 90)
    
    with col3:
        st.write("**Disk Usage Alerts**")
        disk_warning = st.slider("Disk Warning Threshold (%)", 0, 100, 80)
        disk_critical = st.slider("Disk Critical Threshold (%)", 0, 100, 95)
    
    # Update thresholds
    thresholds = {
        'cpu_warning': cpu_warning,
        'cpu_critical': cpu_critical,
        'memory_warning': memory_warning,
        'memory_critical': memory_critical,
        'disk_warning': disk_warning,
        'disk_critical': disk_critical
    }
    
    if st.button("💾 Save Alert Settings"):
        alert_manager.update_thresholds(thresholds)
        st.success("✅ Alert thresholds updated successfully!")
    
    # Current alerts
    st.subheader("🚨 Current Alerts")
    
    current_metrics = collector.collect_metrics()
    alerts = alert_manager.check_alerts(current_metrics)
    
    if alerts:
        for alert in alerts:
            if alert['level'] == 'critical':
                st.error(f"🔴 **CRITICAL**: {alert['message']}")
            elif alert['level'] == 'warning':
                st.warning(f"🟡 **WARNING**: {alert['message']}")
    else:
        st.success("✅ No active alerts - all systems normal")
    
    # System settings
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Collection Settings**")
        collection_interval = st.slider("Collection Interval (seconds)", 1, 60, 5)
        enable_detailed_logging = st.checkbox("Enable Detailed Logging", value=False)
    
    with col2:
        st.write("**Storage Settings**")
        max_data_age = st.slider("Maximum Data Age (days)", 1, 30, 7)
        auto_cleanup = st.checkbox("Auto-cleanup Old Data", value=True)
    
    if st.button("💾 Save System Settings"):
        settings = {
            'collection_interval': collection_interval,
            'enable_detailed_logging': enable_detailed_logging,
            'max_data_age': max_data_age,
            'auto_cleanup': auto_cleanup
        }
        st.success("✅ System settings updated successfully!")

def render_data_export(data_storage):
    """Render the data export page"""
    st.title("📤 Data Export")
    
    # Export options
    st.subheader("📊 Export Historical Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last Week", "Last Month", "All Data"])
    
    with col2:
        metrics_to_export = st.multiselect(
            "Metrics to Export",
            ["cpu_percent", "memory_percent", "disk_percent", "uptime", "all"],
            default=["all"]
        )
    
    # Map time range to hours
    time_mapping = {
        "Last Hour": 1,
        "Last 24 Hours": 24,
        "Last Week": 168,
        "Last Month": 720,
        "All Data": None
    }
    
    if st.button("📥 Generate Export"):
        try:
            hours = time_mapping[time_range]
            data = data_storage.get_historical_data(hours=hours) if hours else data_storage.get_all_data()
            
            if not data:
                st.warning("⚠️ No data available for export.")
                return
            
            df = pd.DataFrame(data)
            
            # Filter metrics if specific ones selected
            if "all" not in metrics_to_export:
                columns_to_keep = ["timestamp"] + metrics_to_export
                df = df[columns_to_keep]
            
            # Export based on format
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "JSON":
                json_data = df.to_json(orient="records", date_format="iso")
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            st.info(f"✅ Export ready! Total records: {len(df)}")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
    
    # Data statistics
    st.subheader("📈 Data Statistics")
    
    try:
        all_data = data_storage.get_all_data()
        if all_data:
            df = pd.DataFrame(all_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            
            with col2:
                oldest_record = pd.to_datetime(df['timestamp']).min()
                st.metric("Oldest Record", oldest_record.strftime("%Y-%m-%d"))
            
            with col3:
                newest_record = pd.to_datetime(df['timestamp']).max()
                st.metric("Newest Record", newest_record.strftime("%Y-%m-%d"))
            
            with col4:
                data_size = len(str(df)) / 1024  # Approximate size in KB
                st.metric("Approx. Size", f"{data_size:.1f} KB")
        
        else:
            st.info("📊 No data collected yet.")
            
    except Exception as e:
        st.error(f"❌ Error loading data statistics: {str(e)}")

if __name__ == "__main__":
    main()
