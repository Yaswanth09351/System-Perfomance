# System Performance Monitoring Dashboard

## Overview

This is a comprehensive real-time system performance monitoring dashboard built with Streamlit that provides system metrics collection, machine learning-based performance prediction, and AI-powered assistance for interpreting system behavior. The application monitors CPU, RAM, and disk usage with real-time visualization and predictive analytics capabilities.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit-based web interface with responsive design
- **Data Collection Layer**: psutil-based system metrics collection
- **Machine Learning Layer**: scikit-learn models for performance prediction
- **AI Integration**: Google Gemini API for natural language query processing
- **Storage Layer**: File-based data persistence with in-memory caching
- **Alert System**: Configurable threshold-based alerting

## Key Components

### Core Components

1. **SystemMetricsCollector** (`components/data_collector.py`)
   - Real-time system metrics collection using psutil
   - Collects CPU, memory, disk, network, and process information
   - Provides current system state and uptime tracking

2. **PerformancePredictor** (`components/ml_predictor.py`)
   - Machine learning models for time-series forecasting
   - Supports Linear Regression and Random Forest algorithms
   - Feature engineering with lag variables, rolling averages, and trend analysis
   - Model training, validation, and prediction capabilities

3. **AIAssistant** (`components/ai_assistant.py`)
   - Google Gemini API integration for natural language processing
   - Interprets user queries about system performance
   - Provides contextual responses based on current system data

4. **Dashboard** (`components/dashboard.py`)
   - Visualization components using Plotly
   - Gauge charts, time series plots, and interactive displays
   - Color-coded alerts based on configurable thresholds

5. **AlertManager** (`components/alerts.py`)
   - Configurable threshold management for system alerts
   - JSON-based configuration persistence
   - Alert history tracking and notification system

### Utility Components

1. **DataStorage** (`utils/data_storage.py`)
   - File-based data persistence (CSV and JSON formats)
   - In-memory caching for performance optimization
   - Historical data retrieval and cleanup functionality

2. **Helpers** (`utils/helpers.py`)
   - Utility functions for data formatting
   - Byte conversion, uptime formatting, and color coding
   - Safe mathematical operations and percentage calculations

## Data Flow

1. **Collection**: SystemMetricsCollector gathers real-time system metrics using psutil
2. **Storage**: DataStorage component persists metrics to CSV files and maintains in-memory cache
3. **Processing**: PerformancePredictor processes historical data to train ML models
4. **Prediction**: Trained models generate future performance forecasts
5. **Visualization**: Dashboard component renders interactive charts and gauges
6. **Analysis**: AIAssistant processes user queries and provides insights
7. **Alerting**: AlertManager monitors thresholds and triggers notifications

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualization and charting
- **psutil**: System and process monitoring
- **scikit-learn**: Machine learning algorithms and utilities
- **numpy**: Numerical computing support

### External APIs
- **Google Gemini API**: Natural language processing and AI assistance
  - Requires GEMINI_API_KEY environment variable
  - Used for interpreting user queries about system performance

### Data Storage
- **File-based storage**: CSV and JSON files for persistence
- **In-memory caching**: Recent metrics stored in memory for performance
- **Configuration files**: JSON-based alert threshold configuration

## Deployment Strategy

The application is designed for local deployment with the following considerations:

1. **Environment Setup**:
   - Python 3.7+ required
   - Install dependencies via pip
   - Configure GEMINI_API_KEY environment variable

2. **Data Directory**:
   - Creates local 'data' directory for metric storage
   - Automatic cleanup of old data to manage disk usage

3. **Configuration**:
   - JSON-based configuration for alert thresholds
   - Persistent settings across application restarts

4. **Scalability**:
   - In-memory caching limits to prevent memory overflow
   - File-based storage for long-term data retention
   - Modular architecture allows for easy component replacement

## Changelog

```
Changelog:
- July 01, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```