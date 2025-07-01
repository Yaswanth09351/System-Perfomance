import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformancePredictor:
    """Machine learning model for predicting system performance trends"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scalers = {}
        self.trained_models = {}
        self.feature_columns = []
    
    def prepare_features(self, df, target_column):
        """Prepare features for machine learning"""
        try:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Create time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['minute'] = df['timestamp'].dt.minute
            
            # Create lag features (previous values)
            for lag in [1, 2, 3, 5, 10]:
                df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
            # Create rolling averages
            for window in [5, 10, 20]:
                df[f'{target_column}_rolling_{window}'] = df[target_column].rolling(window=window).mean()
            
            # Create trend features
            df[f'{target_column}_diff'] = df[target_column].diff()
            df[f'{target_column}_change_rate'] = df[target_column].pct_change()
            
            # Drop rows with NaN values (due to lag and rolling features)
            df = df.dropna()
            
            if len(df) < 5:
                raise ValueError("Insufficient data after feature engineering")
            
            # Define feature columns (exclude timestamp and target)
            self.feature_columns = [col for col in df.columns 
                                  if col not in ['timestamp', target_column]]
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in feature preparation: {str(e)}")
    
    def train_model(self, df, target_column, model_type='random_forest'):
        """Train a machine learning model"""
        try:
            # Prepare features
            df_features = self.prepare_features(df.copy(), target_column)
            
            if len(df_features) < 10:
                raise ValueError("Need at least 10 data points for training")
            
            # Prepare X and y
            X = df_features[self.feature_columns]
            y = df_features[target_column]
            
            # Split data
            test_size = min(0.2, max(0.1, len(df_features) * 0.2))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.models[model_type]
            model.fit(X_train_scaled, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            performance = {
                'r2_score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'model_type': model_type
            }
            
            # Store trained model and scaler
            self.trained_models[target_column] = model
            self.scalers[target_column] = scaler
            
            return performance
            
        except Exception as e:
            raise Exception(f"Error in model training: {str(e)}")
    
    def predict_future_usage(self, df, target_column, hours_ahead=24, model_type='random_forest'):
        """Predict future system usage"""
        try:
            # Train the model first
            performance = self.train_model(df, target_column, model_type)
            
            # Prepare the most recent data for prediction
            df_prepared = self.prepare_features(df.copy(), target_column)
            
            if len(df_prepared) == 0:
                raise ValueError("No data available after preparation")
            
            # Get the most recent data point
            last_row = df_prepared.iloc[-1:].copy()
            
            # Generate predictions
            predictions = []
            current_data = last_row.copy()
            
            # Number of predictions (assuming 5-minute intervals)
            num_predictions = max(1, hours_ahead * 12)  # 12 predictions per hour (5-min intervals)
            
            model = self.trained_models[target_column]
            scaler = self.scalers[target_column]
            
            for i in range(num_predictions):
                # Prepare features for prediction
                X_pred = current_data[self.feature_columns]
                X_pred_scaled = scaler.transform(X_pred)
                
                # Make prediction
                pred = model.predict(X_pred_scaled)[0]
                
                # Ensure prediction is within reasonable bounds
                pred = max(0, min(100, pred))
                predictions.append(pred)
                
                # Update current_data for next prediction
                # This is a simplified approach - in practice, you might want more sophisticated updating
                current_data[target_column] = pred
                
                # Update lag features
                for lag in [1, 2, 3, 5, 10]:
                    if f'{target_column}_lag_{lag}' in current_data.columns:
                        if lag == 1:
                            current_data[f'{target_column}_lag_{lag}'] = pred
                        else:
                            # Shift lag features
                            prev_lag = f'{target_column}_lag_{lag-1}'
                            if prev_lag in current_data.columns:
                                current_data[f'{target_column}_lag_{lag}'] = current_data[prev_lag].iloc[0]
                
                # Update time features (increment by 5 minutes)
                if i == 0:
                    base_time = pd.to_datetime(df['timestamp'].iloc[-1]) + timedelta(minutes=5)
                else:
                    base_time = base_time + timedelta(minutes=5)
                
                current_data['hour'] = base_time.hour
                current_data['minute'] = base_time.minute
                current_data['day_of_week'] = base_time.dayofweek
            
            return predictions, performance
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")
    
    def detect_anomalies(self, df, target_column, threshold_std=2):
        """Detect anomalies in system metrics"""
        try:
            values = df[target_column].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Calculate z-scores
            z_scores = np.abs((values - mean_val) / std_val) if std_val > 0 else np.zeros_like(values)
            
            # Find anomalies
            anomalies = []
            for i, (timestamp, value, z_score) in enumerate(zip(df['timestamp'], values, z_scores)):
                if z_score > threshold_std:
                    anomalies.append({
                        'timestamp': timestamp,
                        'value': value,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
            
            return anomalies
            
        except Exception as e:
            raise Exception(f"Error in anomaly detection: {str(e)}")
    
    def get_trend_analysis(self, df, target_column):
        """Analyze trends in system metrics"""
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate trend over different time windows
            trends = {}
            
            # Recent trend (last 10% of data)
            recent_data = df.tail(max(5, len(df) // 10))
            if len(recent_data) > 1:
                recent_slope = np.polyfit(range(len(recent_data)), recent_data[target_column], 1)[0]
                trends['recent'] = {
                    'slope': recent_slope,
                    'direction': 'increasing' if recent_slope > 0.1 else 'decreasing' if recent_slope < -0.1 else 'stable'
                }
            
            # Overall trend
            if len(df) > 1:
                overall_slope = np.polyfit(range(len(df)), df[target_column], 1)[0]
                trends['overall'] = {
                    'slope': overall_slope,
                    'direction': 'increasing' if overall_slope > 0.1 else 'decreasing' if overall_slope < -0.1 else 'stable'
                }
            
            # Statistical summary
            trends['statistics'] = {
                'mean': df[target_column].mean(),
                'std': df[target_column].std(),
                'min': df[target_column].min(),
                'max': df[target_column].max(),
                'current': df[target_column].iloc[-1] if len(df) > 0 else 0
            }
            
            return trends
            
        except Exception as e:
            raise Exception(f"Error in trend analysis: {str(e)}")
    
    def get_model_insights(self, target_column):
        """Get insights about the trained model"""
        try:
            if target_column not in self.trained_models:
                return {"error": "Model not trained for this metric"}
            
            model = self.trained_models[target_column]
            insights = {
                'model_type': type(model).__name__,
                'feature_count': len(self.feature_columns),
                'features': self.feature_columns
            }
            
            # Add feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
                insights['feature_importance'] = dict(sorted(feature_importance.items(), 
                                                           key=lambda x: x[1], reverse=True))
            
            return insights
            
        except Exception as e:
            return {"error": f"Error getting model insights: {str(e)}"}
