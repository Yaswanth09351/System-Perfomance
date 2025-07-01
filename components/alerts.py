import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class AlertManager:
    """Manages system alerts and thresholds"""
    
    def __init__(self, config_file="alert_config.json"):
        self.config_file = config_file
        self.default_thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 90,
            'memory_warning': 75,
            'memory_critical': 90,
            'disk_warning': 80,
            'disk_critical': 95
        }
        self.thresholds = self.load_thresholds()
        self.alert_history = []
    
    def load_thresholds(self) -> Dict[str, float]:
        """Load alert thresholds from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('thresholds', self.default_thresholds)
            return self.default_thresholds
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            return self.default_thresholds
    
    def save_thresholds(self) -> bool:
        """Save current thresholds to config file"""
        try:
            config = {'thresholds': self.thresholds}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving thresholds: {e}")
            return False
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """Update alert thresholds"""
        try:
            self.thresholds.update(new_thresholds)
            return self.save_thresholds()
        except Exception as e:
            print(f"Error updating thresholds: {e}")
            return False
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check current metrics against thresholds and generate alerts"""
        alerts = []
        current_time = datetime.now()
        
        # Check CPU alerts
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append({
                'timestamp': current_time.isoformat(),
                'level': 'critical',
                'metric': 'cpu_percent',
                'value': cpu_percent,
                'threshold': self.thresholds['cpu_critical'],
                'message': f"Critical: CPU usage at {cpu_percent:.1f}% (threshold: {self.thresholds['cpu_critical']}%)",
                'severity': 'high'
            })
        elif cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append({
                'timestamp': current_time.isoformat(),
                'level': 'warning',
                'metric': 'cpu_percent',
                'value': cpu_percent,
                'threshold': self.thresholds['cpu_warning'],
                'message': f"Warning: CPU usage at {cpu_percent:.1f}% (threshold: {self.thresholds['cpu_warning']}%)",
                'severity': 'medium'
            })
        
        # Check Memory alerts
        memory_percent = metrics.get('memory_percent', 0)
        if memory_percent >= self.thresholds['memory_critical']:
            alerts.append({
                'timestamp': current_time.isoformat(),
                'level': 'critical',
                'metric': 'memory_percent',
                'value': memory_percent,
                'threshold': self.thresholds['memory_critical'],
                'message': f"Critical: Memory usage at {memory_percent:.1f}% (threshold: {self.thresholds['memory_critical']}%)",
                'severity': 'high'
            })
        elif memory_percent >= self.thresholds['memory_warning']:
            alerts.append({
                'timestamp': current_time.isoformat(),
                'level': 'warning',
                'metric': 'memory_percent',
                'value': memory_percent,
                'threshold': self.thresholds['memory_warning'],
                'message': f"Warning: Memory usage at {memory_percent:.1f}% (threshold: {self.thresholds['memory_warning']}%)",
                'severity': 'medium'
            })
        
        # Check Disk alerts
        disk_percent = metrics.get('disk_percent', 0)
        if disk_percent >= self.thresholds['disk_critical']:
            alerts.append({
                'timestamp': current_time.isoformat(),
                'level': 'critical',
                'metric': 'disk_percent',
                'value': disk_percent,
                'threshold': self.thresholds['disk_critical'],
                'message': f"Critical: Disk usage at {disk_percent:.1f}% (threshold: {self.thresholds['disk_critical']}%)",
                'severity': 'high'
            })
        elif disk_percent >= self.thresholds['disk_warning']:
            alerts.append({
                'timestamp': current_time.isoformat(),
                'level': 'warning',
                'metric': 'disk_percent',
                'value': disk_percent,
                'threshold': self.thresholds['disk_warning'],
                'message': f"Warning: Disk usage at {disk_percent:.1f}% (threshold: {self.thresholds['disk_warning']}%)",
                'severity': 'medium'
            })
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.alert_history = [alert for alert in self.alert_history 
                            if datetime.fromisoformat(alert['timestamp']) > cutoff_time]
        
        return alerts
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for the specified number of hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history 
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        recent_alerts = self.get_alert_history()
        
        summary = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a['level'] == 'critical']),
            'warning_alerts': len([a for a in recent_alerts if a['level'] == 'warning']),
            'by_metric': {},
            'latest_alert': None
        }
        
        # Group by metric
        for alert in recent_alerts:
            metric = alert['metric']
            if metric not in summary['by_metric']:
                summary['by_metric'][metric] = {'count': 0, 'latest': None}
            summary['by_metric'][metric]['count'] += 1
            if (summary['by_metric'][metric]['latest'] is None or 
                alert['timestamp'] > summary['by_metric'][metric]['latest']['timestamp']):
                summary['by_metric'][metric]['latest'] = alert
        
        # Get latest alert overall
        if recent_alerts:
            summary['latest_alert'] = max(recent_alerts, key=lambda x: x['timestamp'])
        
        return summary
    
    def clear_alert_history(self) -> bool:
        """Clear all alert history"""
        try:
            self.alert_history = []
            return True
        except Exception as e:
            print(f"Error clearing alert history: {e}")
            return False
    
    def get_threshold_recommendations(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze historical data and recommend threshold adjustments"""
        if not historical_data:
            return {}
        
        try:
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame(historical_data)
            recommendations = {}
            
            for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if metric in df.columns:
                    values = df[metric].values
                    
                    # Calculate statistics
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    p95 = np.percentile(values, 95)
                    p99 = np.percentile(values, 99)
                    
                    # Recommend thresholds based on data distribution
                    base_name = metric.replace('_percent', '')
                    
                    # Warning threshold: mean + 1.5*std or 85th percentile, whichever is lower
                    warning_threshold = min(mean_val + 1.5 * std_val, np.percentile(values, 85))
                    warning_threshold = max(50, min(warning_threshold, 85))  # Bounded between 50-85%
                    
                    # Critical threshold: mean + 2*std or 95th percentile, whichever is lower
                    critical_threshold = min(mean_val + 2 * std_val, p95)
                    critical_threshold = max(warning_threshold + 10, min(critical_threshold, 95))  # At least 10% above warning
                    
                    recommendations[f'{base_name}_warning'] = round(warning_threshold, 1)
                    recommendations[f'{base_name}_critical'] = round(critical_threshold, 1)
            
            return recommendations
            
        except Exception as e:
            print(f"Error calculating threshold recommendations: {e}")
            return {}
    
    def create_alert_rule(self, name: str, condition: str, message: str) -> bool:
        """Create a custom alert rule (future enhancement)"""
        # Placeholder for custom alert rules
        # This could be extended to support complex conditions
        pass
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        recent_alerts = self.get_alert_history()
        
        if not recent_alerts:
            return {
                'total': 0,
                'by_level': {},
                'by_metric': {},
                'by_hour': {},
                'average_per_day': 0,
                'most_frequent_metric': None
            }
        
        try:
            # Count by level
            by_level = {}
            for alert in recent_alerts:
                level = alert['level']
                by_level[level] = by_level.get(level, 0) + 1
            
            # Count by metric
            by_metric = {}
            for alert in recent_alerts:
                metric = alert['metric']
                by_metric[metric] = by_metric.get(metric, 0) + 1
            
            # Count by hour
            by_hour = {}
            for alert in recent_alerts:
                hour = datetime.fromisoformat(alert['timestamp']).hour
                by_hour[hour] = by_hour.get(hour, 0) + 1
            
            # Calculate average per day
            if recent_alerts:
                first_alert_time = datetime.fromisoformat(min(recent_alerts, key=lambda x: x['timestamp'])['timestamp'])
                last_alert_time = datetime.fromisoformat(max(recent_alerts, key=lambda x: x['timestamp'])['timestamp'])
                days_span = max(1, (last_alert_time - first_alert_time).days + 1)
                average_per_day = len(recent_alerts) / days_span
            else:
                average_per_day = 0
            
            # Most frequent metric
            most_frequent_metric = max(by_metric.items(), key=lambda x: x[1])[0] if by_metric else None
            
            return {
                'total': len(recent_alerts),
                'by_level': by_level,
                'by_metric': by_metric,
                'by_hour': by_hour,
                'average_per_day': round(average_per_day, 2),
                'most_frequent_metric': most_frequent_metric
            }
            
        except Exception as e:
            print(f"Error calculating alert statistics: {e}")
            return {'error': str(e)}
