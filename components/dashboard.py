import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class Dashboard:
    """Dashboard component for rendering various visualizations"""
    
    def __init__(self):
        self.color_scheme = {
            'cpu': '#FF6B6B',
            'memory': '#4ECDC4', 
            'disk': '#45B7D1',
            'network': '#96CEB4',
            'warning': '#FFE66D',
            'critical': '#FF6B6B',
            'good': '#4ECDC4'
        }
    
    def render_metric_gauge(self, value, title, max_value=100, thresholds=None):
        """Render a gauge chart for a metric"""
        if thresholds is None:
            thresholds = {'warning': 70, 'critical': 90}
        
        # Determine color based on thresholds
        if value >= thresholds['critical']:
            color = self.color_scheme['critical']
        elif value >= thresholds['warning']:
            color = self.color_scheme['warning']
        else:
            color = self.color_scheme['good']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': thresholds['warning']},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, thresholds['warning']], 'color': "lightgray"},
                    {'range': [thresholds['warning'], thresholds['critical']], 'color': "yellow"},
                    {'range': [thresholds['critical'], max_value], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds['critical']
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    def render_time_series_chart(self, df, metrics, title="System Metrics Over Time"):
        """Render a time series chart for multiple metrics"""
        fig = go.Figure()
        
        colors = [self.color_scheme['cpu'], self.color_scheme['memory'], 
                 self.color_scheme['disk'], self.color_scheme['network']]
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{metric.replace("_", " ").title()}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Value: %{y:.1f}%<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            hovermode='x unified',
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def render_cpu_core_chart(self, core_usages):
        """Render CPU usage per core"""
        if not core_usages:
            return None
        
        cores = [f"Core {i}" for i in range(len(core_usages))]
        
        fig = go.Figure(data=[
            go.Bar(
                x=cores,
                y=core_usages,
                marker_color=[self.color_scheme['critical'] if usage > 80 
                            else self.color_scheme['warning'] if usage > 60 
                            else self.color_scheme['good'] for usage in core_usages],
                text=[f"{usage:.1f}%" for usage in core_usages],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="CPU Usage per Core",
            xaxis_title="CPU Cores",
            yaxis_title="Usage (%)",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def render_process_table(self, processes):
        """Render top processes table"""
        if not processes:
            return pd.DataFrame()
        
        df = pd.DataFrame(processes)
        df = df.round(2)
        return df
    
    def render_system_health_card(self, metrics):
        """Render system health overview card"""
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory_percent', 0)
        disk = metrics.get('disk_percent', 0)
        
        # Calculate overall health score
        health_score = 100 - max(cpu, memory, disk)
        
        if health_score >= 80:
            status = "Excellent"
            color = self.color_scheme['good']
            icon = "✅"
        elif health_score >= 60:
            status = "Good"
            color = self.color_scheme['good']
            icon = "👍"
        elif health_score >= 40:
            status = "Fair"
            color = self.color_scheme['warning']
            icon = "⚠️"
        elif health_score >= 20:
            status = "Poor"
            color = self.color_scheme['critical']
            icon = "❌"
        else:
            status = "Critical"
            color = self.color_scheme['critical']
            icon = "🚨"
        
        return {
            'status': status,
            'score': health_score,
            'color': color,
            'icon': icon,
            'details': {
                'cpu': cpu,
                'memory': memory,
                'disk': disk
            }
        }
    
    def render_resource_distribution_pie(self, metrics):
        """Render resource usage distribution pie chart"""
        labels = ['CPU Usage', 'Memory Usage', 'Disk Usage']
        values = [
            metrics.get('cpu_percent', 0),
            metrics.get('memory_percent', 0),
            metrics.get('disk_percent', 0)
        ]
        
        colors = [self.color_scheme['cpu'], self.color_scheme['memory'], self.color_scheme['disk']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Current Resource Usage Distribution",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def render_trend_indicators(self, historical_data, metric='cpu_percent'):
        """Render trend indicators for a metric"""
        if len(historical_data) < 2:
            return None
        
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate trend
        values = df[metric].values
        trend_slope = np.polyfit(range(len(values)), values, 1)[0]
        
        # Determine trend direction
        if trend_slope > 0.1:
            trend = "Increasing"
            icon = "📈"
            color = "red" if trend_slope > 0.5 else "orange"
        elif trend_slope < -0.1:
            trend = "Decreasing"
            icon = "📉"
            color = "green"
        else:
            trend = "Stable"
            icon = "➡️"
            color = "blue"
        
        return {
            'trend': trend,
            'slope': trend_slope,
            'icon': icon,
            'color': color,
            'change_rate': f"{abs(trend_slope):.2f}% per hour"
        }
    
    def render_alert_timeline(self, alerts):
        """Render timeline of alerts"""
        if not alerts:
            return None
        
        df = pd.DataFrame(alerts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create timeline chart
        fig = go.Figure()
        
        for level in ['critical', 'warning', 'info']:
            level_alerts = df[df['level'] == level]
            if not level_alerts.empty:
                color = self.color_scheme['critical'] if level == 'critical' else \
                       self.color_scheme['warning'] if level == 'warning' else \
                       self.color_scheme['good']
                
                fig.add_trace(go.Scatter(
                    x=level_alerts['timestamp'],
                    y=[level] * len(level_alerts),
                    mode='markers',
                    name=level.title(),
                    marker=dict(size=12, color=color),
                    text=level_alerts['message'],
                    hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Message: %{text}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Alert Timeline",
            xaxis_title="Time",
            yaxis_title="Alert Level",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def render_network_traffic_chart(self, df):
        """Render network traffic chart"""
        if 'network_bytes_sent' not in df.columns or 'network_bytes_recv' not in df.columns:
            return None
        
        fig = go.Figure()
        
        # Convert bytes to MB/s (assuming 5-minute intervals)
        interval_minutes = 5
        df['network_sent_mbps'] = df['network_bytes_sent'].diff() / (1024 * 1024 * interval_minutes * 60)
        df['network_recv_mbps'] = df['network_bytes_recv'].diff() / (1024 * 1024 * interval_minutes * 60)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['network_sent_mbps'],
            mode='lines',
            name='Sent (MB/s)',
            line=dict(color='#FF6B6B')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['network_recv_mbps'],
            mode='lines',
            name='Received (MB/s)',
            line=dict(color='#4ECDC4')
        ))
        
        fig.update_layout(
            title="Network Traffic",
            xaxis_title="Time",
            yaxis_title="Traffic (MB/s)",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def render_comparison_chart(self, current_metrics, historical_avg):
        """Render current vs average comparison"""
        metrics = ['CPU', 'Memory', 'Disk']
        current = [
            current_metrics.get('cpu_percent', 0),
            current_metrics.get('memory_percent', 0),
            current_metrics.get('disk_percent', 0)
        ]
        average = [
            historical_avg.get('cpu_percent', 0),
            historical_avg.get('memory_percent', 0),
            historical_avg.get('disk_percent', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(name='Current', x=metrics, y=current, marker_color='#FF6B6B'),
            go.Bar(name='24h Average', x=metrics, y=average, marker_color='#4ECDC4')
        ])
        
        fig.update_layout(
            title='Current vs 24h Average Usage',
            xaxis_title='Metrics',
            yaxis_title='Usage (%)',
            barmode='group',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
