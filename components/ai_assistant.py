import os
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import json
from datetime import datetime, timedelta

class AIAssistant:
    """AI assistant for interpreting system performance metrics using Gemini API"""
    
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "demo_key"))
        self.model_name = "gemini-2.5-flash"
    
    def get_response(self, user_question, system_data):
        """Get AI response to user question about system performance"""
        try:
            # Prepare system data summary
            data_summary = self._prepare_data_summary(system_data)
            
            # Create comprehensive system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(user_question, data_summary)
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=user_prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            return response.text if response.text else "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            return f"I encountered an error while analyzing your system data: {str(e)}. Please ensure your Gemini API key is properly configured."
    
    def _prepare_data_summary(self, system_data):
        """Prepare a summary of system data for the AI"""
        try:
            if not system_data:
                return "No system data available."
            
            df = pd.DataFrame(system_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Get latest metrics
            latest = df.iloc[-1] if len(df) > 0 else {}
            
            # Calculate statistics
            stats = {
                'cpu': {
                    'current': latest.get('cpu_percent', 0),
                    'average': df['cpu_percent'].mean() if 'cpu_percent' in df else 0,
                    'max': df['cpu_percent'].max() if 'cpu_percent' in df else 0,
                    'min': df['cpu_percent'].min() if 'cpu_percent' in df else 0
                },
                'memory': {
                    'current': latest.get('memory_percent', 0),
                    'average': df['memory_percent'].mean() if 'memory_percent' in df else 0,
                    'max': df['memory_percent'].max() if 'memory_percent' in df else 0,
                    'min': df['memory_percent'].min() if 'memory_percent' in df else 0
                },
                'disk': {
                    'current': latest.get('disk_percent', 0),
                    'average': df['disk_percent'].mean() if 'disk_percent' in df else 0,
                    'max': df['disk_percent'].max() if 'disk_percent' in df else 0,
                    'min': df['disk_percent'].min() if 'disk_percent' in df else 0
                }
            }
            
            # Calculate trends
            trends = {}
            for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if metric in df.columns and len(df) > 1:
                    values = df[metric].values
                    trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                    trends[metric] = {
                        'slope': trend_slope,
                        'direction': 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable'
                    }
            
            # Time range info
            time_range = {
                'start': df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S') if len(df) > 0 else 'Unknown',
                'end': df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') if len(df) > 0 else 'Unknown',
                'duration_hours': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600 if len(df) > 1 else 0,
                'data_points': len(df)
            }
            
            summary = {
                'current_metrics': {
                    'cpu_percent': round(latest.get('cpu_percent', 0), 1),
                    'memory_percent': round(latest.get('memory_percent', 0), 1),
                    'disk_percent': round(latest.get('disk_percent', 0), 1),
                    'uptime_hours': round(latest.get('uptime', 0) / 3600, 1) if latest.get('uptime') else 0
                },
                'statistics': stats,
                'trends': trends,
                'time_range': time_range
            }
            
            return json.dumps(summary, indent=2)
            
        except Exception as e:
            return f"Error preparing data summary: {str(e)}"
    
    def _create_system_prompt(self):
        """Create system prompt for the AI assistant"""
        return """You are an expert system performance analyst and monitoring specialist. Your role is to:

1. Analyze system performance data (CPU, memory, disk usage) and provide insights
2. Identify potential issues, bottlenecks, or concerning trends
3. Offer practical recommendations for system optimization
4. Explain technical concepts in an accessible way
5. Provide actionable advice based on the data

Guidelines for your responses:
- Be concise but thorough in your analysis
- Use percentages and specific numbers from the data when relevant
- Highlight any concerning metrics (>80% usage typically warrants attention)
- Suggest practical solutions when issues are identified
- Use emojis appropriately to make responses more engaging
- If asked about trends, focus on the data patterns and their implications
- Always base your analysis on the actual data provided

Key thresholds to consider:
- CPU: >70% warning, >90% critical
- Memory: >75% warning, >90% critical  
- Disk: >80% warning, >95% critical

Remember: You're helping users understand and optimize their system performance."""
    
    def _create_user_prompt(self, user_question, data_summary):
        """Create user prompt with question and system data context"""
        return f"""User Question: {user_question}

Current System Data Summary:
{data_summary}

Please analyze the system data and provide a helpful response to the user's question. Base your answer on the actual metrics and trends shown in the data."""
    
    def analyze_performance_trend(self, system_data, metric='cpu_percent', hours=24):
        """Analyze performance trends for a specific metric"""
        try:
            prompt = f"""Analyze the {metric} performance trend over the last {hours} hours based on this system data:

{self._prepare_data_summary(system_data)}

Please provide:
1. Current status assessment
2. Trend analysis (increasing/decreasing/stable)
3. Any concerning patterns
4. Recommendations if needed"""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800
                )
            )
            
            return response.text if response.text else "Unable to analyze trend."
            
        except Exception as e:
            return f"Error analyzing trend: {str(e)}"
    
    def get_optimization_suggestions(self, system_data):
        """Get AI-powered optimization suggestions"""
        try:
            data_summary = self._prepare_data_summary(system_data)
            
            prompt = f"""Based on this system performance data, provide specific optimization recommendations:

{data_summary}

Please suggest:
1. Immediate actions for any critical issues
2. Performance optimization opportunities
3. Preventive measures for identified trends
4. Resource allocation recommendations

Focus on practical, actionable advice."""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            return response.text if response.text else "Unable to generate optimization suggestions."
            
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"
    
    def explain_metric(self, metric_name, current_value, context_data=None):
        """Explain what a specific metric means and its implications"""
        try:
            context = ""
            if context_data:
                context = f"\nCurrent system context:\n{self._prepare_data_summary(context_data)}"
            
            prompt = f"""Explain the system metric "{metric_name}" with current value {current_value}%:

1. What this metric measures
2. What the current value indicates
3. Normal vs concerning ranges
4. Potential causes if the value is problematic
5. Recommended actions if needed

{context}

Keep the explanation clear and practical."""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=600
                )
            )
            
            return response.text if response.text else f"Unable to explain {metric_name}."
            
        except Exception as e:
            return f"Error explaining metric: {str(e)}"
    
    def get_health_summary(self, system_data):
        """Get an overall system health summary"""
        try:
            data_summary = self._prepare_data_summary(system_data)
            
            prompt = f"""Provide a comprehensive system health summary based on this data:

{data_summary}

Include:
1. Overall health status (Excellent/Good/Fair/Poor/Critical)
2. Key strengths of the current system performance
3. Areas of concern or potential issues
4. Priority recommendations
5. Short-term and long-term outlook

Use a clear, executive summary style."""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=800
                )
            )
            
            return response.text if response.text else "Unable to generate health summary."
            
        except Exception as e:
            return f"Error generating health summary: {str(e)}"
