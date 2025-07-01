import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

def format_bytes(bytes_value: Union[int, float]) -> str:
    """Convert bytes to human readable format"""
    if bytes_value == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    size_index = int(math.floor(math.log(bytes_value, 1024)))
    
    if size_index >= len(size_names):
        size_index = len(size_names) - 1
    
    power = math.pow(1024, size_index)
    size = round(bytes_value / power, 2)
    
    return f"{size} {size_names[size_index]}"

def format_uptime(seconds: Union[int, float]) -> str:
    """Convert seconds to human readable uptime format"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"

def get_color_for_usage(usage_percent: float) -> str:
    """Get color code based on usage percentage"""
    if usage_percent >= 90:
        return "#FF4444"  # Red - Critical
    elif usage_percent >= 70:
        return "#FFA500"  # Orange - Warning
    elif usage_percent >= 50:
        return "#FFFF00"  # Yellow - Caution
    else:
        return "#00FF00"  # Green - Good

def calculate_percentage(value: Union[int, float], total: Union[int, float]) -> float:
    """Calculate percentage with safe division"""
    if total == 0:
        return 0.0
    return round((value / total) * 100, 2)

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """Format timestamp to readable string"""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def get_time_ago(timestamp: Union[str, datetime]) -> str:
    """Get human readable time difference from now"""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return "Unknown"
    
    now = datetime.now()
    if timestamp.tzinfo:
        # Make now timezone aware if timestamp is
        from datetime import timezone
        now = now.replace(tzinfo=timezone.utc)
    
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

def validate_threshold(value: Union[int, float], min_val: Union[int, float] = 0, max_val: Union[int, float] = 100) -> bool:
    """Validate if a threshold value is within acceptable range"""
    try:
        return min_val <= float(value) <= max_val
    except (ValueError, TypeError):
        return False

def calculate_trend(values: List[Union[int, float]]) -> Dict[str, Any]:
    """Calculate trend information from a list of values"""
    if len(values) < 2:
        return {
            'direction': 'unknown',
            'slope': 0,
            'strength': 'none',
            'description': 'Insufficient data for trend analysis'
        }
    
    try:
        import numpy as np
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine direction
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        # Determine strength
        abs_slope = abs(slope)
        if abs_slope > 2:
            strength = 'strong'
        elif abs_slope > 0.5:
            strength = 'moderate'
        elif abs_slope > 0.1:
            strength = 'weak'
        else:
            strength = 'stable'
        
        # Create description
        if direction == 'stable':
            description = 'Values are relatively stable'
        else:
            description = f'Values are {direction} with {strength} trend'
        
        return {
            'direction': direction,
            'slope': round(slope, 3),
            'strength': strength,
            'description': description
        }
        
    except ImportError:
        # Fallback calculation without numpy
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        diff = avg_second - avg_first
        
        if diff > 2:
            return {
                'direction': 'increasing',
                'slope': diff / len(values),
                'strength': 'moderate',
                'description': 'Values appear to be increasing'
            }
        elif diff < -2:
            return {
                'direction': 'decreasing',
                'slope': diff / len(values),
                'strength': 'moderate',
                'description': 'Values appear to be decreasing'
            }
        else:
            return {
                'direction': 'stable',
                'slope': 0,
                'strength': 'stable',
                'description': 'Values appear to be stable'
            }

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    # Remove or replace characters that are not safe for filenames
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
    return sanitized[:255]  # Limit length to 255 characters

def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """Format number with appropriate decimal places and thousand separators"""
    try:
        if isinstance(number, int) or number.is_integer():
            return f"{int(number):,}"
        else:
            return f"{number:,.{decimals}f}"
    except (ValueError, AttributeError):
        return str(number)

def get_severity_color(severity: str) -> str:
    """Get color code for severity levels"""
    severity_colors = {
        'low': '#00FF00',      # Green
        'medium': '#FFA500',   # Orange
        'high': '#FF4444',     # Red
        'critical': '#8B0000', # Dark Red
        'info': '#4169E1',     # Royal Blue
        'warning': '#FFA500',  # Orange
        'error': '#FF4444',    # Red
        'success': '#00FF00'   # Green
    }
    
    return severity_colors.get(severity.lower(), '#808080')  # Default to gray

def calculate_moving_average(values: List[Union[int, float]], window: int = 5) -> List[float]:
    """Calculate moving average for a list of values"""
    if len(values) < window:
        return values
    
    moving_averages = []
    for i in range(len(values) - window + 1):
        window_values = values[i:i + window]
        avg = sum(window_values) / window
        moving_averages.append(round(avg, 2))
    
    return moving_averages

def detect_spikes(values: List[Union[int, float]], threshold_multiplier: float = 2.0) -> List[Dict[str, Any]]:
    """Detect unusual spikes in data"""
    if len(values) < 3:
        return []
    
    try:
        import numpy as np
        
        # Calculate mean and standard deviation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        spikes = []
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
            
            if z_score > threshold_multiplier:
                spikes.append({
                    'index': i,
                    'value': value,
                    'z_score': round(z_score, 2),
                    'deviation': round(value - mean_val, 2),
                    'severity': 'high' if z_score > 3 else 'medium'
                })
        
        return spikes
        
    except ImportError:
        # Fallback without numpy
        if len(values) < 5:
            return []
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance)
        
        spikes = []
        for i, value in enumerate(values):
            if std_val > 0 and abs(value - mean_val) > threshold_multiplier * std_val:
                spikes.append({
                    'index': i,
                    'value': value,
                    'deviation': round(value - mean_val, 2),
                    'severity': 'medium'
                })
        
        return spikes

def merge_time_series_data(data1: List[Dict], data2: List[Dict], key: str = 'timestamp') -> List[Dict]:
    """Merge two time series datasets on a common key"""
    merged = {}
    
    # Add data from first dataset
    for record in data1:
        timestamp = record.get(key)
        if timestamp:
            merged[timestamp] = record.copy()
    
    # Merge data from second dataset
    for record in data2:
        timestamp = record.get(key)
        if timestamp:
            if timestamp in merged:
                merged[timestamp].update(record)
            else:
                merged[timestamp] = record.copy()
    
    # Convert back to list and sort by timestamp
    result = list(merged.values())
    result.sort(key=lambda x: x.get(key, ''))
    
    return result

def create_time_buckets(start_time: datetime, end_time: datetime, bucket_size_minutes: int = 5) -> List[datetime]:
    """Create time buckets for data aggregation"""
    buckets = []
    current_time = start_time
    
    while current_time <= end_time:
        buckets.append(current_time)
        current_time += timedelta(minutes=bucket_size_minutes)
    
    return buckets

def aggregate_metrics_by_time(data: List[Dict], bucket_size_minutes: int = 5) -> List[Dict]:
    """Aggregate metrics data into time buckets"""
    if not data:
        return []
    
    try:
        # Convert timestamps and sort
        for record in data:
            if isinstance(record.get('timestamp'), str):
                record['timestamp'] = datetime.fromisoformat(record['timestamp'])
        
        data.sort(key=lambda x: x['timestamp'])
        
        # Create buckets
        start_time = data[0]['timestamp']
        end_time = data[-1]['timestamp']
        buckets = create_time_buckets(start_time, end_time, bucket_size_minutes)
        
        aggregated = []
        bucket_delta = timedelta(minutes=bucket_size_minutes)
        
        for bucket_time in buckets:
            bucket_end = bucket_time + bucket_delta
            
            # Find data points in this bucket
            bucket_data = [
                record for record in data
                if bucket_time <= record['timestamp'] < bucket_end
            ]
            
            if bucket_data:
                # Calculate averages for numeric fields
                aggregated_record = {'timestamp': bucket_time.isoformat()}
                
                numeric_fields = ['cpu_percent', 'memory_percent', 'disk_percent', 'uptime']
                for field in numeric_fields:
                    values = [record.get(field, 0) for record in bucket_data if record.get(field) is not None]
                    if values:
                        aggregated_record[field] = round(sum(values) / len(values), 2)
                
                aggregated.append(aggregated_record)
        
        return aggregated
        
    except Exception as e:
        print(f"Error aggregating metrics: {e}")
        return data
