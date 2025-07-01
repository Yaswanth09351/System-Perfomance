import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class DataStorage:
    """Handles storage and retrieval of system metrics data"""
    
    def __init__(self, storage_dir="data"):
        self.storage_dir = storage_dir
        self.csv_file = os.path.join(storage_dir, "system_metrics.csv")
        self.json_file = os.path.join(storage_dir, "system_metrics.json")
        self.create_storage_directory()
        self.in_memory_data = []
        self.max_memory_records = 1000  # Keep last 1000 records in memory
        
    def create_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating storage directory: {e}")
    
    def store_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Store system metrics data"""
        try:
            # Add to in-memory storage
            self.in_memory_data.append(metrics.copy())
            
            # Keep only recent records in memory
            if len(self.in_memory_data) > self.max_memory_records:
                self.in_memory_data = self.in_memory_data[-self.max_memory_records:]
            
            # Append to CSV file
            self._append_to_csv(metrics)
            
            # Periodically clean up old data
            if len(self.in_memory_data) % 100 == 0:  # Every 100 records
                self._cleanup_old_data()
            
            return True
            
        except Exception as e:
            print(f"Error storing metrics: {e}")
            return False
    
    def get_historical_data(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical data for the specified number of hours"""
        try:
            # First try to get from memory
            if hours is None:
                return self.in_memory_data.copy()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter in-memory data
            filtered_data = []
            for record in self.in_memory_data:
                try:
                    if isinstance(record['timestamp'], str):
                        record_time = datetime.fromisoformat(record['timestamp'])
                    else:
                        record_time = record['timestamp']
                    
                    if record_time >= cutoff_time:
                        filtered_data.append(record)
                except (ValueError, TypeError) as e:
                    print(f"Error parsing timestamp {record.get('timestamp')}: {e}")
                    continue
            
            # If we don't have enough data in memory, try to load from CSV
            if len(filtered_data) < 10 and os.path.exists(self.csv_file):
                csv_data = self._load_from_csv(hours)
                # Merge with in-memory data, avoiding duplicates
                existing_timestamps = {record['timestamp'] for record in filtered_data}
                for record in csv_data:
                    if record['timestamp'] not in existing_timestamps:
                        filtered_data.append(record)
            
            # Sort by timestamp
            filtered_data.sort(key=lambda x: x['timestamp'])
            
            return filtered_data
            
        except Exception as e:
            print(f"Error retrieving historical data: {e}")
            return []
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all stored data"""
        try:
            # Load from CSV if it exists
            if os.path.exists(self.csv_file):
                return self._load_from_csv()
            else:
                return self.in_memory_data.copy()
        except Exception as e:
            print(f"Error retrieving all data: {e}")
            return []
    
    def _append_to_csv(self, metrics: Dict[str, Any]) -> bool:
        """Append metrics to CSV file"""
        try:
            df = pd.DataFrame([metrics])
            
            # If file exists, append without header
            if os.path.exists(self.csv_file):
                df.to_csv(self.csv_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_file, mode='w', header=True, index=False)
            
            return True
            
        except Exception as e:
            print(f"Error appending to CSV: {e}")
            return False
    
    def _load_from_csv(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        try:
            if not os.path.exists(self.csv_file):
                return []
            
            df = pd.read_csv(self.csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if hours is not None:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                df = df[df['timestamp'] >= cutoff_time]
            
            # Convert timestamps back to strings and return as records
            records = []
            for _, row in df.iterrows():
                record = row.to_dict()
                if isinstance(record['timestamp'], pd.Timestamp):
                    record['timestamp'] = record['timestamp'].strftime('%Y-%m-%dT%H:%M:%S')
                records.append(record)
            return records
            
        except Exception as e:
            print(f"Error loading from CSV: {e}")
            return []
    
    def export_to_json(self, hours: Optional[int] = None) -> str:
        """Export data to JSON format"""
        try:
            data = self.get_historical_data(hours) if hours else self.get_all_data()
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return "[]"
    
    def export_to_csv(self, hours: Optional[int] = None) -> str:
        """Export data to CSV format"""
        try:
            data = self.get_historical_data(hours) if hours else self.get_all_data()
            if not data:
                return ""
            
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return ""
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        try:
            all_data = self.get_all_data()
            
            if not all_data:
                return {
                    'total_records': 0,
                    'date_range': None,
                    'storage_size_mb': 0,
                    'metrics_available': []
                }
            
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate storage size
            csv_size = os.path.getsize(self.csv_file) / (1024 * 1024) if os.path.exists(self.csv_file) else 0
            
            stats = {
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                },
                'storage_size_mb': round(csv_size, 2),
                'metrics_available': [col for col in df.columns if col != 'timestamp'],
                'memory_records': len(self.in_memory_data)
            }
            
            # Add basic statistics for key metrics
            for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if metric in df.columns:
                    stats[f'{metric}_stats'] = {
                        'mean': round(df[metric].mean(), 2),
                        'min': round(df[metric].min(), 2),
                        'max': round(df[metric].max(), 2),
                        'std': round(df[metric].std(), 2)
                    }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating data statistics: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_data(self, max_age_days: int = 7) -> bool:
        """Clean up data older than specified days"""
        try:
            if not os.path.exists(self.csv_file):
                return True
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Load current data
            df = pd.read_csv(self.csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter recent data
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            # Save back to file
            recent_df.to_csv(self.csv_file, index=False)
            
            print(f"Cleaned up old data. Removed {len(df) - len(recent_df)} records older than {max_age_days} days.")
            
            return True
            
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
            return False
    
    def backup_data(self, backup_dir: str = "backups") -> bool:
        """Create a backup of current data"""
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"system_metrics_backup_{timestamp}.csv")
            
            if os.path.exists(self.csv_file):
                import shutil
                shutil.copy2(self.csv_file, backup_file)
                print(f"Data backed up to {backup_file}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """Restore data from a backup file"""
        try:
            if not os.path.exists(backup_file):
                print(f"Backup file {backup_file} not found")
                return False
            
            import shutil
            shutil.copy2(backup_file, self.csv_file)
            
            # Reload in-memory data
            self.in_memory_data = self._load_from_csv(hours=24)  # Load last 24 hours
            
            print(f"Data restored from {backup_file}")
            return True
            
        except Exception as e:
            print(f"Error restoring from backup: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """Clear all stored data (use with caution)"""
        try:
            # Clear in-memory data
            self.in_memory_data = []
            
            # Remove CSV file
            if os.path.exists(self.csv_file):
                os.remove(self.csv_file)
            
            # Remove JSON file if it exists
            if os.path.exists(self.json_file):
                os.remove(self.json_file)
            
            print("All data cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
