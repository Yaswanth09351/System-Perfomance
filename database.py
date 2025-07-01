import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import pandas as pd

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SystemMetrics(Base):
    """Database model for system metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_percent = Column(Float)
    cpu_freq = Column(Float)
    cpu_count = Column(Integer)
    memory_percent = Column(Float)
    memory_used = Column(Float)
    memory_total = Column(Float)
    memory_available = Column(Float)
    disk_percent = Column(Float)
    disk_used = Column(Float)
    disk_total = Column(Float)
    disk_free = Column(Float)
    uptime = Column(Float)
    network_bytes_sent = Column(Float)
    network_bytes_recv = Column(Float)
    process_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class Users(Base):
    """Database model for users"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))
    full_name = Column(String(255))
    role = Column(String(50), default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    preferences = Column(Text)  # JSON string for user preferences

class Alerts(Base):
    """Database model for alerts"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(50))  # warning, critical, info
    metric = Column(String(100))
    value = Column(Float)
    threshold = Column(Float)
    message = Column(Text)
    severity = Column(String(50))
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

class AlertThresholds(Base):
    """Database model for alert thresholds"""
    __tablename__ = "alert_thresholds"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), unique=True)
    warning_threshold = Column(Float)
    critical_threshold = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass

class DatabaseStorage:
    """Database-backed storage for system metrics and application data"""
    
    def __init__(self):
        create_tables()
        self.db = SessionLocal()
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()
    
    def store_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Store system metrics in database"""
        try:
            # Parse timestamp
            if isinstance(metrics.get('timestamp'), str):
                timestamp = datetime.fromisoformat(metrics['timestamp'].replace('Z', '+00:00'))
            else:
                timestamp = datetime.utcnow()
            
            metric_record = SystemMetrics(
                timestamp=timestamp,
                cpu_percent=metrics.get('cpu_percent', 0.0),
                cpu_freq=metrics.get('cpu_freq', 0.0),
                cpu_count=metrics.get('cpu_count', 0),
                memory_percent=metrics.get('memory_percent', 0.0),
                memory_used=metrics.get('memory_used', 0),
                memory_total=metrics.get('memory_total', 0),
                memory_available=metrics.get('memory_available', 0),
                disk_percent=metrics.get('disk_percent', 0.0),
                disk_used=metrics.get('disk_used', 0),
                disk_total=metrics.get('disk_total', 0),
                disk_free=metrics.get('disk_free', 0),
                uptime=metrics.get('uptime', 0.0),
                network_bytes_sent=metrics.get('network_bytes_sent', 0),
                network_bytes_recv=metrics.get('network_bytes_recv', 0),
                process_count=metrics.get('process_count', 0)
            )
            
            self.db.add(metric_record)
            self.db.commit()
            
            # Clean up old records (keep only last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            self.db.query(SystemMetrics).filter(SystemMetrics.timestamp < cutoff_date).delete()
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"Error storing metrics: {e}")
            return False
    
    def get_historical_data(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical metrics data from database"""
        try:
            query = self.db.query(SystemMetrics).order_by(SystemMetrics.timestamp.desc())
            
            if hours is not None:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                query = query.filter(SystemMetrics.timestamp >= cutoff_time)
            
            # Limit to prevent memory issues
            records = query.limit(10000).all()
            
            # Convert to list of dictionaries
            data = []
            for record in records:
                data.append({
                    'timestamp': record.timestamp.isoformat(),
                    'cpu_percent': record.cpu_percent,
                    'cpu_freq': record.cpu_freq,
                    'cpu_count': record.cpu_count,
                    'memory_percent': record.memory_percent,
                    'memory_used': record.memory_used,
                    'memory_total': record.memory_total,
                    'memory_available': record.memory_available,
                    'disk_percent': record.disk_percent,
                    'disk_used': record.disk_used,
                    'disk_total': record.disk_total,
                    'disk_free': record.disk_free,
                    'uptime': record.uptime,
                    'network_bytes_sent': record.network_bytes_sent,
                    'network_bytes_recv': record.network_bytes_recv,
                    'process_count': record.process_count
                })
            
            # Return in chronological order
            return list(reversed(data))
            
        except Exception as e:
            print(f"Error retrieving historical data: {e}")
            return []
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all stored metrics data"""
        return self.get_historical_data()
    
    def export_to_json(self, hours: Optional[int] = None) -> str:
        """Export data to JSON format"""
        try:
            data = self.get_historical_data(hours)
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return "[]"
    
    def export_to_csv(self, hours: Optional[int] = None) -> str:
        """Export data to CSV format"""
        try:
            data = self.get_historical_data(hours)
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
            total_records = self.db.query(SystemMetrics).count()
            
            if total_records == 0:
                return {
                    'total_records': 0,
                    'date_range': None,
                    'metrics_available': []
                }
            
            # Get date range
            oldest = self.db.query(func.min(SystemMetrics.timestamp)).scalar()
            newest = self.db.query(func.max(SystemMetrics.timestamp)).scalar()
            
            stats = {
                'total_records': total_records,
                'date_range': {
                    'start': oldest.isoformat() if oldest else None,
                    'end': newest.isoformat() if newest else None,
                    'duration_hours': (newest - oldest).total_seconds() / 3600 if oldest and newest else 0
                },
                'metrics_available': [
                    'cpu_percent', 'memory_percent', 'disk_percent', 'uptime',
                    'network_bytes_sent', 'network_bytes_recv', 'process_count'
                ]
            }
            
            # Add basic statistics for key metrics
            for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                column = getattr(SystemMetrics, metric)
                result = self.db.query(
                    func.avg(column).label('mean'),
                    func.min(column).label('min'),
                    func.max(column).label('max'),
                    func.stddev(column).label('std')
                ).first()
                
                if result and result.mean is not None:
                    stats[f'{metric}_stats'] = {
                        'mean': round(float(result.mean), 2),
                        'min': round(float(result.min), 2),
                        'max': round(float(result.max), 2),
                        'std': round(float(result.std or 0), 2)
                    }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating data statistics: {e}")
            return {'error': str(e)}
    
    def clear_all_data(self) -> bool:
        """Clear all stored data"""
        try:
            self.db.query(SystemMetrics).delete()
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error clearing data: {e}")
            return False
    
    # User management methods
    def create_user(self, email: str, password_hash: str, full_name: str, role: str = 'user') -> bool:
        """Create a new user in database"""
        try:
            user = Users(
                email=email,
                password_hash=password_hash,
                full_name=full_name,
                role=role,
                preferences=json.dumps({
                    'theme': 'light',
                    'auto_refresh': True,
                    'refresh_interval': 5
                })
            )
            self.db.add(user)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error creating user: {e}")
            return False
    
    def get_user(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = self.db.query(Users).filter(Users.email == email).first()
            if user:
                return {
                    'email': user.email,
                    'password': user.password_hash,
                    'full_name': user.full_name,
                    'role': user.role,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'preferences': json.loads(user.preferences) if user.preferences else {}
                }
            return None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def update_user_login(self, email: str) -> bool:
        """Update user's last login time"""
        try:
            user = self.db.query(Users).filter(Users.email == email).first()
            if user:
                user.last_login = datetime.utcnow()
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"Error updating user login: {e}")
            return False
    
    def get_all_users(self) -> Dict[str, Dict[str, Any]]:
        """Get all users"""
        try:
            users = self.db.query(Users).all()
            result = {}
            for user in users:
                result[user.email] = {
                    'email': user.email,
                    'password': user.password_hash,
                    'full_name': user.full_name,
                    'role': user.role,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'preferences': json.loads(user.preferences) if user.preferences else {}
                }
            return result
        except Exception as e:
            print(f"Error getting all users: {e}")
            return {}
    
    def update_user_preferences(self, email: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            user = self.db.query(Users).filter(Users.email == email).first()
            if user:
                current_prefs = json.loads(user.preferences) if user.preferences else {}
                current_prefs.update(preferences)
                user.preferences = json.dumps(current_prefs)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"Error updating user preferences: {e}")
            return False
    
    def delete_user(self, email: str) -> bool:
        """Delete user"""
        try:
            user = self.db.query(Users).filter(Users.email == email).first()
            if user:
                self.db.delete(user)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"Error deleting user: {e}")
            return False
    
    def update_user_role(self, email: str, new_role: str) -> bool:
        """Update user role"""
        try:
            user = self.db.query(Users).filter(Users.email == email).first()
            if user:
                user.role = new_role
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"Error updating user role: {e}")
            return False