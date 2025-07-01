import psutil
import platform
from datetime import datetime
import time

class SystemMetricsCollector:
    """Collects real-time system performance metrics using psutil"""
    
    def __init__(self):
        self.boot_time = datetime.fromtimestamp(psutil.boot_time())
    
    def collect_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
            memory_available = memory.available
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used = disk.used
            disk_total = disk.total
            disk_free = disk.free
            
            # System uptime
            uptime_seconds = time.time() - psutil.boot_time()
            
            # Network metrics (optional)
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': round(cpu_percent, 2),
                'cpu_freq': round(cpu_freq.current if cpu_freq else 0, 2),
                'cpu_count': cpu_count,
                'memory_percent': round(memory_percent, 2),
                'memory_used': memory_used,
                'memory_total': memory_total,
                'memory_available': memory_available,
                'disk_percent': round(disk_percent, 2),
                'disk_used': disk_used,
                'disk_total': disk_total,
                'disk_free': disk_free,
                'uptime': round(uptime_seconds, 2),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': process_count
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return self._get_fallback_metrics()
    
    def get_system_info(self):
        """Get static system information"""
        try:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'hostname': platform.node(),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'cpu_freq': cpu_freq.current if cpu_freq else 0,
                'total_memory': memory.total,
                'total_disk': disk.total,
                'available_disk': disk.free,
                'boot_time': self.boot_time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {
                'platform': 'Unknown',
                'processor': 'Unknown',
                'architecture': 'Unknown',
                'hostname': 'Unknown',
                'cpu_cores': 0,
                'cpu_threads': 0,
                'cpu_freq': 0,
                'total_memory': 0,
                'total_disk': 0,
                'available_disk': 0,
                'boot_time': 'Unknown'
            }
    
    def get_process_info(self, limit=10):
        """Get information about top processes by CPU usage"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['cpu_percent'] is not None:
                        processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage and return top processes
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:limit]
            
        except Exception as e:
            print(f"Error getting process info: {e}")
            return []
    
    def _get_fallback_metrics(self):
        """Return fallback metrics when collection fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': 0.0,
            'cpu_freq': 0.0,
            'cpu_count': 1,
            'memory_percent': 0.0,
            'memory_used': 0,
            'memory_total': 0,
            'memory_available': 0,
            'disk_percent': 0.0,
            'disk_used': 0,
            'disk_total': 0,
            'disk_free': 0,
            'uptime': 0.0,
            'network_bytes_sent': 0,
            'network_bytes_recv': 0,
            'process_count': 0
        }
    
    def get_cpu_per_core(self):
        """Get CPU usage per core"""
        try:
            return psutil.cpu_percent(percpu=True, interval=1)
        except Exception as e:
            print(f"Error getting per-core CPU usage: {e}")
            return []
    
    def get_disk_io(self):
        """Get disk I/O statistics"""
        try:
            disk_io = psutil.disk_io_counters()
            return {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        except Exception as e:
            print(f"Error getting disk I/O: {e}")
            return {
                'read_bytes': 0,
                'write_bytes': 0,
                'read_count': 0,
                'write_count': 0
            }
    
    def get_temperature(self):
        """Get system temperature (if available)"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Return the first available temperature sensor
                for name, entries in temps.items():
                    if entries:
                        return {
                            'sensor': name,
                            'current': entries[0].current,
                            'high': entries[0].high,
                            'critical': entries[0].critical
                        }
            return None
        except Exception as e:
            print(f"Error getting temperature: {e}")
            return None
