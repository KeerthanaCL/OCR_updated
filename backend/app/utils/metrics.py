import time
from typing import Dict, Any, List
from datetime import datetime

class MetricsCollector:
    """Simple metrics collector for tracking agent performance"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time = None
        self.metrics = {
            'agent': agent_name,
            'started_at': None,
            'completed_at': None,
            'duration_seconds': 0.0,
            'success': False,
            'errors': []
        }
    
    def start(self):
        """Start tracking time"""
        self.start_time = time.time()
        self.metrics['started_at'] = datetime.utcnow().isoformat()
        return self
    
    def record_error(self, error: str, severity: str = 'error'):
        """Record an error"""
        self.metrics['errors'].append({
            'message': error,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def complete(self, success: bool = True) -> Dict[str, Any]:
        """Complete and return metrics"""
        if self.start_time:
            self.metrics['duration_seconds'] = time.time() - self.start_time
        self.metrics['completed_at'] = datetime.utcnow().isoformat()
        self.metrics['success'] = success
        return self.metrics
    
    def add_custom_metric(self, key: str, value: Any):
        """Add custom metric"""
        self.metrics[key] = value