import logging
from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all intelligent agents.
    Provides state management, learning, and decision-making infrastructure.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.state = {
            'execution_history': [],
            'performance_metrics': {},
            'learned_patterns': {},
            'total_executions': 0,
            'success_count': 0,
            'failure_count': 0
        }
        self.goals = {}
        self.constraints = {}
        
    def record_execution(self, task: str, result: Dict[str, Any], success: bool):
        """Record execution in agent's memory"""
        execution_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'task': task,
            'result': result,
            'success': success
        }
        
        self.state['execution_history'].append(execution_record)
        self.state['total_executions'] += 1
        
        if success:
            self.state['success_count'] += 1
        else:
            self.state['failure_count'] += 1
        
        # Keep only last 100 executions
        if len(self.state['execution_history']) > 100:
            self.state['execution_history'] = self.state['execution_history'][-100:]
    
    def get_success_rate(self) -> float:
        """Calculate agent's success rate"""
        if self.state['total_executions'] == 0:
            return 0.0
        return self.state['success_count'] / self.state['total_executions']
    
    def learn_from_execution(self, pattern_key: str, outcome: Dict[str, Any]):
        """Agent learns patterns from executions"""
        if pattern_key not in self.state['learned_patterns']:
            self.state['learned_patterns'][pattern_key] = []
        
        self.state['learned_patterns'][pattern_key].append(outcome)
        
        # Keep only recent patterns
        if len(self.state['learned_patterns'][pattern_key]) > 50:
            self.state['learned_patterns'][pattern_key] = \
                self.state['learned_patterns'][pattern_key][-50:]
    
    def get_learned_pattern(self, pattern_key: str) -> List[Dict]:
        """Retrieve learned patterns"""
        return self.state['learned_patterns'].get(pattern_key, [])
    
    def save_state(self, filepath: str):
        """Persist agent state to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.info(f"{self.agent_name} state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load agent state from disk"""
        try:
            with open(filepath, 'r') as f:
                self.state = json.load(f)
            logger.info(f"{self.agent_name} state loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"No saved state found for {self.agent_name}")