from enum import Enum
from dataclasses import dataclass
from typing import Optional

class State(Enum):
    NEW = 1
    READY = 2
    RUNNING = 3
    WAITING = 4
    TERMINATED = 5

@dataclass
class ProcessStatistics:
    """Container for process statistics."""
    waiting_time: int = 0
    turnaround_time: int = 0
    response_time: int = -1
    completion_time: int = 0
    start_time: int = -1
    missed_deadline: bool = False

class Process:
    """Represents a process in the scheduling system."""
    
    def __init__(self, pid: int, arrival_time: int, cpu_burst: int, 
                 io_burst: int = 0, priority: int = 5, 
                 deadline: Optional[int] = None):
        # Validate inputs
        if cpu_burst <= 0:
            raise ValueError("CPU burst must be positive")
        if io_burst < 0:
            raise ValueError("IO burst cannot be negative")
        if priority < 1 or priority > 10:
            raise ValueError("Priority must be between 1 and 10")
        
        # Core attributes
        self.pid = pid
        self.arrival_time = arrival_time
        self.cpu_burst = cpu_burst
        self.remaining_cpu = cpu_burst
        self.io_burst = io_burst
        self.original_priority = priority
        self.current_priority = priority
        
        # Deadline handling
        if deadline is None:
            # Default deadline: arrival + cpu_burst * 2 + io_burst
            self.deadline = arrival_time + (cpu_burst * 2) + io_burst
        else:
            self.deadline = deadline
        
        # State management
        self.state = State.NEW
        
        # Scheduling algorithm-specific attributes
        self.level = 0  # For MLFQ (0 = highest priority)
        
        # CFS attributes
        self.weight = self._calculate_weight(priority)
        self.vruntime = 0.0
        
        # Statistics
        self.stats = ProcessStatistics()
    
    def _calculate_weight(self, priority: int) -> float:
        """Calculate CFS weight from priority."""
        # Map priority 1-10 to nice values -20 to +19
        # priority 1 = highest = nice -20
        # priority 10 = lowest = nice +19
        nice = priority - 1 - 20  # Map 1-10 to -20 to +19
        # Weight formula: weight = 1024 / (1.25 ** nice)
        return 1024 / (1.25 ** nice)
    
    @property
    def waiting_time(self) -> int:
        return self.stats.waiting_time
    
    @waiting_time.setter
    def waiting_time(self, value: int):
        self.stats.waiting_time = value
    
    @property
    def turnaround_time(self) -> int:
        return self.stats.turnaround_time
    
    @turnaround_time.setter
    def turnaround_time(self, value: int):
        self.stats.turnaround_time = value
    
    @property
    def response_time(self) -> int:
        return self.stats.response_time
    
    @response_time.setter
    def response_time(self, value: int):
        self.stats.response_time = value
    
    @property
    def completion_time(self) -> int:
        return self.stats.completion_time
    
    @completion_time.setter
    def completion_time(self, value: int):
        self.stats.completion_time = value
    
    @property
    def start_time(self) -> int:
        return self.stats.start_time
    
    @start_time.setter
    def start_time(self, value: int):
        self.stats.start_time = value
    
    @property
    def missed_deadline(self) -> bool:
        return self.stats.missed_deadline
    
    @missed_deadline.setter
    def missed_deadline(self, value: bool):
        self.stats.missed_deadline = value
    
    def __repr__(self) -> str:
        return (f"Process(pid={self.pid}, arrival={self.arrival_time}, "
                f"cpu={self.cpu_burst}, io={self.io_burst}, "
                f"priority={self.original_priority}, state={self.state.name})")