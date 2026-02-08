import heapq
import math
from typing import List, Tuple, Dict, Optional, Any
from process import Process, State

class Simulator:
    """CPU scheduling simulator with support for multiple algorithms and CPUs."""
    
    def __init__(self, algorithm: str, processes: List[Process], 
                 context_switch: int = 2, quantum: int = 20,
                 age_interval: int = 100, num_cpus: int = 1, 
                 tick_interval: int = 4):
        # Algorithm configuration
        self.algorithm = algorithm.upper()
        self.processes = {p.pid: p for p in processes}
        self.num_cpus = num_cpus
        
        # Time tracking
        self.clock = 0
        self.event_queue = []  # (time, counter, event_type, data)
        self.counter = 0
        
        # CPU state
        self.ready_queues = [[] for _ in range(num_cpus)]
        self.current_processes = [None] * num_cpus
        self.cpu_assignments = {}  # pid -> cpu_id
        self.current_start = [0] * num_cpus
        
        # Algorithm parameters
        self.context_switch = context_switch
        self.quantum = quantum
        self.age_interval = age_interval
        self.tick_interval = tick_interval
        
        # MLFQ configuration
        if self.algorithm == 'MLFQ':
            self.num_levels = 3
            self.quanta = [8, 16, 0]  # last level FCFS
            self.boost_interval = 200  # Priority boost interval
        
        # Results
        self.gantt = []  # List of (start, end, pid, cpu_id)
        self.completed = []
        
        # Statistics
        self.context_switches = 0
        self.preemptions = 0
        self.idle_time = [0] * num_cpus
        
        # Optimization flags
        self.use_heap = self.algorithm in ['CFS', 'EDF']
        self.needs_aging = 'PRIORITY' in self.algorithm
        self.needs_tick = self.algorithm in ['CFS', 'MLFQ']
        
        # Initialize event queue
        self._initialize_events()
    
    def _initialize_events(self):
        """Initialize the event queue with process arrivals."""
        for p in self.processes.values():
            heapq.heappush(self.event_queue, 
                          (p.arrival_time, self.counter, "ARRIVAL", p))
            self.counter += 1
        
        # Add periodic events
        if self.needs_aging:
            heapq.heappush(self.event_queue, 
                          (self.age_interval, self.counter, "AGING", None))
            self.counter += 1
        
        if self.needs_tick:
            heapq.heappush(self.event_queue, 
                          (self.tick_interval, self.counter, "TICK", None))
            self.counter += 1
        
        if self.algorithm == 'MLFQ':
            heapq.heappush(self.event_queue, 
                          (self.boost_interval, self.counter, "BOOST", None))
            self.counter += 1
    
    def _update_idle_time(self, cpu_id: int, duration: int):
        """Update idle time tracking for a CPU."""
        if self.current_processes[cpu_id] is None:
            self.idle_time[cpu_id] += duration
    
    def get_least_loaded_cpu(self) -> int:
        """Return the CPU with the least load."""
        loads = []
        for cpu_id in range(self.num_cpus):
            load = len(self.ready_queues[cpu_id])
            if self.current_processes[cpu_id] is not None:
                load += 1
            loads.append(load)
        return loads.index(min(loads))
    
    def add_to_ready(self, process: Process, cpu_id: Optional[int] = None):
        """Add a process to the appropriate ready queue."""
        if cpu_id is None:
            cpu_id = self.get_least_loaded_cpu()
        
        process.state = State.READY
        
        if self.use_heap:
            if self.algorithm == 'CFS':
                key = (process.vruntime, process.arrival_time, process.pid, process)
            elif self.algorithm == 'EDF':
                deadline = process.deadline if hasattr(process, 'deadline') else float('inf')
                key = (deadline, process.arrival_time, process.pid, process)
            heapq.heappush(self.ready_queues[cpu_id], key)
        else:
            self.ready_queues[cpu_id].append(process)
    
    def select_process(self, cpu_id: int) -> Optional[Process]:
        """Select the next process to run on the given CPU."""
        queue = self.ready_queues[cpu_id]
        if not queue:
            return None
        
        if self.use_heap:
            _, _, _, process = heapq.heappop(queue)
            return process
        
        # Different selection strategies for different algorithms
        if self.algorithm in ["FCFS", "RR"]:
            return queue.pop(0)
        elif self.algorithm == "SJF":
            idx = min(range(len(queue)), key=lambda i: (queue[i].cpu_burst, queue[i].arrival_time))
            return queue.pop(idx)
        elif self.algorithm == "SRTF":
            idx = min(range(len(queue)), key=lambda i: (queue[i].remaining_cpu, queue[i].arrival_time))
            return queue.pop(idx)
        elif "PRIORITY" in self.algorithm:
            idx = min(range(len(queue)), key=lambda i: (queue[i].current_priority, queue[i].arrival_time))
            return queue.pop(idx)
        elif self.algorithm == 'MLFQ':
            idx = min(range(len(queue)), key=lambda i: (queue[i].level, queue[i].arrival_time))
            return queue.pop(idx)
        
        return queue.pop(0)
    
    def should_preempt(self, new_process: Process, current_process: Process, 
                      cpu_id: int) -> bool:
        """Determine if current process should be preempted."""
        if current_process is None:
            return False
        
        if self.algorithm == "SRTF":
            return new_process.remaining_cpu < current_process.remaining_cpu
        elif self.algorithm == "PRIORITY_PRE":
            return new_process.current_priority < current_process.current_priority
        elif self.algorithm == "EDF":
            new_deadline = getattr(new_process, 'deadline', float('inf'))
            cur_deadline = getattr(current_process, 'deadline', float('inf'))
            return new_deadline < cur_deadline
        elif self.algorithm == "CFS":
            return new_process.vruntime < current_process.vruntime
        elif self.algorithm == "MLFQ":
            return new_process.level < current_process.level
        
        return False
    
    def dispatch(self, process: Process, cpu_id: int):
        """Dispatch a process to run on a CPU."""
        # Handle context switch overhead
        if self.context_switch > 0:
            self.clock += self.context_switch
            self.context_switches += 1
        
        # Update CPU state
        self.current_processes[cpu_id] = process
        self.cpu_assignments[process] = cpu_id
        self.current_start[cpu_id] = self.clock
        process.state = State.RUNNING
        
        # Record response time (first time running)
        if process.start_time == -1:
            process.start_time = self.clock
            process.response_time = self.clock - process.arrival_time
        
        # Calculate run time based on algorithm
        run_time = process.remaining_cpu
        
        if self.algorithm in ["RR", "CFS"]:
            run_time = min(run_time, self.quantum)
        elif self.algorithm == "MLFQ":
            quantum = self.quanta[process.level]
            if quantum > 0:
                run_time = min(run_time, quantum)
        
        # Schedule events
        if run_time < process.remaining_cpu:
            # Schedule preemption
            heapq.heappush(self.event_queue, 
                          (self.clock + run_time, self.counter, "PREEMPT", process))
            self.counter += 1
        
        # Always schedule completion event
        heapq.heappush(self.event_queue, 
                      (self.clock + process.remaining_cpu, self.counter, "CPU_COMPLETE", process))
        self.counter += 1
    
    def preempt(self, process: Process):
        """Preempt a running process."""
        cpu_id = self.cpu_assignments.get(process)
        if cpu_id is None or self.current_processes[cpu_id] != process:
            return
        
        # Record Gantt chart entry
        self.gantt.append((self.current_start[cpu_id], self.clock, process.pid, cpu_id))
        self.preemptions += 1
        
        # Update process state
        process.state = State.READY
        
        # MLFQ: demote process
        if self.algorithm == "MLFQ":
            process.level = min(process.level + 1, self.num_levels - 1)
        
        # Return to ready queue
        self.add_to_ready(process, cpu_id)
        
        # Clear CPU assignment
        del self.cpu_assignments[process]
        self.current_processes[cpu_id] = None
        
        # Context switch overhead
        if self.context_switch > 0:
            self.clock += self.context_switch
            self.context_switches += 1
    
    def try_dispatch(self):
        """Attempt to dispatch processes to idle CPUs."""
        for cpu_id in range(self.num_cpus):
            if (self.current_processes[cpu_id] is None and 
                self.ready_queues[cpu_id]):
                process = self.select_process(cpu_id)
                if process:
                    self.dispatch(process, cpu_id)
    
    def handle_arrival(self, process: Process):
        """Handle process arrival event."""
        cpu_id = self.get_least_loaded_cpu()
        self.add_to_ready(process, cpu_id)
        
        # Check for preemption
        current = self.current_processes[cpu_id]
        if current and self.should_preempt(process, current, cpu_id):
            self.preempt(current)
        
        self.try_dispatch()
    
    def handle_cpu_complete(self, process: Process):
        """Handle CPU completion event."""
        cpu_id = self.cpu_assignments.get(process)
        
        # Ignore if process is no longer running
        if cpu_id is None or self.current_processes[cpu_id] != process:
            return
        
        # Record Gantt entry
        self.gantt.append((self.current_start[cpu_id], self.clock, process.pid, cpu_id))
        
        # Clear CPU assignment
        del self.cpu_assignments[process]
        self.current_processes[cpu_id] = None
        
        # Handle IO or completion
        if process.io_burst > 0:
            process.state = State.WAITING
            heapq.heappush(self.event_queue, 
                          (self.clock + process.io_burst, self.counter, "IO_COMPLETE", process))
            self.counter += 1
        else:
            self.complete_process(process)
        
        # Try to dispatch next process
        self.try_dispatch()
    
    def handle_io_complete(self, process: Process):
        """Handle IO completion event."""
        self.complete_process(process)
        self.try_dispatch()
    
    def complete_process(self, process: Process):
        """Mark a process as completed and calculate statistics."""
        process.state = State.TERMINATED
        process.completion_time = self.clock
        process.turnaround_time = self.clock - process.arrival_time
        process.waiting_time = process.turnaround_time - process.cpu_burst - process.io_burst
        
        # Check for deadline miss
        if hasattr(process, 'deadline') and process.deadline is not None:
            if self.clock > process.deadline:
                process.missed_deadline = True
        
        self.completed.append(process)
    
    def handle_aging(self):
        """Handle priority aging event."""
        for cpu_queue in self.ready_queues:
            if self.use_heap:
                for _, _, _, process in cpu_queue:
                    if process.current_priority > 1:
                        process.current_priority -= 1
            else:
                for process in cpu_queue:
                    if process.current_priority > 1:
                        process.current_priority -= 1
        
        # Reschedule next aging event
        if len(self.completed) < len(self.processes):
            heapq.heappush(self.event_queue, 
                          (self.clock + self.age_interval, self.counter, "AGING", None))
            self.counter += 1
    
    def handle_tick(self):
        """Handle tick event for CFS and MLFQ."""
        if self.algorithm == 'CFS':
            for cpu_id in range(self.num_cpus):
                current = self.current_processes[cpu_id]
                if current and self.ready_queues[cpu_id]:
                    # Get the process with smallest vruntime
                    min_vruntime = self.ready_queues[cpu_id][0][0]
                    if min_vruntime < current.vruntime:
                        self.preempt(current)
        
        # Reschedule next tick
        if len(self.completed) < len(self.processes):
            heapq.heappush(self.event_queue, 
                          (self.clock + self.tick_interval, self.counter, "TICK", None))
            self.counter += 1
    
    def handle_boost(self):
        """Handle MLFQ priority boost event."""
        for cpu_id in range(self.num_cpus):
            for i, item in enumerate(self.ready_queues[cpu_id]):
                if self.use_heap:
                    _, _, _, process = item
                else:
                    process = item
                process.level = 0  # Boost to highest priority
        
        # Reschedule next boost
        if len(self.completed) < len(self.processes):
            heapq.heappush(self.event_queue, 
                          (self.clock + self.boost_interval, self.counter, "BOOST", None))
            self.counter += 1
    
    def run(self) -> Tuple[List[Process], int, List[Tuple]]:
        """Run the simulation."""
        last_clock = 0
        
        while self.event_queue and len(self.completed) < len(self.processes):
            time, _, event_type, data = heapq.heappop(self.event_queue)
            
            # Update idle time for all CPUs
            elapsed = time - last_clock
            for cpu_id in range(self.num_cpus):
                self._update_idle_time(cpu_id, elapsed)
            
            # Advance clock and update running processes
            elapsed = time - self.clock
            for cpu_id in range(self.num_cpus):
                current = self.current_processes[cpu_id]
                if current:
                    current.remaining_cpu -= elapsed
                    if self.algorithm == 'CFS':
                        current.vruntime += elapsed * 1024 / current.weight
            
            self.clock = time
            last_clock = time
            
            # Handle event
            if event_type == "ARRIVAL":
                self.handle_arrival(data)
            elif event_type == "CPU_COMPLETE":
                self.handle_cpu_complete(data)
            elif event_type == "IO_COMPLETE":
                self.handle_io_complete(data)
            elif event_type == "PREEMPT":
                self.preempt(data)
            elif event_type == "AGING":
                self.handle_aging()
            elif event_type == "TICK":
                self.handle_tick()
            elif event_type == "BOOST":
                self.handle_boost()
        
        # Complete any remaining running processes
        for cpu_id in range(self.num_cpus):
            if self.current_processes[cpu_id]:
                process = self.current_processes[cpu_id]
                self.gantt.append((self.current_start[cpu_id], self.clock, process.pid, cpu_id))
                self.complete_process(process)
        
        # Add simulation statistics to results
        total_idle = sum(self.idle_time)
        total_busy = (self.clock * self.num_cpus) - total_idle
        
        return self.completed, self.clock, self.gantt