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
        self.context_switch_cost = context_switch
        
        # Algorithm parameters
        self.quantum = quantum
        self.age_interval = age_interval
        self.tick_interval = tick_interval
        
        # MLFQ configuration
        if self.algorithm == 'MLFQ':
            self.num_levels = 3
            self.quanta = [8, 16, 0]  # last level FCFS (0 = infinite quantum)
            self.boost_interval = 200  # Priority boost interval
            self.time_slice_used = {}  # Track time used by each process in current quantum
        
        # CFS configuration
        if self.algorithm == 'CFS':
            self.min_granularity = 6  # Minimum scheduling granularity
            self.latency_target = 48  # Target latency
            
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
        self.needs_tick = self.algorithm in ['CFS']
        self.needs_boost = self.algorithm == 'MLFQ'
        
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
        
        if self.needs_boost:
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
        
        if self.algorithm == 'CFS':
            # For CFS, we store (vruntime, pid, process) in heap
            key = (process.vruntime, process.pid, process)
            heapq.heappush(self.ready_queues[cpu_id], key)
        elif self.algorithm == 'EDF':
            # For EDF, we store (deadline, pid, process) in heap
            deadline = getattr(process, 'deadline', float('inf'))
            key = (deadline, process.pid, process)
            heapq.heappush(self.ready_queues[cpu_id], key)
        else:
            # For other algorithms, just append to list
            self.ready_queues[cpu_id].append(process)
    
    def select_process(self, cpu_id: int) -> Optional[Process]:
        """Select the next process to run on the given CPU."""
        queue = self.ready_queues[cpu_id]
        if not queue:
            return None
        
        if self.use_heap:
            _, _, process = heapq.heappop(queue)
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
            # Find process with highest priority (lowest level)
            min_level = min(p.level for p in queue)
            # Among processes with min_level, pick the one that arrived first
            candidates = [p for p in queue if p.level == min_level]
            process = min(candidates, key=lambda p: p.arrival_time)
            queue.remove(process)
            return process
        
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
            # CFS doesn't preempt on arrival, only on tick events
            return False
        elif self.algorithm == "MLFQ":
            return new_process.level < current_process.level
        
        return False
    
    def calculate_cfs_time_slice(self, process: Process, num_ready: int) -> int:
        """Calculate time slice for CFS based on weight and number of ready processes."""
        if num_ready == 0:
            return self.quantum
        
        # CFS formula: time_slice = (weight / total_weight) * latency_target
        # Simplified version for our simulator
        total_weight = sum(p.weight for p in self.processes.values() if p.state != State.TERMINATED)
        if total_weight == 0:
            return self.quantum
        
        time_slice = max(self.min_granularity, 
                        int((process.weight / total_weight) * self.latency_target))
        return min(time_slice, process.remaining_cpu)
    
    def dispatch(self, process: Process, cpu_id: int):
        """Dispatch a process to run on a CPU."""
        # Handle context switch overhead
        if self.context_switch_cost > 0:
            self.clock += self.context_switch_cost
            self.context_switches += 1
            # Update idle time for context switch duration
            self._update_idle_time(cpu_id, self.context_switch_cost)
        
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
        
        if self.algorithm == "RR":
            run_time = min(run_time, self.quantum)
            # Schedule preemption
            heapq.heappush(self.event_queue, 
                          (self.clock + run_time, self.counter, "PREEMPT", process))
            self.counter += 1
        
        elif self.algorithm == "CFS":
            num_ready = len([p for q in self.ready_queues for p in q])
            run_time = self.calculate_cfs_time_slice(process, num_ready)
            # Schedule preemption for CFS
            heapq.heappush(self.event_queue, 
                          (self.clock + run_time, self.counter, "PREEMPT", process))
            self.counter += 1
        
        elif self.algorithm == "MLFQ":
            quantum = self.quanta[process.level]
            if quantum > 0:
                run_time = min(run_time, quantum)
                # Schedule preemption
                heapq.heappush(self.event_queue, 
                              (self.clock + run_time, self.counter, "PREEMPT", process))
                self.counter += 1
            # If quantum is 0 (lowest level), run to completion
        
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
        
        # MLFQ: demote process if it used its full quantum
        if self.algorithm == "MLFQ":
            quantum = self.quanta[process.level]
            if quantum > 0 and (self.clock - self.current_start[cpu_id]) >= quantum:
                process.level = min(process.level + 1, self.num_levels - 1)
        
        # Return to ready queue
        self.add_to_ready(process, cpu_id)
        
        # Clear CPU assignment
        del self.cpu_assignments[process]
        self.current_processes[cpu_id] = None
    
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
        
        # Check for preemption (except for CFS which preempts on tick)
        current = self.current_processes[cpu_id]
        if current and self.should_preempt(process, current, cpu_id) and self.algorithm != "CFS":
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
        process.state = State.READY
        cpu_id = self.get_least_loaded_cpu()
        self.add_to_ready(process, cpu_id)
        
        # Check for preemption
        current = self.current_processes[cpu_id]
        if current and self.should_preempt(process, current, cpu_id):
            self.preempt(current)
        
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
                for _, _, process in cpu_queue:
                    if process.current_priority > 1:
                        process.current_priority -= 1
            else:
                for process in cpu_queue:
                    if process.current_priority > 1:
                        process.current_priority -= 1
        
        # Also age currently running processes
        for process in self.cpu_assignments.keys():
            if process.current_priority > 1:
                process.current_priority -= 1
        
        # Reschedule next aging event
        if len(self.completed) < len(self.processes):
            heapq.heappush(self.event_queue, 
                          (self.clock + self.age_interval, self.counter, "AGING", None))
            self.counter += 1
    
    def handle_tick(self):
        """Handle tick event for CFS."""
        if self.algorithm == 'CFS':
            for cpu_id in range(self.num_cpus):
                current = self.current_processes[cpu_id]
                if current and self.ready_queues[cpu_id]:
                    # Get the process with smallest vruntime
                    min_vruntime = self.ready_queues[cpu_id][0][0]
                    if min_vruntime < current.vruntime:
                        self.preempt(current)
                        self.try_dispatch()
        
        # Reschedule next tick
        if len(self.completed) < len(self.processes):
            heapq.heappush(self.event_queue, 
                          (self.clock + self.tick_interval, self.counter, "TICK", None))
            self.counter += 1
    
    def handle_boost(self):
        """Handle MLFQ priority boost event."""
        for cpu_id in range(self.num_cpus):
            # Boost processes in ready queue
            for item in self.ready_queues[cpu_id]:
                if isinstance(item, tuple):  # Heap item
                    _, _, process = item
                else:  # Regular list item
                    process = item
                process.level = 0  # Boost to highest priority
            
            # Boost currently running process
            current = self.current_processes[cpu_id]
            if current:
                current.level = 0
        
        # Reschedule next boost
        if len(self.completed) < len(self.processes):
            heapq.heappush(self.event_queue, 
                          (self.clock + self.boost_interval, self.counter, "BOOST", None))
            self.counter += 1
    
    def run(self) -> Tuple[List[Process], int, List[Tuple]]:
        """Run the simulation."""
        last_clock = 0
        max_iterations = 100000  # Safety limit to prevent infinite loops
        iteration = 0
        
        while self.event_queue and len(self.completed) < len(self.processes):
            iteration += 1
            if iteration > max_iterations:
                print(f"Warning: Exceeded maximum iterations ({max_iterations})")
                print(f"Completed: {len(self.completed)}/{len(self.processes)}")
                print(f"Event queue size: {len(self.event_queue)}")
                break
            
            time, _, event_type, data = heapq.heappop(self.event_queue)
            
            # Skip events that are in the past (should not happen)
            if time < self.clock:
                continue
            
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
                        # Update vruntime based on weight
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
                self.try_dispatch()
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
        
        print(f"Simulation completed in {iteration} iterations")
        print(f"Total time: {self.clock}, Context switches: {self.context_switches}")
        
        return self.completed, self.clock, self.gantt