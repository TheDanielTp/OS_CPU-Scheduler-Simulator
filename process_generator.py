import random
import csv
from typing import List, Optional
from process import Process

def generate_synthetic(num_processes: int, mean_interarrival: float = 20, 
                      mean_cpu: float = 50, sigma_cpu: float = 20,
                      io_probability: float = 0.3, io_min: int = 10, 
                      io_max: int = 100, pri_min: int = 1, pri_max: int = 10,
                      distribution: str = 'exponential', 
                      deadline_factor: float = 2.0) -> List[Process]:
    """
    Generate synthetic processes with various distributions.
    
    Args:
        num_processes: Number of processes to generate
        distribution: 'exponential', 'uniform', 'bimodal', or 'pareto'
        deadline_factor: Multiplier for CPU burst to calculate deadline
    """
    processes = []
    current_time = 0
    
    # Set random seed for reproducibility
    random.seed(42)  # Fixed seed for reproducibility
    
    for i in range(num_processes):
        # Inter-arrival time
        inter_arrival = random.expovariate(1 / mean_interarrival)
        arrival = current_time + inter_arrival
        
        # CPU burst based on distribution
        if distribution == 'exponential':
            cpu = random.expovariate(1 / mean_cpu)
        elif distribution == 'uniform':
            cpu = random.uniform(mean_cpu - sigma_cpu, mean_cpu + sigma_cpu)
        elif distribution == 'bimodal':
            # Bimodal distribution: mix of short and long processes
            if random.random() < 0.7:  # 70% short processes
                cpu = random.expovariate(1 / (mean_cpu * 0.3))
            else:  # 30% long processes
                cpu = random.expovariate(1 / (mean_cpu * 3))
        elif distribution == 'pareto':
            # Pareto distribution (heavy-tailed)
            alpha = 2.0
            cpu = random.paretovariate(alpha) * mean_cpu / 2
        else:
            cpu = random.gauss(mean_cpu, sigma_cpu)
        
        cpu = max(1, int(cpu))
        
        # IO burst (some processes have no IO)
        if random.random() < io_probability:
            io = random.randint(io_min, io_max)
        else:
            io = 0
        
        # Priority
        pri = random.randint(pri_min, pri_max)
        
        # Deadline (optional, for EDF)
        deadline = int(arrival + cpu * deadline_factor + io)
        
        # Create process
        processes.append(Process(
            pid=i + 1,
            arrival_time=int(arrival),
            cpu_burst=cpu,
            io_burst=io,
            priority=pri,
            deadline=deadline
        ))
        
        current_time = arrival
    
    return processes

def load_trace(file_path: str) -> List[Process]:
    """Load processes from a trace file."""
    processes = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                
                # Support different formats
                if len(row) >= 5:
                    pid = int(row[0])
                    arrival = int(row[1])
                    cpu = int(row[2])
                    io = int(row[3])
                    pri = int(row[4])
                    deadline = int(row[5]) if len(row) > 5 else None
                    
                    processes.append(Process(
                        pid=pid,
                        arrival_time=arrival,
                        cpu_burst=cpu,
                        io_burst=io,
                        priority=pri,
                        deadline=deadline
                    ))
    except FileNotFoundError:
        raise FileNotFoundError(f"Trace file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing trace file: {e}")
    
    return processes

def save_trace(processes: List[Process], file_path: str):
    """Save processes to a trace file."""
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# pid,arrival_time,cpu_burst,io_burst,priority,deadline'])
        for p in processes:
            writer.writerow([
                p.pid,
                p.arrival_time,
                p.cpu_burst,
                p.io_burst,
                p.original_priority,
                getattr(p, 'deadline', '')
            ])

def generate_workload_mix(num_processes: int) -> List[Process]:
    """Generate a mixed workload with different types of processes."""
    processes = []
    
    # Interactive processes (short CPU, frequent IO)
    interactive = int(num_processes * 0.4)
    for i in range(interactive):
        arrival = random.randint(0, 50)
        cpu = random.randint(1, 20)
        io = random.randint(10, 50) if random.random() < 0.8 else 0
        p = Process(
            pid=len(processes) + 1,
            arrival_time=arrival,
            cpu_burst=cpu,
            io_burst=io,
            priority=random.randint(1, 3)
        )
        processes.append(p)
    
    # Batch processes (long CPU, little IO)
    batch = int(num_processes * 0.4)
    for i in range(batch):
        offset = len(processes)
        p = Process(
            pid=offset + i + 1,
            arrival_time=random.randint(0, 100),
            cpu_burst=random.randint(100, 500),
            io_burst=0,
            priority=random.randint(7, 10)
        )
        processes.append(p)
    
    # Real-time processes (varying deadlines)
    realtime = num_processes - interactive - batch
    for i in range(realtime):
        offset = len(processes)
        cpu = random.randint(5, 50)
        arrival = random.randint(0, 50)
        p = Process(
            pid=offset + i + 1,
            arrival_time=arrival,
            cpu_burst=cpu,
            io_burst=0,
            priority=random.randint(1, 3),
            deadline=arrival + cpu + random.randint(0, 20)
        )
        processes.append(p)
    
    # Sort by arrival time
    processes.sort(key=lambda p: p.arrival_time)
    
    # Reassign PIDs
    for i, p in enumerate(processes):
        p.pid = i + 1
    
    return processes