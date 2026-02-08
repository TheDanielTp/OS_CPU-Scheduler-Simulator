from typing import List, Dict, Any
from process import Process

def calculate_metrics(completed_processes: List[Process], total_time: float, 
                     num_cpus: int = 1, algorithm: str = "") -> Dict[str, Any]:
    """Calculate comprehensive scheduling metrics."""
    if not completed_processes:
        return {'error': 'No processes completed'}
    
    n = len(completed_processes)
    
    # Basic metrics
    turnaround_times = [p.turnaround_time for p in completed_processes]
    waiting_times = [p.waiting_time for p in completed_processes]
    response_times = [p.response_time for p in completed_processes if p.response_time >= 0]
    
    avg_turnaround = sum(turnaround_times) / n if n > 0 else 0
    avg_waiting = sum(waiting_times) / n if n > 0 else 0
    avg_response = sum(response_times) / len(response_times) if response_times else 0
    
    # Throughput and utilization
    cpu_burst_total = sum(p.cpu_burst for p in completed_processes)
    cpu_util = (cpu_burst_total / (total_time * num_cpus) * 100) if total_time > 0 else 0
    throughput = n / total_time if total_time > 0 else 0
    
    # Variability metrics
    import math
    std_turnaround = math.sqrt(sum((t - avg_turnaround) ** 2 for t in turnaround_times) / n) if n > 1 else 0
    std_waiting = math.sqrt(sum((w - avg_waiting) ** 2 for w in waiting_times) / n) if n > 1 else 0
    
    # Jain's Fairness Index
    x = [p.turnaround_time / max(p.cpu_burst, 0.001) for p in completed_processes]
    sum_x = sum(x)
    sum_x2 = sum(xi ** 2 for xi in x)
    fairness = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1
    
    # Bounded slowdown (avoiding division by zero)
    slowdowns = []
    for p in completed_processes:
        service_time = p.cpu_burst + p.io_burst
        if service_time > 0:
            slowdown = p.turnaround_time / service_time
            slowdowns.append(slowdown)
    avg_slowdown = sum(slowdowns) / len(slowdowns) if slowdowns else 0
    
    # Prepare metrics dictionary
    metrics = {
        'total_processes': n,
        'total_time': total_time,
        'avg_turnaround': avg_turnaround,
        'std_turnaround': std_turnaround,
        'avg_waiting': avg_waiting,
        'std_waiting': std_waiting,
        'avg_response': avg_response,
        'cpu_utilization': cpu_util,
        'throughput': throughput,
        'fairness': fairness,
        'avg_slowdown': avg_slowdown,
        'algorithm': algorithm,
        'num_cpus': num_cpus,
    }
    
    # Algorithm-specific metrics
    algorithm = algorithm.upper()
    
    if algorithm == "EDF":
        missed = [p for p in completed_processes if hasattr(p, 'missed_deadline') and p.missed_deadline]
        metrics.update({
            'deadline_misses': len(missed),
            'deadline_miss_rate': (len(missed) / n * 100) if n > 0 else 0,
            'missed_pids': [p.pid for p in missed],
            'tardiness': sum(max(0, p.completion_time - p.deadline) 
                           for p in completed_processes if hasattr(p, 'deadline') and p.deadline)
        })
    
    elif algorithm in ["PRIORITY", "PRIORITY_PRE"]:
        # Priority inversion detection (simplified)
        high_pri_wait = [p.waiting_time for p in completed_processes if p.original_priority <= 3]
        low_pri_wait = [p.waiting_time for p in completed_processes if p.original_priority >= 8]
        
        if high_pri_wait and low_pri_wait:
            metrics['priority_inversion_risk'] = (
                sum(high_pri_wait) / len(high_pri_wait) - 
                sum(low_pri_wait) / len(low_pri_wait)
            )
    
    elif algorithm == "MLFQ":
        # Level distribution
        from collections import Counter
        levels = Counter(p.level for p in completed_processes if hasattr(p, 'level'))
        metrics['mlfq_level_distribution'] = dict(levels)
    
    return metrics