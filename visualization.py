import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os

def plot_gantt(gantt: List[tuple], output_file: str = 'gantt.png', 
               max_processes: int = 30, figsize: tuple = (12, 6)):
    """
    Plot Gantt chart for multiple CPUs.
    
    Args:
        gantt: List of (start, end, pid, cpu_id)
        output_file: Output file path
        max_processes: Maximum number of unique processes to show
        figsize: Figure size
    """
    if not gantt:
        print("No Gantt data to plot")
        return
    
    # Sort Gantt entries
    gantt = sorted(gantt, key=lambda x: (x[0], x[3]))
    
    # Get unique CPUs and processes
    cpu_ids = sorted(set(entry[3] for entry in gantt))
    pids = sorted(set(entry[2] for entry in gantt))
    
    # Limit number of processes for readability
    if len(pids) > max_processes:
        # Show first max_processes/2 and last max_processes/2
        show_pids = pids[:max_processes//2] + pids[-max_processes//2:]
        gantt = [entry for entry in gantt if entry[2] in show_pids]
        pids = show_pids
    
    # Create mapping from pid to y-position
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    
    fig, axes = plt.subplots(len(cpu_ids), 1, figsize=figsize, 
                           sharex=True, squeeze=False)
    axes = axes.flatten()
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(pids)))
    
    for cpu_idx, cpu_id in enumerate(cpu_ids):
        ax = axes[cpu_idx]
        cpu_entries = [entry for entry in gantt if entry[3] == cpu_id]
        
        for start, end, pid, _ in cpu_entries:
            color_idx = pid_to_idx[pid] % len(colors)
            ax.barh(0, end - start, left=start, height=0.6, 
                   color=colors[color_idx], edgecolor='black')
            
            # Add process label
            if (end - start) > 5:  # Only label if bar is wide enough
                ax.text(start + (end - start) / 2, 0, f'P{pid}', 
                       va='center', ha='center', fontsize=8)
        
        ax.set_ylabel(f'CPU {cpu_id}')
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Gantt Chart - CPU Timeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def plot_histogram_metrics(processes: List[Any], output_file: str = 'histograms.png'):
    """
    Plot histograms of key metrics.
    
    Args:
        processes: List of completed processes
        output_file: Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Turnaround time distribution
    turnaround_times = [p.turnaround_time for p in processes]
    axes[0].hist(turnaround_times, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_title('Turnaround Time Distribution')
    axes[0].set_xlabel('Turnaround Time')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(turnaround_times), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(turnaround_times):.1f}')
    axes[0].legend()
    
    # Waiting time distribution
    waiting_times = [p.waiting_time for p in processes]
    axes[1].hist(waiting_times, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Waiting Time Distribution')
    axes[1].set_xlabel('Waiting Time')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(waiting_times), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(waiting_times):.1f}')
    axes[1].legend()
    
    # Response time distribution
    response_times = [p.response_time for p in processes if p.response_time >= 0]
    if response_times:
        axes[2].hist(response_times, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[2].set_title('Response Time Distribution')
        axes[2].set_xlabel('Response Time')
        axes[2].set_ylabel('Frequency')
        axes[2].axvline(np.mean(response_times), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(response_times):.1f}')
        axes[2].legend()
    
    # CPU vs IO burst ratio
    cpu_io_ratios = [p.cpu_burst / max(p.io_burst, 1) for p in processes]
    axes[3].hist(cpu_io_ratios, bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[3].set_title('CPU/IO Burst Ratio Distribution')
    axes[3].set_xlabel('CPU/IO Ratio')
    axes[3].set_ylabel('Frequency')
    axes[3].axvline(np.mean(cpu_io_ratios), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(cpu_io_ratios):.1f}')
    axes[3].legend()
    
    plt.suptitle('Process Metrics Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def plot_timeline(processes: List[Any], output_file: str = 'process_timeline.png'):
    """
    Plot process lifecycle timeline.
    
    Args:
        processes: List of completed processes
        output_file: Output file path
    """
    if not processes:
        print("No processes to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort processes by arrival time
    processes = sorted(processes, key=lambda p: p.arrival_time)
    
    y_positions = np.arange(len(processes))
    
    for i, p in enumerate(processes):
        # Arrival to start (waiting)
        if p.start_time > p.arrival_time:
            ax.plot([p.arrival_time, p.start_time], [i, i], 
                   'k--', alpha=0.5, linewidth=1)
        
        # Running period
        ax.plot([p.start_time, p.completion_time - p.io_burst], [i, i], 
               'b-', linewidth=3, label='Running' if i == 0 else "")
        
        # IO period (if any)
        if p.io_burst > 0:
            io_start = p.completion_time - p.io_burst
            ax.plot([io_start, p.completion_time], [i, i], 
                   'g-', linewidth=3, label='IO' if i == 0 else "")
        
        # Mark deadline if exists
        if hasattr(p, 'deadline') and p.deadline:
            ax.axvline(p.deadline, color='r', alpha=0.3, linestyle=':')
            if hasattr(p, 'missed_deadline') and p.missed_deadline:
                ax.text(p.deadline, i, '✗', color='red', fontsize=12, 
                       ha='center', va='center')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'P{p.pid}' for p in processes])
    ax.set_xlabel('Time')
    ax.set_title('Process Lifecycle Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='b', lw=3),
                    Line2D([0], [0], color='g', lw=3),
                    Line2D([0], [0], color='k', linestyle='--', lw=1),
                    Line2D([0], [0], color='r', linestyle=':', lw=2)]
    
    ax.legend(custom_lines, ['Running', 'IO', 'Waiting', 'Deadline'], 
              loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()