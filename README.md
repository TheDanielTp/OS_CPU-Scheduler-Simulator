# CPU Scheduling Algorithm Simulator

A comprehensive CPU scheduling simulator with multiple algorithms, visualization tools, and performance analysis capabilities.

## Features

### Supported Scheduling Algorithms

- FCFS (First-Come, First-Served) - Non-preemptive

- SJF (Shortest Job First) - Non-preemptive

- SRTF (Shortest Remaining Time First) - Preemptive

- RR (Round Robin) - Time quantum based

- PRIORITY - Non-preemptive priority scheduling

- PRIORITY_PRE - Preemptive priority scheduling

- MLFQ (Multi-Level Feedback Queue) - 3-level adaptive scheduling

- CFS (Completely Fair Scheduler) - Linux-style fair scheduling

- EDF (Earliest Deadline First) - Real-time scheduling

## Key Features

- Interactive Terminal Menu - Easy-to-use interface for configuration

- Multiple Workload Types - Synthetic and trace-based workloads

- Multi-CPU Support - Simulate systems with multiple processors

- Comprehensive Metrics - Turnaround time, waiting time, response time, CPU utilization, throughput, fairness index

- Visualization Tools - Gantt charts, timeline diagrams, histogram distributions

- Algorithm Comparison - Side-by-side performance comparison

- Deadline Tracking - For EDF algorithm with miss detection

- Context Switch Simulation - Realistic overhead modeling

- Error Handling - Robust error handling and validation

## Project Structure

```bash
cpu_scheduler/
├── main.py              # Main program with interactive menu
├── simulator.py         # Core simulation engine
├── process.py          # Process class and definitions
├── process_generator.py # Workload generation utilities
├── metrics.py          # Performance metric calculations
├── visualization.py    # Plotting and visualization tools
└── README.md           # This documentation
```

## Installation

### Requirements

- Python 3.7+

- Required packages: matplotlib, numpy

### Quick Setup

```bash
# Clone or download the project
# Navigate to project directory
cd cpu_scheduler

# Install dependencies
pip install matplotlib numpy

# Run the simulator
python main.py
```

## Usage

### Interactive Mode (Recommended)

Run without arguments to start the interactive menu:

```bash
python main.py
```

Menu Options:

1- Run Simulation with Default Parameters - Quick test with FCFS and 100 processes

2- Configure Simulation Parameters - Full customization of all parameters

3- Compare Multiple Algorithms - Run multiple algorithms on same workload

4- Load Process Trace from File - Use CSV trace files

5- Generate Custom Workload - Create and save custom workloads

6- View Help & Documentation - Detailed algorithm explanations

7- Exit - Quit the program

### Command-Line Mode

For scripting and automation, use command-line arguments:

```bash
# Basic usage
python main.py --algorithm FCFS --num_processes 100

# Round Robin with custom quantum
python main.py --algorithm RR --quantum 20 --num_processes 50 --plot_gantt

# EDF with deadline tracking
python main.py --algorithm EDF --num_processes 200

# Multi-CPU simulation
python main.py --algorithm SJF --num_cpus 4 --num_processes 300

# Load from trace file
python main.py --algorithm CFS --workload trace --trace_file my_trace.csv
```

#### Full Command-Line Options

```bash
python main.py --help

Options:
  --algorithm {FCFS,SJF,SRTF,RR,PRIORITY,PRIORITY_PRE,MLFQ,CFS,EDF}
                        Scheduling algorithm to use
  --workload {synthetic,trace}
                        Type of workload to simulate [default: synthetic]
  --trace_file TRACE_FILE
                        Path to trace file (required for trace workload)
  --save_trace SAVE_TRACE
                        Save generated synthetic processes to file
  --num_processes NUM_PROCESSES
                        Number of processes to generate [default: 100]
  --quantum QUANTUM     Time quantum for RR, CFS, and MLFQ algorithms [default: 20]
  --num_cpus NUM_CPUS   Number of CPUs for parallel execution [default: 1]
  --context_switch CONTEXT_SWITCH
                        Context switch overhead time [default: 2]
  --output_dir OUTPUT_DIR
                        Directory to save results [default: results]
  --plot_gantt          Generate Gantt chart visualization
  --max_gantt_processes MAX_GANTT_PROCESSES
                        Maximum number of processes to show in Gantt chart [default: 30]
  --quiet               Suppress detailed output
```

## Algorithm Details

### FCFS (First-Come, First-Served)

- Type: Non-preemptive

- Description: Processes are executed in order of arrival

- Pros: Simple to implement, no starvation

- Cons: Convoy effect, poor for short processes

### SJF (Shortest Job First)

- Type: Non-preemptive

- Description: Executes process with shortest CPU burst next

- Pros: Minimizes average waiting time (optimal)

- Cons: Requires knowledge of burst times, starvation possible

### SRTF (Shortest Remaining Time First)

- Type: Preemptive version of SJF

- Description: Always runs process with shortest remaining time

- Pros: Better response time than SJF

- Cons: Higher overhead, starvation possible

### RR (Round Robin)

- Type: Preemptive with time quantum

- Description: Each process gets equal time slices

- Pros: Good for time-sharing, no starvation

- Cons: Performance depends on quantum size

### PRIORITY Scheduling

- Type: Non-preemptive and preemptive variants

- Description: Higher priority processes execute first

- Pros: Important processes get preference

- Cons: Starvation of low-priority processes

### MLFQ (Multi-Level Feedback Queue)

- Type: Adaptive multi-level

- Description: 3 queues with different priorities and time quanta

- Configuration: Queue 1 (quantum=8), Queue 2 (quantum=16), Queue 3 (FCFS)

- Pros: Adapts to process behavior, good for mixed workloads

- Cons: Complex implementation

### CFS (Completely Fair Scheduler)

- Type: Weighted fair scheduling

- Description: Linux-style scheduler using virtual runtime

- Pros: High fairness, good for interactive systems

- Cons: More complex calculations

### EDF (Earliest Deadline First)

- Type: Real-time scheduling

- Description: Executes process with earliest deadline first

- Pros: Optimal for meeting deadlines

- Cons: Requires deadline knowledge, can miss deadlines under overload

## Workload Generation

### Synthetic Workloads

The simulator can generate synthetic processes with various characteristics:

- Exponential distribution (default) - Realistic for many systems

- Uniform distribution - Evenly distributed burst times

- Bimodal distribution - Mix of short and long processes

- Pareto distribution - Heavy-tailed distribution

- Mixed workload - Interactive, batch, and real-time mix

### Trace Files

Load pre-defined process traces from CSV files:

```text
# Format: pid,arrival_time,cpu_burst,io_burst,priority,deadline
1,0,50,10,5,100
2,10,30,5,3,80
3,20,100,20,8,200
```

## Metrics Calculated

### Core Metrics

- Turnaround Time: Completion time - Arrival time

- Waiting Time: Turnaround time - (CPU burst + IO burst)

- Response Time: First execution time - Arrival time

- CPU Utilization: Percentage of time CPU was busy

- Throughput: Processes completed per time unit

- Fairness Index: Jain's fairness index (0-1, higher is fairer)

### Advanced Metrics

- Standard Deviation: Variability of turnaround/waiting times

- Slowdown: Turnaround time divided by service time

- Deadline Miss Rate: For EDF algorithm

- Tardiness: How much deadlines were missed

- Priority Inversion Risk: For priority scheduling

## Output and Visualization

### Console Output

```text
SIMULATION RESULTS: RR
============================================================
Processes: 100/100 completed
Total simulation time: 1456.00
CPU Utilization: 92.34%
Throughput: 0.0687 processes/time unit
Avg Turnaround: 728.32
Avg Waiting: 678.42
Avg Response: 145.21
Fairness Index: 0.9345
```

### Generated Files

- JSON metrics file: results/{algorithm}_metrics.json

- CSV process details: results/{algorithm}_processes.csv

- Gantt chart: results/{algorithm}_gantt.png (if requested)

- Timeline diagram: results/{algorithm}_timeline.png

- Histograms: results/{algorithm}_histograms.png

### Visualization Examples

1- Gantt Chart: Shows process execution timeline across CPUs

2- Timeline Diagram: Visualizes individual process lifecycles

3- Histograms: Distribution of turnaround, waiting, response times

4- Comparison Charts: Side-by-side algorithm performance

## License

This project is for educational use. Feel free to modify and distribute for academic purposes.

## Contributing

Contributions welcome! Areas for improvement:

- Additional scheduling algorithms

- More visualization options

- Network and distributed scheduling

- Energy-aware scheduling models

- Real-time guarantees analysis