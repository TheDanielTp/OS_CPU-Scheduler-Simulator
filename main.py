import argparse
import json
import os
import sys
import time
from simulator import Simulator
from process_generator import generate_synthetic, load_trace, save_trace, generate_workload_mix
from metrics import calculate_metrics
from visualization import plot_gantt, plot_histogram_metrics, plot_timeline

def run_simulation(algorithm, processes, quantum=None, num_cpus=1, context_switch=2):
    """Run simulation with given parameters."""
    sim = Simulator(
        algorithm, 
        processes, 
        quantum=quantum, 
        num_cpus=num_cpus,
        context_switch=context_switch
    )
    completed, total_time, gantt = sim.run()
    metrics = calculate_metrics(completed, total_time, num_cpus=num_cpus, algorithm=algorithm)
    return metrics, gantt, completed

def save_results(algorithm, metrics, processes, gantt, output_dir="results"):
    """Save simulation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = os.path.join(output_dir, f"{algorithm}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save process completion details
    processes_file = os.path.join(output_dir, f"{algorithm}_processes.csv")
    with open(processes_file, 'w') as f:
        f.write("pid,arrival,cpu_burst,io_burst,priority,completion,turnaround,waiting,response,missed_deadline\n")
        for p in processes:
            f.write(f"{p.pid},{p.arrival_time},{p.cpu_burst},{p.io_burst},{p.original_priority},")
            f.write(f"{p.completion_time},{p.turnaround_time},{p.waiting_time},{p.response_time},")
            f.write(f"{getattr(p, 'missed_deadline', False)}\n")
    
    return metrics_file, processes_file

def print_header():
    """Print a fancy header for the program."""
    print("\n" + "="*60)
    print("        CPU SCHEDULING SIMULATOR")
    print("="*60)

def print_menu():
    """Print the main menu."""
    print("\nMAIN MENU")
    print("-" * 40)
    print("1. Run Simulation with Default Parameters")
    print("2. Configure Simulation Parameters")
    print("3. Compare Multiple Algorithms")
    print("4. Load Process Trace from File")
    print("5. Generate Custom Workload")
    print("6. View Help & Documentation")
    print("7. Exit")
    print("-" * 40)

def get_algorithm_choice():
    """Get algorithm choice from user."""
    algorithms = {
        '1': 'FCFS',
        '2': 'SJF',
        '3': 'SRTF',
        '4': 'RR',
        '5': 'PRIORITY',
        '6': 'PRIORITY_PRE',
        '7': 'MLFQ',
        '8': 'CFS',
        '9': 'EDF'
    }
    
    print("\nSELECT SCHEDULING ALGORITHM")
    print("-" * 40)
    print("1. FCFS (First Come First Served)")
    print("2. SJF (Shortest Job First)")
    print("3. SRTF (Shortest Remaining Time First)")
    print("4. RR (Round Robin)")
    print("5. Priority (Non-preemptive)")
    print("6. Priority (Preemptive)")
    print("7. MLFQ (Multi-Level Feedback Queue)")
    print("8. CFS (Completely Fair Scheduler)")
    print("9. EDF (Earliest Deadline First)")
    print("-" * 40)
    
    while True:
        choice = input("Enter choice (1-9): ").strip()
        if choice in algorithms:
            return algorithms[choice]
        print("Invalid choice. Please enter 1-9.")

def get_workload_choice():
    """Get workload type from user."""
    print("\nSELECT WORKLOAD TYPE")
    print("-" * 40)
    print("1. Synthetic (Default - Exponential distribution)")
    print("2. Synthetic (Uniform distribution)")
    print("3. Synthetic (Bimodal - Mix of short and long)")
    print("4. Synthetic (Real-time mixed workload)")
    print("5. Load from trace file")
    print("-" * 40)
    
    choices = {
        '1': ('synthetic', 'exponential'),
        '2': ('synthetic', 'uniform'),
        '3': ('synthetic', 'bimodal'),
        '4': ('synthetic', 'mixed'),
        '5': ('trace', None)
    }
    
    while True:
        choice = input("Enter choice (1-5): ").strip()
        if choice in choices:
            return choices[choice]
        print("Invalid choice. Please enter 1-5.")

def get_integer_input(prompt, default, min_val=1, max_val=10000):
    """Get validated integer input from user."""
    while True:
        try:
            value_str = input(f"{prompt} [{default}]: ").strip()
            if not value_str:
                return default
            value = int(value_str)
            if min_val <= value <= max_val:
                return value
            print(f"Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid integer.")

def get_float_input(prompt, default, min_val=0.0):
    """Get validated float input from user."""
    while True:
        try:
            value_str = input(f"{prompt} [{default}]: ").strip()
            if not value_str:
                return default
            value = float(value_str)
            if value >= min_val:
                return value
            print(f"Value must be at least {min_val}")
        except ValueError:
            print("Please enter a valid number.")

def configure_simulation():
    """Interactive simulation configuration."""
    config = {}
    
    print("\nSIMULATION CONFIGURATION")
    print("-" * 40)
    
    # Algorithm
    config['algorithm'] = get_algorithm_choice()
    
    # Workload
    workload_type, distribution = get_workload_choice()
    config['workload'] = workload_type
    
    if workload_type == 'trace':
        trace_file = input("Enter trace file path: ").strip()
        while not os.path.exists(trace_file):
            print(f"File not found: {trace_file}")
            trace_file = input("Enter trace file path: ").strip()
        config['trace_file'] = trace_file
    else:
        config['num_processes'] = get_integer_input("Number of processes", 100, 1, 1000)
        config['distribution'] = distribution
        
        if distribution != 'mixed':
            config['mean_cpu'] = get_float_input("Mean CPU burst time", 50.0, 1.0)
            config['mean_interarrival'] = get_float_input("Mean inter-arrival time", 20.0, 0.1)
            config['io_probability'] = get_float_input("IO probability (0-1)", 0.3, 0.0)
    
    # Algorithm-specific parameters
    if config['algorithm'] in ['RR', 'CFS', 'MLFQ']:
        config['quantum'] = get_integer_input("Time quantum", 20, 1, 100)
    
    # System parameters
    config['num_cpus'] = get_integer_input("Number of CPUs", 1, 1, 16)
    config['context_switch'] = get_integer_input("Context switch time", 2, 0, 10)
    
    # Output options
    print("\nOUTPUT OPTIONS")
    print("-" * 40)
    config['plot_gantt'] = input("Generate Gantt chart? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
    if config['plot_gantt']:
        config['max_gantt'] = get_integer_input("Max processes in Gantt chart", 30, 1, 100)
    
    config['output_dir'] = input(f"Output directory [results]: ").strip()
    if not config['output_dir']:
        config['output_dir'] = 'results'
    
    config['quiet'] = input("Suppress detailed output? (y/n) [n]: ").strip().lower() in ['y', 'yes']
    
    return config

def compare_algorithms():
    """Compare multiple algorithms with the same workload."""
    print("\nALGORITHM COMPARISON MODE")
    print("-" * 40)
    
    # Configuration
    num_processes = get_integer_input("Number of processes", 100)
    algorithms_input = input("Algorithms to compare (comma-separated, e.g., FCFS,SJF,RR) [FCFS,SJF,RR]: ").strip()
    
    if not algorithms_input:
        algorithms = ['FCFS', 'SJF', 'RR']
    else:
        algorithms = [a.strip().upper() for a in algorithms_input.split(',')]
        valid_algorithms = ['FCFS', 'SJF', 'SRTF', 'RR', 'PRIORITY', 'PRIORITY_PRE', 'MLFQ', 'CFS', 'EDF']
        for algo in algorithms:
            if algo not in valid_algorithms:
                print(f"Invalid algorithm: {algo}")
                return
    
    num_cpus = get_integer_input("Number of CPUs", 1)
    quantum = get_integer_input("Quantum (for RR/CFS/MLFQ)", 20)
    
    # Generate workload
    print("\nGenerating workload...")
    processes = generate_synthetic(num_processes)
    
    # Run simulations
    all_results = []
    
    for algo in algorithms:
        print(f"\nRunning {algo}...")
        
        try:
            metrics, gantt, completed = run_simulation(
                algo,
                processes,
                quantum=quantum if algo in ['RR', 'CFS', 'MLFQ'] else None,
                num_cpus=num_cpus
            )
            
            metrics['algorithm'] = algo
            all_results.append(metrics)
            
            print(f"  ✓ Completed {len(completed)} processes")
            print(f"  Avg Turnaround: {metrics.get('avg_turnaround', 0):.2f}")
            print(f"  Avg Waiting: {metrics.get('avg_waiting', 0):.2f}")
            print(f"  CPU Utilization: {metrics.get('cpu_utilization', 0):.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error running {algo}: {e}")
    
    # Save comparison results
    if all_results:
        os.makedirs("comparisons", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_file = f"comparisons/comparison_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nComparison results saved to {comparison_file}")
        
        # Create comparison table
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*80)
        print(f"{'Algorithm':<12} {'Turnaround':<12} {'Waiting':<12} {'Response':<12} {'CPU %':<10} {'Throughput':<12}")
        print("-"*80)
        
        for result in all_results:
            algo = result.get('algorithm', 'Unknown')
            turnaround = result.get('avg_turnaround', 0)
            waiting = result.get('avg_waiting', 0)
            response = result.get('avg_response', 0)
            cpu_util = result.get('cpu_utilization', 0)
            throughput = result.get('throughput', 0)
            
            print(f"{algo:<12} {turnaround:<12.2f} {waiting:<12.2f} {response:<12.2f} {cpu_util:<10.2f} {throughput:<12.4f}")
        
        # Find best algorithm for each metric
        if len(all_results) > 1:
            print("\n" + "="*80)
            print("BEST PERFORMANCE BY METRIC:")
            print("-"*80)
            
            # Turnaround (lower is better)
            best_turnaround = min(all_results, key=lambda x: x.get('avg_turnaround', float('inf')))
            print(f"Best Turnaround Time: {best_turnaround.get('algorithm')} ({best_turnaround.get('avg_turnaround', 0):.2f})")
            
            # Waiting (lower is better)
            best_waiting = min(all_results, key=lambda x: x.get('avg_waiting', float('inf')))
            print(f"Best Waiting Time: {best_waiting.get('algorithm')} ({best_waiting.get('avg_waiting', 0):.2f})")
            
            # CPU Utilization (higher is better)
            best_cpu = max(all_results, key=lambda x: x.get('cpu_utilization', 0))
            print(f"Best CPU Utilization: {best_cpu.get('algorithm')} ({best_cpu.get('cpu_utilization', 0):.2f}%)")
            
            # Throughput (higher is better)
            best_throughput = max(all_results, key=lambda x: x.get('throughput', 0))
            print(f"Best Throughput: {best_throughput.get('algorithm')} ({best_throughput.get('throughput', 0):.4f})")

def show_help():
    """Display help and documentation."""
    print("\n" + "="*60)
    print("CPU SCHEDULING SIMULATOR - HELP & DOCUMENTATION")
    print("="*60)
    
    print("\nSCHEDULING ALGORITHMS:")
    print("-" * 40)
    print("1. FCFS (First Come First Served)")
    print("   - Processes executed in order of arrival")
    print("   - Simple but can lead to convoy effect")
    
    print("\n2. SJF (Shortest Job First)")
    print("   - Executes process with shortest CPU burst next")
    print("   - Optimal for minimizing waiting time")
    
    print("\n3. SRTF (Shortest Remaining Time First)")
    print("   - Preemptive version of SJF")
    print("   - Can minimize average waiting time further")
    
    print("\n4. RR (Round Robin)")
    print("   - Each process gets a time quantum")
    print("   - Good for time-sharing systems")
    
    print("\n5. Priority Scheduling")
    print("   - Processes with higher priority execute first")
    print("   - Can be preemptive or non-preemptive")
    
    print("\n6. MLFQ (Multi-Level Feedback Queue)")
    print("   - Multiple queues with different priorities")
    print("   - Processes move between queues based on behavior")
    
    print("\n7. CFS (Completely Fair Scheduler)")
    print("   - Linux's default scheduler")
    print("   - Uses virtual runtime for fairness")
    
    print("\n8. EDF (Earliest Deadline First)")
    print("   - Real-time scheduling")
    print("   - Executes process with earliest deadline first")
    
    print("\n\nWORKLOAD TYPES:")
    print("-" * 40)
    print("1. Synthetic: Computer-generated processes")
    print("   - Exponential: Common in real systems")
    print("   - Uniform: Even distribution")
    print("   - Bimodal: Mix of short and long processes")
    print("   - Mixed: Interactive, batch, and real-time mix")
    
    print("\n2. Trace: Load from file")
    print("   - CSV format: pid,arrival,cpu,io,priority,deadline")
    
    print("\n\nKEY METRICS:")
    print("-" * 40)
    print("• Turnaround Time: Completion - Arrival")
    print("• Waiting Time: Turnaround - (CPU + IO)")
    print("• Response Time: First run - Arrival")
    print("• CPU Utilization: % of time CPU busy")
    print("• Throughput: Processes completed per time unit")
    print("• Fairness Index: Jain's fairness index (1 = perfectly fair)")
    
    print("\n\nCOMMAND-LINE USAGE:")
    print("-" * 40)
    print("python main.py --algorithm FCFS --num_processes 100")
    print("python main.py --algorithm RR --quantum 20 --plot_gantt")
    print("python main.py --algorithm EDF --workload trace --trace_file mytrace.csv")
    
    print("\nPress Enter to continue...")
    input()

def interactive_mode():
    """Run the interactive terminal menu."""
    print_header()
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            # Quick run with defaults
            print("\nQUICK SIMULATION WITH DEFAULTS")
            print("-" * 40)
            
            algorithm = get_algorithm_choice()
            processes = generate_synthetic(50)  # Default 50 processes
            
            print(f"\nRunning {algorithm} with 50 processes...")
            
            metrics, gantt, completed = run_simulation(
                algorithm,
                processes,
                quantum=20 if algorithm in ['RR', 'CFS', 'MLFQ'] else None
            )
            
            print(f"\n✓ Simulation completed!")
            print(f"  Processes: {len(completed)}")
            print(f"  Avg Turnaround: {metrics.get('avg_turnaround', 0):.2f}")
            print(f"  Avg Waiting: {metrics.get('avg_waiting', 0):.2f}")
            print(f"  CPU Utilization: {metrics.get('cpu_utilization', 0):.2f}%")
            
            # Ask to save results
            save = input("\nSave results? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
            if save:
                save_results(algorithm, metrics, completed, gantt)
                print("Results saved to 'results' directory")
        
        elif choice == '2':
            # Configure and run
            config = configure_simulation()
            
            print("\n" + "="*60)
            print("STARTING SIMULATION")
            print("="*60)
            
            # Load or generate processes
            if config['workload'] == 'trace':
                processes = load_trace(config['trace_file'])
            elif config.get('distribution') == 'mixed':
                processes = generate_workload_mix(config['num_processes'])
            else:
                processes = generate_synthetic(
                    config['num_processes'],
                    mean_interarrival=config.get('mean_interarrival', 20),
                    mean_cpu=config.get('mean_cpu', 50),
                    distribution=config.get('distribution', 'exponential'),
                    io_probability=config.get('io_probability', 0.3)
                )
            
            print(f"Running {config['algorithm']} with {len(processes)} processes...")
            
            try:
                metrics, gantt, completed = run_simulation(
                    config['algorithm'],
                    processes,
                    quantum=config.get('quantum'),
                    num_cpus=config['num_cpus'],
                    context_switch=config['context_switch']
                )
                
                if not config.get('quiet', False):
                    print(f"\n{'='*60}")
                    print(f"SIMULATION RESULTS: {config['algorithm']}")
                    print(f"{'='*60}")
                    print(f"Processes: {len(completed)}/{len(processes)} completed")
                    print(f"Total simulation time: {metrics.get('total_time', 0):.2f}")
                    print(f"CPU Utilization: {metrics.get('cpu_utilization', 0):.2f}%")
                    print(f"Throughput: {metrics.get('throughput', 0):.4f} processes/time unit")
                    print(f"Avg Turnaround: {metrics.get('avg_turnaround', 0):.2f}")
                    print(f"Avg Waiting: {metrics.get('avg_waiting', 0):.2f}")
                    print(f"Avg Response: {metrics.get('avg_response', 0):.2f}")
                    print(f"Fairness Index: {metrics.get('fairness', 0):.4f}")
                    
                    # Deadline miss warning for EDF
                    if config['algorithm'].upper() == "EDF":
                        misses = metrics.get('deadline_misses', 0)
                        if misses > 0:
                            print(f"\n⚠️  WARNING: {misses} process(es) missed their deadline!")
                            print(f"   Miss rate: {metrics.get('deadline_miss_rate', 0):.2f}%")
                        else:
                            print("\n✅ All processes met their deadlines!")
                
                # Save results
                metrics_file, processes_file = save_results(
                    config['algorithm'], metrics, completed, gantt, config['output_dir']
                )
                
                if not config.get('quiet', False):
                    print(f"\nResults saved to:")
                    print(f"  Metrics: {metrics_file}")
                    print(f"  Process details: {processes_file}")
                
                # Generate plots
                if config.get('plot_gantt', False) and gantt:
                    gantt_file = os.path.join(config['output_dir'], f"{config['algorithm']}_gantt.png")
                    plot_gantt(gantt, output_file=gantt_file, max_processes=config.get('max_gantt', 30))
                    print(f"  Gantt chart: {gantt_file}")
                    
                    # Generate additional visualizations
                    timeline_file = os.path.join(config['output_dir'], f"{config['algorithm']}_timeline.png")
                    plot_timeline(completed[:50], output_file=timeline_file)
                    
                    histogram_file = os.path.join(config['output_dir'], f"{config['algorithm']}_histograms.png")
                    plot_histogram_metrics(completed, output_file=histogram_file)
                
                print("\n✓ Simulation completed successfully!")
                
            except Exception as e:
                print(f"\n✗ Error during simulation: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '3':
            # Compare algorithms
            compare_algorithms()
        
        elif choice == '4':
            # Load from trace file
            print("\nLOAD PROCESS TRACE")
            print("-" * 40)
            trace_file = input("Enter trace file path: ").strip()
            
            if os.path.exists(trace_file):
                try:
                    processes = load_trace(trace_file)
                    print(f"✓ Loaded {len(processes)} processes from {trace_file}")
                    
                    # Show preview
                    print("\nFirst 5 processes:")
                    print("-" * 40)
                    print("PID | Arrival | CPU | IO | Priority | Deadline")
                    print("-" * 40)
                    for p in processes[:5]:
                        deadline = getattr(p, 'deadline', 'N/A')
                        print(f"{p.pid:3} | {p.arrival_time:7} | {p.cpu_burst:3} | {p.io_burst:2} | {p.original_priority:8} | {deadline}")
                    
                    # Ask to run simulation
                    run_now = input("\nRun simulation with these processes? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
                    if run_now:
                        algorithm = get_algorithm_choice()
                        metrics, gantt, completed = run_simulation(algorithm, processes)
                        
                        print(f"\n✓ Simulation completed!")
                        print(f"  Processes: {len(completed)}")
                        print(f"  Avg Turnaround: {metrics.get('avg_turnaround', 0):.2f}")
                        
                        save = input("\nSave results? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
                        if save:
                            save_results(algorithm, metrics, completed, gantt)
                            print("Results saved to 'results' directory")
                
                except Exception as e:
                    print(f"✗ Error loading trace file: {e}")
            else:
                print(f"✗ File not found: {trace_file}")
        
        elif choice == '5':
            # Generate custom workload
            print("\nCUSTOM WORKLOAD GENERATION")
            print("-" * 40)
            
            num_processes = get_integer_input("Number of processes", 100)
            distribution = get_workload_choice()[1]
            
            if distribution == 'mixed':
                processes = generate_workload_mix(num_processes)
            else:
                mean_cpu = get_float_input("Mean CPU burst time", 50.0)
                mean_interarrival = get_float_input("Mean inter-arrival time", 20.0)
                io_prob = get_float_input("IO probability (0-1)", 0.3)
                
                processes = generate_synthetic(
                    num_processes,
                    mean_interarrival=mean_interarrival,
                    mean_cpu=mean_cpu,
                    distribution=distribution,
                    io_probability=io_prob
                )
            
            print(f"\n✓ Generated {len(processes)} processes")
            
            # Save option
            save = input("Save to trace file? (y/n) [n]: ").strip().lower() in ['y', 'yes']
            if save:
                filename = input("Filename [workload.csv]: ").strip()
                if not filename:
                    filename = "workload.csv"
                save_trace(processes, filename)
                print(f"✓ Workload saved to {filename}")
            
            # Preview
            preview = input("Show preview? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
            if preview:
                print("\nFirst 10 processes:")
                print("-" * 60)
                print("PID | Arrival | CPU | IO | Priority | Deadline")
                print("-" * 60)
                for p in processes[:10]:
                    deadline = getattr(p, 'deadline', 'N/A')
                    print(f"{p.pid:3} | {p.arrival_time:7} | {p.cpu_burst:3} | {p.io_burst:2} | {p.original_priority:8} | {deadline}")
        
        elif choice == '6':
            # Help
            show_help()
        
        elif choice == '7':
            # Exit
            print("\nThank you for using CPU Scheduling Simulator!")
            print("Goodbye!\n")
            break
        
        else:
            print("Invalid choice. Please enter 1-7.")
        
        input("\nPress Enter to continue...")

def command_line_mode(args):
    """Run in command-line mode with arguments."""
    # Validate arguments
    if args.workload == 'trace' and not args.trace_file:
        print("Error: --trace_file is required for trace workload")
        return 1
    
    if args.num_cpus < 1:
        print("Error: --num_cpus must be at least 1")
        return 1
    
    if args.quantum <= 0:
        print("Error: --quantum must be positive")
        return 1
    
    if args.context_switch < 0:
        print("Error: --context_switch cannot be negative")
        return 1
    
    # Generate or load processes
    try:
        if args.workload == 'synthetic':
            processes = generate_synthetic(
                args.num_processes,
                mean_interarrival=20,
                mean_cpu=50,
                sigma_cpu=20,
                io_min=10,
                io_max=100,
                pri_min=1,
                pri_max=10
            )
            if args.save_trace:
                save_trace(processes, args.save_trace)
                if not args.quiet:
                    print(f"Saved synthetic trace to {args.save_trace}")
        else:
            processes = load_trace(args.trace_file)
            
        if not processes:
            print("Error: No processes to simulate")
            return 1
            
    except Exception as e:
        print(f"Error loading processes: {e}")
        return 1
    
    # Run simulation
    try:
        metrics, gantt, completed = run_simulation(
            args.algorithm,
            processes,
            quantum=args.quantum if args.algorithm in ['RR', 'CFS', 'MLFQ'] else None,
            num_cpus=args.num_cpus,
            context_switch=args.context_switch
        )
        
        # Save results
        metrics_file, processes_file = save_results(
            args.algorithm, metrics, completed, gantt, args.output_dir
        )
        
        # Print summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"SIMULATION RESULTS: {args.algorithm}")
            print(f"{'='*60}")
            print(f"Processes: {len(completed)}/{len(processes)} completed")
            print(f"Total simulation time: {metrics.get('total_time', 0):.2f}")
            print(f"CPU Utilization: {metrics.get('cpu_utilization', 0):.2f}%")
            print(f"Throughput: {metrics.get('throughput', 0):.4f} processes/time unit")
            print(f"Avg Turnaround: {metrics.get('avg_turnaround', 0):.2f}")
            print(f"Avg Waiting: {metrics.get('avg_waiting', 0):.2f}")
            print(f"Avg Response: {metrics.get('avg_response', 0):.2f}")
            print(f"Fairness Index: {metrics.get('fairness', 0):.4f}")
            
            # Deadline miss warning for EDF
            if args.algorithm.upper() == "EDF":
                misses = metrics.get('deadline_misses', 0)
                if misses > 0:
                    print(f"\n⚠️  WARNING: {misses} process(es) missed their deadline!")
                    print(f"   Miss rate: {metrics.get('deadline_miss_rate', 0):.2f}%")
                else:
                    print("\n✅ All processes met their deadlines!")
            
            print(f"\nResults saved to:")
            print(f"  Metrics: {metrics_file}")
            print(f"  Process details: {processes_file}")
        
        # Generate plots if requested
        if args.plot_gantt and gantt:
            gantt_file = os.path.join(args.output_dir, f"{args.algorithm}_gantt.png")
            plot_gantt(gantt, output_file=gantt_file, max_processes=args.max_gantt_processes)
            if not args.quiet:
                print(f"  Gantt chart: {gantt_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='CPU Scheduler Simulation - Interactive and Command-line Modes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # We'll handle help manually
    )
    
    # Command-line arguments (only used if provided)
    parser.add_argument('--algorithm', 
                        choices=['FCFS', 'SJF', 'SRTF', 'RR', 'PRIORITY', 'PRIORITY_PRE',
                                 'MLFQ', 'CFS', 'EDF'],
                        help='Scheduling algorithm to use')
    
    parser.add_argument('--workload', default='synthetic', 
                       choices=['synthetic', 'trace'],
                       help='Type of workload to simulate')
    parser.add_argument('--trace_file', type=str,
                       help='Path to trace file (required for trace workload)')
    parser.add_argument('--save_trace', type=str,
                       help='Save generated synthetic processes to file')
    
    parser.add_argument('--num_processes', type=int, default=100,
                       help='Number of processes to generate (synthetic only)')
    parser.add_argument('--quantum', type=int, default=20,
                       help='Time quantum for RR, CFS, and MLFQ algorithms')
    parser.add_argument('--num_cpus', type=int, default=1,
                       help='Number of CPUs for parallel execution')
    parser.add_argument('--context_switch', type=int, default=2,
                       help='Context switch overhead time')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--plot_gantt', action='store_true',
                       help='Generate Gantt chart visualization')
    parser.add_argument('--max_gantt_processes', type=int, default=30,
                       help='Maximum number of processes to show in Gantt chart')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    parser.add_argument('--help', '-h', action='store_true',
                       help='Show command-line help and exit')
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help:
        parser.print_help()
        print("\n\nINTERACTIVE MODE:")
        print("  If no arguments are provided, the program will start in interactive mode")
        print("  with a terminal menu for easy configuration.")
        return 0
    
    # Check if any command-line arguments were provided
    # (excluding the script name itself)
    if len(sys.argv) > 1:
        # Run in command-line mode
        return command_line_mode(args)
    else:
        # Run in interactive mode
        try:
            interactive_mode()
            return 0
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            return 0
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())