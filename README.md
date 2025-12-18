# 4580-Final-Project: Project Title: Modeling LLM-Query Serving Systems
## Authors: Darcy del Real, Alex Gardocki, Joonseok Jung, Rishi Kumar, Anurag Yadav
## Description: 

This project simulates GPU worker scheduling strategies for Large Language Model  workloads. It models the two-phase token processing pipeline—prefill (processing input prompts) and decode (generating output tokens)—and compares two scheduling approaches:
1) Basic Scheduler: Assigns one query at a time to idle workers
2) Chunked Scheduler: Batches multiple queries together and processes prefill operations in configurable chunks to maximize GPU utilization

### Key Features
1) Realistic cost modeling: Processing time includes fixed overhead plus marginal costs that scale with batch size
2) Configurable parameters: Adjust GPU capacity, batch thresholds, arrival rates, and chunk sizes
3) Performance metrics: Tracks Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT) for each query
4) Multi-worker simulation: Models parallel processing across multiple GPU workers
Statistical analysis: Runs multiple replications to capture variance in performance

This simulation helps understand trade-offs between latency, throughput, and resource utilization in LLM serving systems.

---
## Installation: 
Prerequisites: Python 3.8 or higher, pip package manager

1) Clone or download the simulation.py file to your local machine
2) Navigate to the directory containing the file:
3) Run the simulation using : python simulation.py
The simulation will display progress updates every 10% completion and generate CSV output files when finished.

---
##  Usage: 
The simulation runs with the following parameters:

1) pythonmean_marginal_cost        # Cost per token (ms)
2) mean_fixed_cost                 # Fixed batch overhead (ms)
3) min_batch_threshold             # Minimum tokens for efficient batching
4) max_batch_size                  # Maximum tokens per batch
5) query_arrival_rate              # Queries per millisecond (1 per second)
6) num_workers                     # Number of GPU workers
7) chunk_size                      # Prefill chunk size (ChunkedScheduler only)
8) replications                    # Number of simulation trials
9) num_events_to_simulate          # Events per trial

Customizing Parameters:
To modify simulation parameters, input the values in the when prompted in running of simulation.py.

## Understanding the Simulation
Query Generation: Queries arrive according to a Poisson process with randomly generated:
1) Prompt lengths (50-100 tokens)
2) Token budgets (geometrically distributed, 1-32 tokens)

Processing Phases:
1) Prefill: Process entire prompt in parallel
2) Decode: Generate output tokens one at a time

Scheduling Strategies:
1) Basic Scheduler: Simple FIFO assignment to idle workers
2) Chunked Scheduler: Batches queries and processes prefill in chunks to maintain high GPU utilization

## Outputs

The simulation generates two CSV files: 
1) basic_scheduler_metrics.csv
2) chunked_scheduler_metrics.csv

With performance metrics:
1) replication: Trial number (0 to replications-1)
2) arrival_time: When the query entered the system (ms)
3) finish_time: When the query completed processing (ms)
4) TTFT: Time-to-First-Token: latency until first output token (ms)
5) TBT_average: Average Time-Between-Tokens during decoding (ms)

---
