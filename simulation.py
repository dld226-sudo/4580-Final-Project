import random
import numpy as np
import pandas as pd

class Query:
    def __init__(self, prompt_length: int, token_budget: int, arrival_time: float) -> None:
        if not isinstance(prompt_length, int) or prompt_length <= 0:
            raise ValueError("prompt_length must be a positive integer")
        if not isinstance(token_budget, int) or token_budget <= 0:
            raise ValueError("token_budget must be a positive integer")

        self.prompt_length: int = prompt_length
        self.token_budget: int = token_budget
        self.arrival_time: float = arrival_time
        self.finish_time: float = 0.0

        # Time until prefill finishes
        self.time_to_first_token: float = 0.0
        # List of time intervals between each token generation
        # The prefill phase is ignored
        self.time_between_tokens: list[float] = []

        self.prefilled: int = 0
        self.decoded: int = 0
        # Number of prefill tokens currently being processed
        self.prefilling_in_progress: int = 0

    def __repr__(self) -> str:
        return f"Query(prompt_length={self.prompt_length}, token_budget={self.token_budget})"

    @staticmethod
    def generate_random_query(arrival_time: float, min_prompt_length: int = 50, max_prompt_length: int = 100,
                              min_budget: int = 1, max_budget: int = 32) -> 'Query':
        """
        Generates a new Query object with random prompt_length and a geometrically distributed token_budget
        with a maximum of max_budget tokens and an offset of min_budget - 1.
        """
        prompt_length = random.randint(min_prompt_length, max_prompt_length)

        # Generate geometrically distributed token_budget
        p_success = 0.1 # Probability of success in each trial.

        token_budget_val = min_budget - 1 + np.random.geometric(p_success)

        # Ensure the budget does not exceed the maximum
        token_budget_val = min(token_budget_val, max_budget)

        return Query(prompt_length, token_budget_val, arrival_time)

    def prefill(self, num_tokens: int, cur_time: float) -> None:
        """
        Prefill num_tokens tokens from the prompt.
        
        :param num_tokens: Number of tokens to prefill.
        :type num_tokens: int
        :param cur_time: Current time in simulation.
        :type cur_time: float
        """
        if num_tokens > self.prefilling_in_progress:
            raise ValueError("num_tokens must be less than self.prefilling_in_progress")

        self.prefilled += num_tokens
        self.prefilling_in_progress -= num_tokens

        # Update TTFT if finished with prefill
        if self.prefilled == self.prompt_length:
            self.time_to_first_token = cur_time - self.arrival_time
        
        if self.prefilled > self.prompt_length:
            raise RuntimeError("Number of prefilled tokens exceeded prompt length")

    def decode(self, cur_time: float) -> None:
        """
        Decode next token in response.
        
        :param cur_time: Current time in simulation.
        :type cur_time: float
        """
        if self.prefilled != self.prompt_length:
            raise RuntimeError("Cannot decode until prefill phase is complete")

        self.decoded += 1

        # Update TBT
        self.time_between_tokens.append(cur_time - self.arrival_time - self.time_to_first_token - sum(self.time_between_tokens))
        # Update finish time if last token
        if self.decoded == self.token_budget:
            self.finish_time = cur_time
        
        if self.decoded > self.token_budget:
            raise RuntimeError("Number of decoded tokens exceeded token budget")

    def is_complete(self) -> bool:
        """
        Check whether the query is fully processed.
        
        :return: True if query is complete; False otherwise.
        :rtype: bool
        """
        return (self.prefilled == self.prompt_length) and (self.decoded == self.token_budget)

class Worker:
    def __init__(self, worker_id: int, capacity: int, mean_marginal_cost: float, mean_fixed_cost: float, min_batch_threshold: int) -> None:
        self.worker_id: int = worker_id
        # Maximum number of tokens that can be processed in one batch
        self.capacity: int = capacity
        # Total number of tokens to be processed
        self.total_tokens: int = 0
        # Whether worker is currently processing a batch
        self.processing: bool = False
        # List of dicts: {query_object, phase, tokens_planned}
        self.processing_queries: list[dict] = []
        self.mean_marginal_cost: float = mean_marginal_cost
        self.mean_fixed_cost: float = mean_fixed_cost
        self.min_batch_threshold: int = min_batch_threshold

    def __repr__(self) -> str:
        return f"Worker(worker_id={self.worker_id}, active_queries={len(self.processing_queries)})"

    def add_query(self, query: 'Query', phase: str, tokens_planned: int = 1) -> None:
        """
        Add a query to be processed by worker.
        
        :param query: The query to be processed.
        :type query: 'Query'
        :param phase: The processing phase; must be 'prefill' or 'decode'.
        :type phase: str
        :param tokens_planned: The number of tokens to process.
        :type tokens_planned: int
        """
        if phase not in ['prefill', 'decode']:
            raise ValueError("Phase must be 'prefill' or 'decode'")
        if tokens_planned <= 0:
            raise ValueError("tokens_planned must be a positive integer")
        if tokens_planned + self.total_tokens > self.capacity:
            raise ValueError("tokens_planned exceeds worker capacity")
        if self.processing:
            raise RuntimeError("Cannot add query to worker that is processing")
        
        # Update number of prefill tokens in progress
        if phase == 'prefill':
            query.prefilling_in_progress += tokens_planned

        self.total_tokens += tokens_planned
        self.processing_queries.append({'query': query, 'phase': phase, 'tokens_planned': tokens_planned})

    def start_processing(self) -> float:
        """
        Starts processing the current batch.
        
        :returns: The processing time for the current batch.
        :rtype: float
        """
        if self.processing:
            raise RuntimeError("Worker is already processing")
        
        self.processing = True

        a = np.random.exponential(scale=self.mean_marginal_cost)
        c = np.random.exponential(scale=self.mean_fixed_cost)
        processing_time = c + a * max(0, self.total_tokens - self.min_batch_threshold)
        return processing_time

    def finish_processing(self, cur_time: float) -> list['Query']:
        """
        Finish processing all pending queries in current batch.
        
        :param cur_time: Current time in simulation.
        :type cur_time: float
        :return: List of queries that were processed.
        :rtype: list[Query]
        """
        processed_queries = []

        for task_info in self.processing_queries:
            query = task_info['query']
            phase = task_info['phase']
            tokens_planned = task_info['tokens_planned']

            if phase == 'prefill':
                query.prefill(num_tokens=tokens_planned, cur_time=cur_time)
            elif phase == 'decode':
                query.decode(cur_time=cur_time)
            # else phase is invalid

            processed_queries.append(query)

        self.processing = False
        self.total_tokens = 0
        self.processing_queries = []

        return processed_queries

    def get_queries_in_phase(self, phase: str) -> list['Query']:
        return [q['query'] for q in self.processing_queries if q['phase'] == phase]

    def get_prefill_info(self) -> list[dict]:
        return [{'query': q['query'], 'tokens_planned': q['tokens_planned']}
                for q in self.processing_queries if q['phase'] == 'prefill']

class Scheduler:
    def __init__(self, num_workers: int, mean_marginal_cost: float, mean_fixed_cost: float, 
                 min_batch_threshold: int, max_batch_size: int, query_arrival_rate: float) -> None:
        self.completed_queries: list['Query'] = []
        self.workers: list['Worker'] = [
            Worker(i, max_batch_size, mean_marginal_cost, mean_fixed_cost, min_batch_threshold) 
            for i in range(num_workers)]
        self.query_arrival_rate: float = query_arrival_rate
        # List to hold incoming queries
        self.query_queue: list['Query'] = []
        self.clock: float = 0.0
        self.next_query_arrival_time: float = 0.0
        # Initialize worker_completion_times for all workers to float('inf')
        self.worker_completion_times: list[float] = [float('inf') for _ in range(num_workers)]

    def __repr__(self) -> str:
        return (f"Scheduler(num_workers={len(self.workers)}, "
                f"queries_in_queue={len(self.query_queue)}, "
                f"clock={self.clock:.2f})")

    def _handle_query_arrival(self) -> None:
        """
        Generate a new query, update the next query arrival time, and assign tasks to idle workers.
        """
        new_query = Query.generate_random_query(arrival_time=self.next_query_arrival_time)
        self.query_queue.append(new_query)

        self.next_query_arrival_time = self.clock + np.random.exponential(scale=1/self.query_arrival_rate)

        self._distribute_tasks()

    def _assign_task_to_worker(self, worker: 'Worker') -> None:
        """
        Attempts to assign a query from the queue to the given worker.
        
        :param worker: Worker to assign query to.
        :type worker: 'Worker'
        """
        # Cannot assign new tasks to a worker that is processing
        if worker.processing:
            return
        
        # If no pending queries, leave worker idle
        if not self.query_queue:
            self.worker_completion_times[worker.worker_id] = float('inf')
            return

        current_queue_state = list(self.query_queue)
        for i, current_query in enumerate(current_queue_state):
            if current_query.prefilled + current_query.prefilling_in_progress < current_query.prompt_length:
                # Query needs prefilling
                # prompt_length should always be smaller than max batch size
                tokens_to_prefill = current_query.prompt_length
                worker.add_query(current_query, 'prefill', tokens_to_prefill)
                duration = worker.start_processing()
                self.worker_completion_times[worker.worker_id] = self.clock + duration
                self.query_queue.remove(current_query) # Remove the assigned query from the queue
                return
            elif current_query.decoded < current_query.token_budget:
                # Query needs decoding (and prefilling is complete)
                worker.add_query(current_query, 'decode')
                duration = worker.start_processing()
                self.worker_completion_times[worker.worker_id] = self.clock + duration
                self.query_queue.remove(current_query) # Remove the assigned query from the queue
                return

        if not worker.processing_queries:
            self.worker_completion_times[worker.worker_id] = float('inf') # No suitable query found


    def _distribute_tasks(self) -> None:
        """
        Iterate through workers and assign tasks to idle ones.
        """
        for worker in self.workers:
            if not(worker.processing):
                self._assign_task_to_worker(worker)

    def _handle_worker_completion(self, worker: 'Worker') -> None:
        """
         Process completed tasks for a worker, update query states, and reassign new tasks.
        
        :param worker: Worker to finish processing.
        :type worker: 'Worker'
        """
        processed_queries = worker.finish_processing(self.clock)

        for query in processed_queries:
            # Check if the query still needs further processing or is fully complete
            if not(query.is_complete()):
                self.query_queue.insert(0, query) # Add back to front of queue for further processing
            else:
                self.completed_queries.append(query)

        # Assign a new task to this now-free worker if there are queries in the queue
        self._assign_task_to_worker(worker)

    def simulate_event(self) -> None:
        """
        Simulate next event in simulation.
        """
        # Find the next event time
        next_worker_completion = min(self.worker_completion_times)
        next_event_time = min(self.next_query_arrival_time, next_worker_completion)

        # Advance the scheduler's clock
        self.clock = next_event_time

        # Check for query arrival
        if self.clock == self.next_query_arrival_time:
            self._handle_query_arrival()

        # Check for worker completions
        for worker_id, worker in enumerate(self.workers):
            if self.clock == self.worker_completion_times[worker_id]:
                self._handle_worker_completion(worker)


class ChunkedScheduler(Scheduler):
    def __init__(self, num_workers: int, mean_marginal_cost: float, mean_fixed_cost: float,
                 min_batch_threshold: int, max_batch_size: int, query_arrival_rate: float,
                 chunk_size: int) -> None:
        super().__init__(num_workers, mean_marginal_cost, mean_fixed_cost,
                         min_batch_threshold, max_batch_size, query_arrival_rate)
        self.chunk_size: int = chunk_size

    def _distribute_tasks(self) -> None:
        """
        Iterate through workers and assign tasks to idle ones. Prioritize meeting minimum
        batch threshold for all workers.
        """
        # First time through, try to reach min_batch_threshold
        for worker in self.workers:
            if not(worker.processing):
                self._assign_task_with_threshold(worker, worker.min_batch_threshold)
        
        # Second time through, distribute additional tasks
        for worker in self.workers:
            if not(worker.processing):
                self._assign_task_with_threshold(worker, worker.capacity)
                # Start batch if tasks have been assigned
                if worker.processing_queries:
                    duration = worker.start_processing()
                    self.worker_completion_times[worker.worker_id] = self.clock + duration
                else:
                    self.worker_completion_times[worker.worker_id] = float('inf')

    def _assign_task_with_threshold(self, worker: 'Worker', threshold: int) -> None:
        """
        Assign tasks to worker until reaching token threshold.
        
        :param worker: Worker to be assigned tasks.
        :type worker: 'Worker'
        :param threshold: Threshold for number of tokens to accept.
        :type threshold: int
        """
        if worker.processing:
            return

        if not self.query_queue:
            return

        current_queue_state = list(self.query_queue)

        for current_query in current_queue_state:
            # Continue adding chunks until reaching threshold
            remaining_prefill = current_query.prompt_length - (
                current_query.prefilled + current_query.prefilling_in_progress
            )
            while remaining_prefill > 0:
                tokens_to_prefill = min(self.chunk_size, remaining_prefill)
                # Stop adding chunks if threshold will be exceeded
                if worker.total_tokens + tokens_to_prefill > threshold:
                    break
                
                worker.add_query(current_query, 'prefill', tokens_to_prefill)

                remaining_prefill = current_query.prompt_length - (
                current_query.prefilled + current_query.prefilling_in_progress
            )

            if current_query.prefilled == current_query.prompt_length and current_query.decoded < current_query.token_budget:
                # Stop adding tokens if threshold will be exceeded
                if worker.total_tokens + 1 > threshold:
                    break

                worker.add_query(current_query, 'decode')

                self.query_queue.remove(current_query) # Remove the assigned query from the queue

    def _handle_worker_completion(self, worker: 'Worker') -> None:
        """
        Finish processing tasks for a worker, update query states, and reassign new tasks.
        
        :param worker: Worker that is finishing batch.
        :type worker: 'Worker'
        """
        processed_queries = worker.finish_processing(self.clock)

        for query in processed_queries:
            if not query.is_complete():
                # Only need to reinsert into queue if in decode phase
                if query.decoded > 0:
                    self.query_queue.insert(0, query)
            else:
                self.completed_queries.append(query)

        self._distribute_tasks()


def main():
    # User defined constants for simulation
    # mean_marginal_cost = float(input("Mean Marginal Batch Cost (ms/token): "))
    # mean_fixed_cost = float(input("Mean Fixed Batch Cost (ms): "))
    # min_batch_threshold = int(input("Minimum Batch Threshold (tokens): "))
    # max_batch_size = int(input("Maximum Batch Size (tokens): "))
    # query_arrival_rate = float(input("Query Arrival Rate (Queries/s): "))/1000
    # num_workers = int(input("Number of GPU Workers: "))
    # chunk_size = int(input("Chunk Size: "))
    # num_events_to_simulate = int(input("Events per Trial: "))
    # replications = int(input("Number of Trials: "))
    mean_marginal_cost = 0.3
    mean_fixed_cost = 45.5
    min_batch_threshold = 64
    max_batch_size = 200
    query_arrival_rate = 1/1000
    num_workers = 10
    chunk_size = 10

    replications = 100
    num_events_to_simulate = 30000

    df_data = []
    chunked_df_data = []
    for r in range(replications):
        if r % 10 == 0:
            print("Progress:", r/replications)
        basic_scheduler = Scheduler(num_workers, mean_marginal_cost, mean_fixed_cost, min_batch_threshold, max_batch_size, query_arrival_rate)
        chunked_scheduler = ChunkedScheduler(num_workers, mean_marginal_cost, mean_fixed_cost, min_batch_threshold, max_batch_size, query_arrival_rate,chunk_size)

        for _ in range(num_events_to_simulate):
            basic_scheduler.simulate_event()
            chunked_scheduler.simulate_event()

        queries = basic_scheduler.completed_queries
        chunked_queries = basic_scheduler.completed_queries

        # Prepare data for DataFrame
        for query in queries:
            # Calculate average TBT if TBT list is not empty, otherwise 0 or NaN
            avg_tbt = np.mean(query.time_between_tokens) if query.time_between_tokens else 0.0
            df_data.append({
                'replication': r,
                'arrival_time': query.arrival_time,
                'finish_time': query.finish_time,
                'TTFT': query.time_to_first_token,
                'TBT_average': avg_tbt # Store the average TBT
            })

        # Prepare data for DataFrame
        for query in chunked_queries:
            # Calculate average TBT if TBT list is not empty, otherwise 0 or NaN
            avg_tbt = np.mean(query.time_between_tokens) if query.time_between_tokens else 0.0
            chunked_df_data.append({
                'replication': r,
                'arrival_time': query.arrival_time,
                'finish_time': query.finish_time,
                'TTFT': query.time_to_first_token,
                'TBT_average': avg_tbt # Store the average TBT
            })

    # Create DataFrame
    basic_scheduler_df = pd.DataFrame(df_data)
    chunked_scheduler_df = pd.DataFrame(chunked_df_data)

    # Save to CSV
    csv_filename = "basic_scheduler_metrics.csv"
    basic_scheduler_df.to_csv(csv_filename, index=False)

    csv_filename = "chunked_scheduler_metrics.csv"
    chunked_scheduler_df.to_csv(csv_filename, index=False)

if __name__ == "__main__":
    main()