# Function to calculate average metrics from a JSON file given its file path
import json
from collections import defaultdict
# Function to calculate average metrics from a JSON file given its file path
def calculate_average_metrics(results_data):
    # Initializing a dictionary to store the sum and count for each metric type
    metrics_summary = defaultdict(lambda: {'total_sum': 0, 'total_count': 0})

    # Iterating through the tests and their metrics to calculate the sum and count for each metric type
    for test, metrics in results_data.items():
        for metric, value in metrics.items():
            metrics_summary[metric]['total_sum'] += value
            metrics_summary[metric]['total_count'] += 1

    # Calculating the average for each metric type
    average_metrics_by_type = {}
    for metric, summary in metrics_summary.items():
        if summary['total_count'] != 0:
            average_metrics_by_type[metric] = summary['total_sum'] / summary['total_count']
        else:
            average_metrics_by_type[metric] = "No data available for this metric"

    return average_metrics_by_type


def calculate_average_metrics_from_file(file_path):
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        # Extract the "results" section
        results_data = json_data.get('results', {})

        # Calculate and return the average metrics
        return calculate_average_metrics(results_data)

    except Exception as e:
        return {'error': str(e)}

# Testing the function with one of the provided files
test_file_path = 'eval/llama-2-7b.json'
calculate_average_metrics_from_file(test_file_path)
