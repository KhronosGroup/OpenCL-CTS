#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

# Terminal Color Codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare PR test results against a golden reference.")
    parser.add_argument("--golden", required=True, help="Path to golden reference JSON")
    parser.add_argument("--results-dir", required=True, help="Directory containing PR test JSON results")
    return parser.parse_args()

def load_golden_reference(filepath):
    """Load and return the golden reference JSON data."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load golden reference '{filepath}': {e}")
        sys.exit(1)

def categorize_difference(cmd, sub_test, expected, actual):
    """Generate a formatted message based on the type of difference."""
    if actual == "pass":
        return f"{GREEN} FIX: [{cmd}] '{sub_test}' expected '{expected}', but got '{actual}'.{RESET}"
    elif actual == "fail":
        return f"{RED} REGRESSION: [{cmd}] '{sub_test}' expected '{expected}', but got '{actual}'.{RESET}"
    else:
        return f"{YELLOW} DIFFERENCE: [{cmd}] '{sub_test}' expected '{expected}', but got '{actual}'.{RESET}"

def compare_test_subset(cmd, actual_results, expected_results, differences, missing_refs):
    """Compare the run subset against expectations. Mutates lists and returns error status."""
    has_error = False

    for sub_test, actual_status in actual_results.items():
        # Rule 1: Everything run must exist in the reference
        if sub_test not in expected_results:
            missing_refs.append(f"[{cmd}] Sub-test '{sub_test}' not found in golden reference.")
            has_error = True
        else:
            expected_status = expected_results[sub_test]
            # Rule 2: Note ANY difference
            if actual_status != expected_status:
                differences.append(categorize_difference(cmd, sub_test, expected_status, actual_status))
                has_error = True
    return has_error

def process_all_results(results_dir, golden_data):
    """Iterate through all result JSONs and compare them. Returns aggregate data."""
    has_error = False
    differences = []
    missing_refs = []

    if len(list(Path(results_dir).glob("*.json"))) == 0:
        print(f"Error: Results directory '{results_dir}' is empty.")
        sys.exit(1)
    for result_file in Path(results_dir).glob("*.json"):
        with open(result_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing {result_file}. Skipping.")
                has_error = True
                continue

        cmd = os.path.basename(data.get("cmd"))
        results = data.get("results", {})

        if not cmd:
            print(f"File {result_file} is missing the 'cmd' key. Skipping.")
            has_error = True
            continue

        if cmd not in golden_data:
            missing_refs.append(f"cmd '{cmd}' not found in golden reference.")
            has_error = True
            continue

        # Check the specific results against the reference
        subset_error = compare_test_subset(cmd, results, golden_data[cmd], differences, missing_refs)
        if subset_error:
            has_error = True

    return has_error, differences, missing_refs

def print_report_and_exit(has_error, differences, missing_refs):
    """Print the final comparison report and exit with the appropriate status code."""
    if missing_refs:
        print("\n--- Missing References ---")
        for msg in missing_refs:
            print(msg)
        print("\nPlease update the golden reference to include these missing cmd/tests.")

    if differences:
        print("\n--- Test Differences ---")
        for diff in differences:
            print(diff)

    if has_error:
        print(f"\n{RED} Errors found during comparison. Failing the check.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN} All run tests match the golden reference perfectly!{RESET}")
        sys.exit(0)

def main():
    """Main execution flow."""
    args = parse_arguments()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        sys.exit(1)

    golden_data = load_golden_reference(args.golden)

    # Process the files and gather all differences without stopping early
    has_error, differences, missing_refs = process_all_results(results_dir, golden_data)

    # Output the findings and exit
    print_report_and_exit(has_error, differences, missing_refs)

if __name__ == "__main__":
    main()
