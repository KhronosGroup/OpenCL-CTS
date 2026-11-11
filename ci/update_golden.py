#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add missing expected test results to a golden reference JSON file."
    )
    parser.add_argument(
        "-o", "--golden", required=True, help="Path to golden reference JSON"
    )
    parser.add_argument(
        "entries",
        nargs="*",
        metavar="CMD.ARGS.SUB_TEST.STATUS",
        help="Missing test results to add (format: cmd.args.sub_test.status)",
    )
    return parser.parse_args()


def load_golden_reference(filepath):
    """Load and return the golden reference JSON data."""
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load golden reference '{filepath}': {e}")
        sys.exit(1)


def parse_entry(entry):
    """Parse a 'cmd.args.sub_test.status' string into its 4 components."""
    parts = entry.split(".")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid entry '{entry}': expected format 'cmd.args.sub_test.status'."
        )
    cmd, args_str, sub_test, status = parts
    if not cmd or not sub_test or not status:
        raise ValueError(
            f"Invalid entry '{entry}': cmd, sub_test, and status must be non-empty."
        )
    return cmd, args_str, sub_test, status


def merge_missing_results(golden_data, entries):
    """Merge missing test results into the golden reference data."""
    added_count = 0
    for entry in entries:
        try:
            cmd, args_str, sub_test, status = parse_entry(entry)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        golden_data.setdefault(cmd, {}).setdefault(args_str, {})[sub_test] = status
        added_count += 1

    return golden_data, added_count


def main():
    """Main execution flow."""
    args = parse_arguments()
    golden_path = Path(args.golden)

    golden_data = load_golden_reference(golden_path)
    golden_data, added_count = merge_missing_results(golden_data, args.entries)

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w") as f:
        json.dump(golden_data, f, indent=4, sort_keys=True)
        f.write("\n")

    if added_count > 0:
        print(
            f"Added {added_count} result(s) to '{args.golden}'. "
            "Please commit and push the changes."
        )


if __name__ == "__main__":
    main()
