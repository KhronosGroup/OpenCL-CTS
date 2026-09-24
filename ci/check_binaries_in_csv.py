import sys
import os
import re

def main():
    if len(sys.argv) < 3:
        print("Usage: check_binaries_in_csv.py <csv_file> <binary1> <binary2> ...")
        sys.exit(1)

    csv_file = sys.argv[1]
    binaries = sys.argv[2:]

    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        sys.exit(1)

    with open(csv_file, 'r') as f:
        content = f.read()

    csv_binaries = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        fields = [f.strip() for f in line.split(',')]

        for field in fields:
            parts = field.split()
            if not parts:
                continue
            path = parts[0] # The first part should be the path
            binary_name = os.path.basename(path)
            csv_binaries.add(binary_name)

    missing_binaries = [b for b in binaries if b not in csv_binaries]

    if missing_binaries:
        print(f"Error: The following binaries are missing from {csv_file}:")
        for binary in missing_binaries:
            print(f"  {binary}")
        sys.exit(1)

    print(f"All {len(binaries)} binaries found in {csv_file}.")
    sys.exit(0)

if __name__ == "__main__":
    main()
