#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import traceback
import shutil

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print('Usage: "generate_spirv_offline.py <compilation_cache_dir> <cl_device_info_file> [spirv_val_path]"')
    exit(1)

compilation_cache_dir = sys.argv[1]
cl_device_info_filename = sys.argv[2]
spirv_val_path = sys.argv[3] if len(sys.argv) == 4 else 'spirv-val'

def find_spirv_val():
    """Locate spirv-val binary. Returns the path or None if not found."""
    # If an explicit path was provided as argv[3], use it directly.
    if len(sys.argv) == 4:
        if os.path.isfile(spirv_val_path) and os.access(spirv_val_path, os.X_OK):
            return spirv_val_path
        print('Warning: specified spirv-val path not found or not executable: ' + spirv_val_path)
        return None
    # Otherwise search PATH.
    found = shutil.which('spirv-val')
    if found is None:
        print('Warning: spirv-val not found in PATH. SPIR-V validation will be skipped.')
    return found


def validate_spirv(spirv_val, spv_file):
    """Run spirv-val on spv_file. Returns True on success, False on failure."""
    command_line = spirv_val + ' "' + spv_file + '"'
    print(command_line)
    ret = os.system(command_line)
    if ret != 0:
        print('SPIR-V validation FAILED for: ' + spv_file)
        return False
    return True


def generate_spirv():
    print("Generating SPIR-V files")
    build_options = ''

    spirv_val = find_spirv_val()

    validation_errors = []

    if os.path.exists(compilation_cache_dir):
        for root, dirs, files in os.walk(compilation_cache_dir):
            for file in files:
                if file.endswith('.cl'):
                    options_file_name = file[:-2] + "options"
                    if os.path.exists(os.path.join(root, options_file_name)):
                        optFile = open(os.path.join(root, options_file_name), 'r')
                        build_options = optFile.readline().strip()
                        print(build_options)
                    source_filename = os.path.join(root, file)
                    output_filename = os.path.join(root, file[:-2]) + "spv"

                    command_line = ("cl_offline_compiler" +
                                    " --source=" + source_filename +
                                    " --output=" + output_filename +
                                    " --cl-device-info=" + cl_device_info_filename +
                                    " --mode=spir-v -- " +
                                    '"' + build_options + '"')
                    print(command_line)
                    ret = os.system(command_line)

                    if ret != 0:
                        print('Compilation FAILED for: ' + source_filename)
                        validation_errors.append(output_filename)
                        continue

                    # Validate the generated SPIR-V binary if spirv-val is available.
                    if spirv_val is not None:
                        if not validate_spirv(spirv_val, output_filename):
                            validation_errors.append(output_filename)

    if validation_errors:
        print('\nSPIR-V validation errors in the following files:')
        for f in validation_errors:
            print('  ' + f)
        return 1

    return 0


def main():
    try:
        result = generate_spirv()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        result = 1
    sys.exit(result)

if __name__ == "__main__":
    main()
