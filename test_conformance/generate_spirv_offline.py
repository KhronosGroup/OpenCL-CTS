#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import re
import traceback

if len(sys.argv) != 3:
    print('Usage: "generate_spirv_offline.py <compilation_cache_dir> <32|64>"')
    exit(1)

compilation_cache_dir = sys.argv[1]
arch = sys.argv[2]

def generate_spirv():
    print("Generating SPIR-V files")
    ocl_version = '12';
    build_options = ''
    
    if os.path.exists(compilation_cache_dir):
        for root, dirs, files in os.walk(compilation_cache_dir):
            for file in files:
                if file.endswith('.cl'):
                    options_file_name = file[:-2] + "options"
                    ocl_version = '12'
                    if os.path.exists(os.path.join(root, options_file_name)):
                        optFile = open (os.path.join(root, options_file_name), 'rU')
                        for line in optFile:
                            if re.search("-cl-std=CL2.0", line):
                                ocl_version = '20'
                        build_options = re.sub("-cl-std=CL2.0", "", line)
                        print(build_options)
                    source_filename = os.path.join(root, file)
                    output_filename = os.path.join(root, file[:-2]) + "spv" + arch

                    command_line = (".\\build_script_spirv.py" +
                                    " " + source_filename +
                                    " " + output_filename +
                                    " " + arch +
                                    " spir_v" +
                                    " " + ocl_version +
                                    '"' + build_options + '"')
                    print(command_line)
                    os.system(command_line)
    return 0

def main():
    try: 
        generate_spirv()
    except Exception: 
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()



