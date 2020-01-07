#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import traceback

if len(sys.argv) != 3:
    print('Usage: "generate_spirv_offline.py <compilation_cache_dir> <cl_device_info_file>"')
    exit(1)

compilation_cache_dir = sys.argv[1]
cl_device_info_filename = sys.argv[2]

def generate_spirv():
    print("Generating SPIR-V files")
    build_options = ''
    
    if os.path.exists(compilation_cache_dir):
        for root, dirs, files in os.walk(compilation_cache_dir):
            for file in files:
                if file.endswith('.cl'):
                    options_file_name = file[:-2] + "options"
                    if os.path.exists(os.path.join(root, options_file_name)):
                        optFile = open (os.path.join(root, options_file_name), 'r')
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



