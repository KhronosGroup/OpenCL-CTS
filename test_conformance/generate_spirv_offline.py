import sys
import os
import platform
import re
import string
import shutil
import traceback

if len(sys.argv)<2:
    print 'Usage: "generate_spirv_offline.py <input> <32|64>"'
    exit(1)

input_dir = sys.argv[1]
arch = sys.argv[2]

def generate_spirv():
    print "generating spirv"
    ocl_version = '12';
    build_options = ''
    
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
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
                        print build_options
                    input_string = os.path.join(root, file)
                    output_string = os.path.join(root, file[:-2])

                    command_line = ".\\build_script_spirv.py " + input_string + " " + output_string + "spv" + arch + " " + arch + " spir_v " + ocl_version + " \"" + build_options + " \""
                    print command_line
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



