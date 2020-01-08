import argparse
import os
import shutil
import subprocess
from sys import platform

'''
    Assuming the script is placed inside the cloned OpenCL-CTS directory. 
    Script is able to build whole OpenCL-CTS suite on Windows (MSVC 2019, MSVC 2017, icc 2019) 
    or Linux operating system (gcc)
    Requirements:
        python 3.X, cmake, git, MSVC 2017/2019 (Windows), GCC (Linux)
    Examples:
        #Build whole suite using default settings
        build_win_lnx.py
        #Build whole suite using MSVC 2019 on Windows, binaries in bitness x86
        build_win_lnx.py --compiler msvc --bitness x86 --msvc_version 2019
        #As above but: use bitness x64 and do not clone repositories
        build_win_lnx.py --compiler msvc --bitness x64 --msvc_version 2019 --skip_clone
        #As above but: build only test_half. Do not build ICD
        build_win_lnx.py --compiler msvc --bitness x64 --msvc_version 2019 --skip_clone –build_target test_half --skip_icd
'''


root_dir = os.getcwd()


def run():
    args = process_command_line()
    compiler = args.compiler
    msvc_version = args.msvc_version
    bitness = args.bitness
    skip_clone = args.skip_clone
    skip_icd = args.skip_icd
    skip_cts = args.skip_cts
    build_target = args.build_target
    icd_solution_cmd_win = 'cmake .. -DOPENCL_ICD_LOADER_HEADERS_DIR=..\..\OpenCL-Headers'
    icd_solution_cmd_win_x64_msvc_2017 = icd_solution_cmd_win + ' -G "Visual Studio 15 2017" -A x64'
    icd_solution_cmd_win_x64_msvc_2019 = icd_solution_cmd_win + ' -G "Visual Studio 16 2019" -A x64'
    icd_solution_cmd_win_x86_msvc_2017 = icd_solution_cmd_win + ' -G "Visual Studio 15 2017"'
    icd_solution_cmd_win_x86_msvc_2019 = icd_solution_cmd_win + ' -G "Visual Studio 16 2019"'
    icd_solution_cmd_linux = 'cmake -DOPENCL_ICD_LOADER_HEADERS_DIR=../OpenCL-Headers/ ..'
    build_icd_cmd = 'cmake --build . --target ALL_BUILD --config Release -- /m'
    build_cts_cmd = 'cmake --build . --target '+build_target+' --config Release -- /m'

    cts_solution_cmd_win = 'cmake ..\. -DCMAKE_BUILD_TYPE=release -DCL_INCLUDE_DIR=..\OpenCL-Headers -DCL_LIB_DIR=..\OpenCL-ICD-Loader\\build\\Release\\ -DCL_LIBCLCXX_DIR=..\libclcxx\\include -DCL_OFFLINE_COMPILER=..\\dummy\\path\\to\\compiler -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=./bin -DOPENCL_LIBRARIES=OpenCL -DD3D10_IS_SUPPORTED=ON -DD3D11_IS_SUPPORTED=ON'

    cts_solution_cmd_win_x64_msvc_2017 = cts_solution_cmd_win + ' -G "Visual Studio 15 2017" -A x64 -DCMAKE_CL_64=ON -DARCH=x86_64'
    cts_solution_cmd_win_x64_icc_2017 = cts_solution_cmd_win_x64_msvc_2017 + ' -T"Intel C++ Compiler 19.0"'

    cts_solution_cmd_win_x64_msvc_2019 = cts_solution_cmd_win + ' -G "Visual Studio 16 2019" -A x64 -DCMAKE_CL_64=ON -DARCH=x86_64'
    cts_solution_cmd_win_x64_icc_2019 = cts_solution_cmd_win_x64_msvc_2019 + ' -T"Intel C++ Compiler 19.0"'

    cts_solution_cmd_win_x86_msvc_2017 = cts_solution_cmd_win + ' -G "Visual Studio 15 2017" -DCMAKE_CL_64=OFF -DARCH=i686'
    cts_solution_cmd_win_x86_icc_2017 = cts_solution_cmd_win_x86_msvc_2017 + ' -T"Intel C++ Compiler 19.0"'

    cts_solution_cmd_win_x86_msvc_2019 = cts_solution_cmd_win + ' -G "Visual Studio 16 2019" -DCMAKE_CL_64=OFF -DARCH=i686'
    cts_solution_cmd_win_x86_icc_2019 = cts_solution_cmd_win_x86_msvc_2019 + ' -T"Intel C++ Compiler 19.0"'

    cts_solution_cmd_linux = 'cmake -DCL_INCLUDE_DIR=../OpenCL-Headers -DCL_LIB_DIR=../OpenCL-ICD-Loader/build -DCL_LIBCLCXX_DIR=../libclcxx -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=./bin -DOPENCL_LIBRARIES="-lOpenCL -lpthread" ..'

    if platform == "win32":
        print("Windows operating system")
        if bitness == "x64" and msvc_version == "2017":
            icd_cmd = icd_solution_cmd_win_x64_msvc_2017
            if compiler == "msvc":
                cts_cmd = cts_solution_cmd_win_x64_msvc_2017
            elif compiler == "icc":
                cts_cmd = cts_solution_cmd_win_x64_icc_2017
        elif bitness == "x64" and msvc_version == "2019":
            icd_cmd = icd_solution_cmd_win_x64_msvc_2019
            if compiler == "msvc":
                cts_cmd = cts_solution_cmd_win_x64_msvc_2019
            elif compiler == "icc":
                cts_cmd = cts_solution_cmd_win_x64_icc_2019
        elif bitness == "x86" and msvc_version == "2017":
            icd_cmd = icd_solution_cmd_win_x86_msvc_2017
            if compiler == "msvc":
                cts_cmd = cts_solution_cmd_win_x86_msvc_2017
            elif compiler == "icc":
                cts_cmd = cts_solution_cmd_win_x86_icc_2017
        elif bitness == "x86" and msvc_version == "2019":
            icd_cmd = icd_solution_cmd_win_x86_msvc_2019
            if compiler == "msvc":
                cts_cmd = cts_solution_cmd_win_x86_msvc_2019
            elif compiler == "icc":
                cts_cmd = cts_solution_cmd_win_x86_icc_2019

    elif platform[:5] == "linux":
        print("Linux operating system")
        compiler = "gcc"
        msvc_version = "NA"
        icd_cmd = icd_solution_cmd_linux
        cts_cmd = cts_solution_cmd_linux
        build_icd_cmd = 'make -j4'
        build_cts_cmd = 'make -j4'
    else:
        print("Not supported operating system ", platform)
        return

    if not skip_clone:
        clone_directories()

    if not skip_icd:
        print("Build OpenCL-ICD-Loader project")
        build_project('OpenCL-ICD-Loader',
                      icd_cmd, build_icd_cmd)

    if not skip_cts:
        print("Build OpenCL-CTS project")
        build_project('OpenCL-CTS', cts_cmd, build_cts_cmd)

    print("+++ Summary +++")
    print("MSVC version:\t ", msvc_version)
    print("Compiler:\t ", compiler)
    print("Bitness:\t ", bitness)
    print("Operating system:\t ", platform)
    print("ICD solution cmd: \n", icd_cmd)
    print("ICD build cmd: \n", build_icd_cmd)
    print("CTS solution cmd: \n", cts_cmd)
    print("CTS build cmd: \n", build_cts_cmd)


def clean_path(path):
    if os.path.exists(path):
        print("Cleaning path: ", path)
        shutil.rmtree(path)
    else:
        print("The path: ", path, " does not exist")


def process_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msvc_version', default="2017", choices=['2017', '2019'],
                        help='choose msvc version')
    parser.add_argument('--compiler', default="msvc", choices=['msvc', 'icc'],
                        help='choose compiler')
    parser.add_argument('--bitness', default="x64", choices=['x64', 'x86'],
                        help='choose bitness')
    parser.add_argument('--skip_clone', action='store_true',  help='skip cloning repositories')
    parser.add_argument('--skip_icd', action='store_true',  help='skip building ICD project')
    parser.add_argument('--skip_cts', action='store_true', help='skip building CTS project')
    parser.add_argument('--build_target', default="ALL_BUILD", help='target to build (only Windows)')
    return parser.parse_args()


def clone_directories():
    os.chdir(root_dir)
    os.chdir("..")
    #cmd = 'git clone https://github.com/KhronosGroup/OpenCL-CTS.git'
    #subprocess.run(cmd, stderr=subprocess.STDOUT, shell=True, check=True)

    cmd = 'git clone https://github.com/KhronosGroup/OpenCL-Headers.git'
    subprocess.run(cmd, stderr=subprocess.STDOUT, shell=True, check=True)

    cmd = 'git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git'
    subprocess.run(cmd, stderr=subprocess.STDOUT, shell=True, check=True)

    cmd = 'git clone https://github.com/KhronosGroup/libclcxx.git'
    subprocess.run(cmd, stderr=subprocess.STDOUT, shell=True, check=True)


def build_project(project_path, solution_cmd, build_cmd):
    os.chdir(root_dir)
    os.chdir("..")
    os.chdir(project_path)
    path = 'build'
    clean_path(path)
    os.mkdir(path)
    os.chdir(path)
    subprocess.run(solution_cmd, stderr=subprocess.STDOUT, shell=True, check=True)
    subprocess.run(build_cmd, stderr=subprocess.STDOUT, shell=True, check=True)


if __name__ == '__main__':
    run()
