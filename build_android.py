#!/usr/bin/python


#-------------------------------------------------------------------------------#
# android-cmake and android-ndk based build script for conformance
#-------------------------------------------------------------------------------#
"""
Dependencies:

1) android-ndk version android-ndk-r10d or higher is required. Further, the environment
   variable ANDROID_NDK should be defined to point to it.

2) android-cmake should be installed (else the script can install it for you). If installed,
   the environment variable ANDROID_CMAKE should point to install location, unless it is in the current
   working directory in which case it is picked up by default.

3) CL_INCLUDE_DIR should be defined to point to CL headers. Alternately, this can be provided
   as an input (-I)

4) Path to opencl library to link against (libOpenCL.so) can be provided using -L. If this isn't
   available the script will try to use CL_LIB_DIR_64 or CL_LIB_DIR_32 environment variables -
   if available - to pick up the right library for the architecture being built.


"""

import os
import sys
import subprocess
import argparse
import time
import shlex

start  = time.time()
script = os.path.basename( sys.argv[ 0 ] )

def die (msg):
    print msg
    exit(-1)

def execute (cmdline):
    retcode = subprocess.call(cmdline)
    if retcode != 0:
        raise Exception("Failed to execute '%s', got %d" % (commandLine, retcode))

def build(args):
    if not (args.testDir):
        print("building...")
        execute("make")
    else:
        if os.path.exists( os.path.join(args.bld_dir, "test_conformance", args.testDir) ):
            os.chdir( os.path.join("test_conformance",args.testDir) )
            print("Building test: %s..." %args.testDir)
            execute("make")
            os.chdir(args.bld_dir)
        else:
            print ("Error: %s test doesn't exist" %args.testDir)


def configure (args):
    print("configuring...")
    cmdline = []
    cmdline.extend(['cmake', "-DCMAKE_TOOLCHAIN_FILE=" + os.path.join(args.android_cmake,"android.toolchain.cmake")])
    for var in args.cmake_defs :
        cmdline.extend([ '-D', var ])
    cmdline.extend(['-DCL_INCLUDE_DIR=' + args.inc_dir])
    cmdline.extend(['-DCL_LIB_DIR=' + args.lib_dir])
    cmdline.extend(['-DANDROID_NATIVE_API_LEVEL=' + "android-21"])
    if args.arch == "64":
        cmdline.extend(['-DANDROID_ABI=arm64-v8a'])
        cmdline.extend(['-DANDROID_SO_UNDEFINED=ON'])
    cmdline.extend([args.src_dir])
    execute(cmdline)

def check_var (parser, args, name):
    if not(args.__dict__[name]):
        parser.error("%s needs to be defined" % name)

def print_config(args):
    print("----------CONFIGURATION--------------\n")
    print("android_cmake: %s" % args.android_cmake)
    print("android_ndk:   %s" % args.android_ndk)
    print("lib_dir:       %s" % args.lib_dir)
    print("inc_dir:       %s" % args.inc_dir)
    if len(args.cmake_defs):
        print("cmake options:" + "\n:".join( [ " `%s'" % dir for dir in args.cmake_defs ] ))
    print("architecture:  %s" % args.arch)
    print("-------------------------------------\n")

def get_input():
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])

    choice = raw_input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")
        exit()

def install_android_cmake():
    parser.print_help()
    print "\nandroid-cmake doesn't seem to be installed - It should be provided as a) cmdline input b) environment variable $ANDROID_CMAKE or c) present in the current directory\n"
    print "if you would like to download and install it in the current directory please enter yes\n"
    print "if you would like to provide an environment variable($ANDROID_CMAKE) or command-line input(--android_cmake) rerun the script enter no\n"
    print "input: "
    if get_input():
        print("installing android-cmake")
        #subprocess.call(['git', 'clone', 'https://github.com/taka-no-me/android-cmake'])
        # Use a newer fork of android-cmake which has been updated to support Clang. GCC is deprecated in newer NDKs and C11 atomics conformance doesn't build with NDK > 10.
        subprocess.call(['git', 'clone', 'https://github.com/daewoong-jang/android-cmake'])
        args.android_cmake = os.path.join(args.src_dir,"android-cmake")
    else:
        exit()

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--android_cmake', dest='android_cmake', default=os.environ.get('ANDROID_CMAKE'), help="Path to android-cmake (can also be set using environment variable $ANDROID_CMAKE).")
    parser.add_argument('--android_ndk', dest='android_ndk', default=os.environ.get('ANDROID_NDK'), help="Path to android-ndk (can also be set using environment variable $ANDROID_NDK).")
    parser.add_argument('-L','--lib_dir', dest='lib_dir', default="", help="Path to libOpenCL to link against (can also be set using environment variable $CL_LIB_DIR_32 and $CL_LIB_DIR_64).")
    parser.add_argument('-I','--include_dir', dest='inc_dir', default=os.environ.get('CL_INCLUDE_DIR'), help="Path to headers (can also be set using environment variable $CL_INCLUDE_DIR).")
    parser.add_argument('-D', dest='cmake_defs', action='append', default=[], help="Define CMAKE variable")
    parser.add_argument('-a','--arch', default="32", help="Architecture to build for (32 or 64)")
    parser.add_argument('-t','--test', dest='testDir', default="", help="Builds the given test")

    args = parser.parse_args()

    args.src_dir = os.path.realpath(os.path.dirname( sys.argv[ 0 ]))

    if not (args.android_cmake):
        if os.path.exists(os.path.join(args.src_dir,"android-cmake")):
            args.android_cmake = os.path.join(args.src_dir,"android-cmake")
        else:
            install_android_cmake()

    if not (args.lib_dir):
        lib_var_name = "CL_LIB_DIR_" + ("32" if (args.arch == "32") else "64")
        args.lib_dir = os.environ.get(lib_var_name)

    check_var(parser, args, "android_cmake")
    check_var(parser, args, "lib_dir")
    check_var(parser, args, "inc_dir")
    check_var(parser, args, "android_ndk")

    print_config(args)

    args.bld_dir = os.path.join(args.src_dir, 'bld_android_%s' % args.arch)
    if not os.path.exists(args.bld_dir):
        os.makedirs(args.bld_dir)
    os.chdir(args.bld_dir)

    configure(args)
    build(args)

    sys.exit( 0 )

finally:
    finish = time.time()
    print("Elapsed time: %.0f s." % ( finish - start ) )
