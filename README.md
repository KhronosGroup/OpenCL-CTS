OpenCL-CTS [![Build Status](https://api.travis-ci.org/KhronosGroup/OpenCL-CTS.svg?branch=master)](https://travis-ci.org/KhronosGroup/OpenCL-CTS/branches)
=================

This document describes how to build the OpenCL Conformance Test suite.

Requirements
------------

### Common
The following tools must be installed and present in the PATH variable:

 * Git (for checking out sources)
 * CMake 3.5.1 or newer

### Win32
Visual Studio 2015 or newe

### Linux
Standard toolchain (make, gcc/clang)

### Android

 * Android NDK r17c or later.
 * Android SDK with: SDK Tools, SDK Platform-tools, SDK Build-tools, and API 28

If you have downloaded the Android SDK command line tools package (25.2.3 or higher) then you can install the necessary components by running:

	tools/bin/sdkmanager tools platform-tools 'build-tools;25.0.2' 'platforms;android-28'

Building CTS
------------
With CMake, out-of-source builds are always recommended. Create a build directory
of your choosing, and in that directory generate Makefiles or IDE project
using cmake.

### Windows x86-32


### Windows x86-64


### Linux 32-bit Debug


### Linux 64-bit Debug

### Android 32-bit Release

       cmake -S . -B <path to build directory> -DOPENCL_LIBRARIES=OpenCL
       -DCMAKE_TOOLCHAIN_FILE=<path to android.toolchain.cmake within android NDK> -DANDROID_PLATFORM=28
       -DCL_INCLUDE_DIR=<path to a folder containing CL headers>  -DCL_LIB_DIR=<path to libOpenCL> -DCL_LIBCLCXX_DIR=<path to libOpenCL>
       -DANDROID_ABI=armeabi-v7a   -DCMAKE_BUILD_TYPE=release
       cmake --build <build directory>

### Android 64-bit Release

      cmake -S . -B <path to build directory> -DOPENCL_LIBRARIES=OpenCL
      -DCMAKE_TOOLCHAIN_FILE=<path to android.toolchain.cmake within android NDK> -DANDROID_PLATFORM=28
      -DCL_INCLUDE_DIR=<path to a folder containing CL headers>  -DCL_LIB_DIR=<path to libOpenCL> -DCL_LIBCLCXX_DIR=<path to libOpenCL>
      -DANDROID_ABI=arm64-v8a -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=gold  -DCMAKE_BUILD_TYPE=release
      cmake --build <build directory>

