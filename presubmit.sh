#!/usr/bin/env bash

set -e

export TOP=$(pwd)

TOOLCHAIN_PREFIX_arm=arm-linux-gnueabihf
TOOLCHAIN_PREFIX_aarch64=aarch64-linux-gnu

if [[ ${JOB_ARCHITECTURE} == android-* ]]; then
    TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
    CMAKE_CONFIG_ARGS_ANDROID="-DCMAKE_ANDROID_ARCH_ABI=${ANDROID_ARCH_ABI}"
else
    TOOLCHAIN_FILE=${TOP}/toolchain.cmake
    touch ${TOOLCHAIN_FILE}
fi

BUILD_OPENGL_TEST="OFF"
BUILD_VULKAN_TEST="ON"

cmake --version
echo

# Prepare toolchain if needed
if [[ ${JOB_ARCHITECTURE} != android-* ]]; then
    if [[ ${JOB_ARCHITECTURE} != "" && ${RUNNER_OS} != "Windows" ]]; then
        TOOLCHAIN_PREFIX_VAR=TOOLCHAIN_PREFIX_${JOB_ARCHITECTURE}
        TOOLCHAIN_PREFIX=${!TOOLCHAIN_PREFIX_VAR}

        echo "SET(CMAKE_SYSTEM_NAME Linux)" >> ${TOOLCHAIN_FILE}
        echo "SET(CMAKE_SYSTEM_PROCESSOR ${JOB_ARCHITECTURE})" >> ${TOOLCHAIN_FILE}
        echo "SET(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}-gcc)" >> ${TOOLCHAIN_FILE}
        echo "SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)" >> ${TOOLCHAIN_FILE}
        echo "SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)" >> ${TOOLCHAIN_FILE}
        echo "SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)" >> ${TOOLCHAIN_FILE}
        echo "SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)" >> ${TOOLCHAIN_FILE}
    fi
fi

if [[ ( ${JOB_ARCHITECTURE} == "" && ${JOB_ENABLE_GL} == "1" ) ]]; then
    BUILD_OPENGL_TEST="ON"
fi

if [[ ${JOB_ENABLE_DEBUG} == 1 ]]; then
    BUILD_CONFIG="Debug"
else
    BUILD_CONFIG="Release"
fi

#Vulkan Headers
git clone https://github.com/KhronosGroup/Vulkan-Headers.git

# Get and build loader
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
cd ${TOP}/OpenCL-ICD-Loader
mkdir build
cd build
cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DOPENCL_ICD_LOADER_HEADERS_DIR=${TOP}/OpenCL-Headers/ \
      "${CMAKE_CONFIG_ARGS_ANDROID}"
cmake --build . --parallel

#Vulkan Loader
if [[ ${JOB_ARCHITECTURE} != android-* ]]; then
    # Building the Vulkan loader is not supported on Android,
    # instead, the loader is shipped as part of the operating system
    cd ${TOP}
    git clone https://github.com/KhronosGroup/Vulkan-Loader.git
    cd Vulkan-Loader
    mkdir build
    cd build
    python3 ../scripts/update_deps.py
    cmake .. -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
          -DBUILD_WSI_XLIB_SUPPORT=OFF \
          -DBUILD_WSI_XCB_SUPPORT=OFF \
          -DBUILD_WSI_WAYLAND_SUPPORT=OFF \
          -C helper.cmake ..
    cmake --build . --parallel
fi

# Build CTS
cd ${TOP}
ls -l
mkdir build
cd build
if [[ ${RUNNER_OS} == "Windows" ]]; then
    CMAKE_OPENCL_LIBRARIES_OPTION="OpenCL"
else
    CMAKE_OPENCL_LIBRARIES_OPTION="-lOpenCL"
    if [[ ${JOB_ARCHITECTURE} != android-* ]]; then
        CMAKE_OPENCL_LIBRARIES_OPTION="${CMAKE_OPENCL_LIBRARIES_OPTION} -lpthread"
    fi
fi
cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE="${BUILD_CONFIG}" \
      -DCMAKE_CACHE_OPTIONS="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache" \
      -DCL_INCLUDE_DIR=${TOP}/OpenCL-Headers \
      -DCL_LIB_DIR=${TOP}/OpenCL-ICD-Loader/build \
      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=./bin \
      -DOPENCL_LIBRARIES="${CMAKE_OPENCL_LIBRARIES_OPTION}" \
      -DUSE_CL_EXPERIMENTAL=ON \
      -DGL_IS_SUPPORTED=${BUILD_OPENGL_TEST} \
      -DVULKAN_IS_SUPPORTED=${BUILD_VULKAN_TEST} \
      -DVULKAN_INCLUDE_DIR=${TOP}/Vulkan-Headers/include/ \
      -DVULKAN_LIB_DIR=${TOP}/Vulkan-Loader/build/loader/ \
      "${CMAKE_CONFIG_ARGS_ANDROID}"
cmake --build . --parallel
