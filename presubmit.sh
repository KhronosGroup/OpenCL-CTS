#!/usr/bin/env bash

set -e

export TOP=$(pwd)

TOOLCHAIN_URL_arm="https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/arm-linux-gnueabihf/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz"
TOOLCHAIN_URL_aarch64="https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/aarch64-linux-gnu/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz"

TOOLCHAIN_PREFIX_arm=arm-linux-gnueabihf
TOOLCHAIN_PREFIX_aarch64=aarch64-linux-gnu

TOOLCHAIN_FILE=${TOP}/toolchain.cmake
touch ${TOOLCHAIN_FILE}
BUILD_OPENGL_TEST="OFF"
BUILD_VULKAN_TEST="ON"

cmake --version
echo

# Prepare toolchain if needed
if [[ ${JOB_ARCHITECTURE} != "" && ${RUNNER_OS} != "Windows" ]]; then
    TOOLCHAIN_URL_VAR=TOOLCHAIN_URL_${JOB_ARCHITECTURE}
    TOOLCHAIN_URL=${!TOOLCHAIN_URL_VAR}
    wget ${TOOLCHAIN_URL}
    TOOLCHAIN_ARCHIVE=${TOOLCHAIN_URL##*/}
    tar xf ${TOOLCHAIN_ARCHIVE}
    TOOLCHAIN_DIR=${TOP}/${TOOLCHAIN_ARCHIVE%.tar.xz}
    export PATH=${TOOLCHAIN_DIR}/bin:${PATH}

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
      -DOPENCL_ICD_LOADER_HEADERS_DIR=${TOP}/OpenCL-Headers/
cmake --build . -j2

#Vulkan Loader
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
cmake --build . -j2

# Build CTS
cd ${TOP}
ls -l
mkdir build
cd build
if [[ ${RUNNER_OS} == "Windows" ]]; then
  CMAKE_OPENCL_LIBRARIES_OPTION="OpenCL"
  CMAKE_CACHE_OPTIONS=""
else
  CMAKE_OPENCL_LIBRARIES_OPTION="-lOpenCL -lpthread"
  CMAKE_CACHE_OPTIONS="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
fi
cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE="${BUILD_CONFIG}" \
      ${CMAKE_CACHE_OPTIONS} \
      -DCL_INCLUDE_DIR=${TOP}/OpenCL-Headers \
      -DCL_LIB_DIR=${TOP}/OpenCL-ICD-Loader/build \
      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=./bin \
      -DOPENCL_LIBRARIES="${CMAKE_OPENCL_LIBRARIES_OPTION}" \
      -DUSE_CL_EXPERIMENTAL=ON \
      -DGL_IS_SUPPORTED=${BUILD_OPENGL_TEST} \
      -DVULKAN_IS_SUPPORTED=${BUILD_VULKAN_TEST} \
      -DVULKAN_INCLUDE_DIR=${TOP}/Vulkan-Headers/include/ \
      -DVULKAN_LIB_DIR=${TOP}/Vulkan-Loader/build/loader/
cmake --build . -j3
