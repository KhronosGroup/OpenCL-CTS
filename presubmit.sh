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

# Prepare toolchain if needed
if [[ ${JOB_ARCHITECTURE} != "" ]]; then
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
    sudo apt-get update
    sudo apt-get -y install libglu1-mesa-dev freeglut3-dev mesa-common-dev libglew-dev
fi
# Prepare headers
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
cd OpenCL-Headers
ln -s CL OpenCL # For OSX builds
cd ..

# Get and build loader
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
cd ${TOP}/OpenCL-ICD-Loader
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DOPENCL_ICD_LOADER_HEADERS_DIR=${TOP}/OpenCL-Headers/ ..
make

# Get libclcxx
cd ${TOP}
git clone https://github.com/KhronosGroup/libclcxx.git

# Build CTS
ls -l
mkdir build
cd build
cmake -DCL_INCLUDE_DIR=${TOP}/OpenCL-Headers \
      -DCL_LIB_DIR=${TOP}/OpenCL-ICD-Loader/build \
      -DCL_LIBCLCXX_DIR=${TOP}/libclcxx \
      -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=./bin \
      -DOPENCL_LIBRARIES="-lOpenCL -lpthread" \
      -DUSE_CL_EXPERIMENTAL=ON \
      -DGL_IS_SUPPORTED=${BUILD_OPENGL_TEST} \
      ..
make -j2

