# Building the CTS

## Platform support

The CTS supports Linux, Windows, macOS, and Android platforms. In particular,
GitHub Actions CI builds against Ubuntu 20.04, Windows-latest, and
macos-latest.

## Configuring the build

Compiling the CTS minimally requires the following CMake configuration options to be set:

* `CL_INCLUDE_DIR` Points to the unified
  [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers).
* `CL_LIB_DIR` Directory containing the OpenCL library to build against.
* `OPENCL_LIBRARIES` Name of the OpenCL library to link.

It is advised that the [OpenCL ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)
is used as the OpenCL library to build against. Where `CL_LIB_DIR` points to a
build of the ICD loader and `OPENCL_LIBRARIES` is "OpenCL".

## Example Build

Steps on a Linux platform to clone dependencies from GitHub sources, configure
a build, and compile.

```sh
git clone https://github.com/KhronosGroup/OpenCL-CTS.git
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git

mkdir OpenCL-ICD-Loader/build
cmake -S OpenCL-ICD-Loader -B OpenCL-ICD-Loader/build \
      -DOPENCL_ICD_LOADER_HEADERS_DIR=$PWD/OpenCL-Headers
cmake --build ./OpenCL-ICD-Loader/build --config Release

mkdir OpenCL-CTS/build
cmake -S OpenCL-CTS -B OpenCL-CTS/build \
      -DCL_INCLUDE_DIR=$PWD/OpenCL-Headers \
      -DCL_LIB_DIR=$PWD/OpenCL-ICD-Loader/build \
      -DOPENCL_LIBRARIES=OpenCL
cmake --build OpenCL-CTS/build --config Release
```