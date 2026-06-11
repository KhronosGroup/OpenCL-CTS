# OpenCL Conformance Test Suite (CTS)

This is the OpenCL CTS for all versions of the Khronos
[OpenCL](https://www.khronos.org/opencl/) standard.

## Building the CTS

The CTS supports Linux, Windows, macOS, and Android platforms. In particular,
GitHub Actions CI builds against Ubuntu 20.04, Windows-latest, and
macos-latest.

Compiling the CTS requires the following CMake configuration options to be set:

* `CL_INCLUDE_DIR` Points to the unified
  [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers).
* `SPIRV_INCLUDE_DIR` Points to the unified
  [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers).
* `CL_LIB_DIR` Directory containing the OpenCL library to build against.
* `SPIRV_TOOLS_DIR` Directory containing the `spirv-as` and `spirv-val` binaries
   to be used in the CTS build process. Alternatively, the location to these binaries
   can be provided via the `PATH` variable.
* `OPENCL_LIBRARIES` Name of the OpenCL library to link.

It is advised that the [OpenCL ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)
is used as the OpenCL library to build against. Where `CL_LIB_DIR` points to a
build of the ICD loader and `OPENCL_LIBRARIES` is "OpenCL".

### Building CTS on Linux

This section provides an example of building CTS on a Linux platform. It includes steps for
cloning the necessary dependencies from GitHub, setting up the build configuration, and compiling
the project.

```sh
git clone https://github.com/KhronosGroup/OpenCL-CTS.git
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
git clone https://github.com/KhronosGroup/SPIRV-Headers.git
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
git clone https://github.com/KhronosGroup/SPIRV-Tools.git
git clone https://github.com/KhronosGroup/SPIRV-Headers.git SPIRV-Tools/external/spirv-headers
git clone https://github.com/google/effcee.git SPIRV-Tools/external/effcee
git clone https://github.com/google/re2.git SPIRV-Tools/external/re2


mkdir OpenCL-ICD-Loader/build
cmake -S OpenCL-ICD-Loader -B OpenCL-ICD-Loader/build \
      -DOPENCL_ICD_LOADER_HEADERS_DIR=$PWD/OpenCL-Headers
cmake --build ./OpenCL-ICD-Loader/build --config Release

mkdir SPIRV-Tools/build
cmake -S SPIRV-Tools -B SPIRV-Tools/build -DSPIRV_SKIP_TESTS=ON
cmake --build SPIRV-Tools/build --config Release

mkdir OpenCL-CTS/build
cmake -S OpenCL-CTS -B OpenCL-CTS/build \
      -DCL_INCLUDE_DIR=$PWD/OpenCL-Headers \
      -DSPIRV_INCLUDE_DIR=$PWD/SPIRV-Headers \
      -DCL_LIB_DIR=$PWD/OpenCL-ICD-Loader/build \
      -DSPIRV_TOOLS_DIR=$PWD/SPIRV-Tools/build/tools/ \
      -DOPENCL_LIBRARIES=OpenCL
cmake --build OpenCL-CTS/build --config Release
```

## Running the CTS

A build of the CTS contains multiple executables representing the directories in
the `test_conformance` folder. Each of these executables contains sub-tests, and
possibly smaller granularities of testing within the sub-tests.

See the `--help` output on each executable for the list of sub-tests available,
as well as other options for configuring execution.

If the OpenCL library built against is the ICD Loader, and the vendor library to
be tested is not registered in the
[default ICD Loader location](https://github.com/KhronosGroup/OpenCL-ICD-Loader#registering-icds)
then the [OCL_ICD_FILENAMES](https://github.com/KhronosGroup/OpenCL-ICD-Loader#table-of-debug-environment-variables)
environment variable will need to be set for the ICD Loader to detect the OpenCL
library to use at runtime. For example, to run the basic tests on a Linux
platform:

```sh
OCL_ICD_FILENAMES=/path/to/vendor_lib.so ./test_basic
```

### Offline Compilation

Testing OpenCL drivers which do not have a runtime compiler can be done by using
additional command line arguments provided by the test harness for tests which
require compilation, these are:

* `--compilation-mode` Selects if OpenCL-C source code should be compiled using
  an external tool before being passed on to the OpenCL driver in that form for
  testing. Online is the default mode, but also accepts the values `spir-v`, and
  `binary`.

* `--compilation-cache-mode` Controls how the compiled OpenCL-C source code
  should be cached on disk.

* `--compilation-cache-path` Accepts a path to a directory where the compiled
  binary cache should be stored on disk.

* `--compilation-program` Accepts a path to an executable (default:
   cl_offline_compiler) invoked by the test harness to perform offline
   compilation of OpenCL-C source code.  This executable must match the
   [interface description](test_common/harness/cl_offline_compiler-interface.txt).

### Building CTS on Windows

For Windows environments, it is strongly recommended to build CTS using [MSYS2](https://www.msys2.org/),
the MinGW-w64 (GCC) toolchain, and Ninja.
All commands in the following sections should be run from an MSYS2 MinGW64 shell.

#### Prerequisites

Install the required MSYS2 packages:

```sh
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-git mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja mingw-w64-x86_64-python
```

#### Clone Source and Dependencies 

```sh
git clone https://github.com/KhronosGroup/OpenCL-CTS.git
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
git clone https://github.com/KhronosGroup/SPIRV-Headers.git
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
git clone https://github.com/KhronosGroup/SPIRV-Tools.git
git clone https://github.com/KhronosGroup/SPIRV-Headers.git SPIRV-Tools/external/spirv-headers
git clone https://github.com/google/effcee.git SPIRV-Tools/external/effcee
git clone https://github.com/google/re2.git SPIRV-Tools/external/re2
```

#### Build the ICD Loader

```sh
cmake -S OpenCL-ICD-Loader -B OpenCL-ICD-Loader/build -G "Ninja" \
      -DOPENCL_ICD_LOADER_HEADERS_DIR=$PWD/OpenCL-Headers
cmake --build OpenCL-ICD-Loader/build --config Release
```

#### Build SPIRV-Tools

```sh
cmake -S SPIRV-Tools -B SPIRV-Tools/build -G "Ninja" -DSPIRV_SKIP_TESTS=ON
cmake --build SPIRV-Tools/build --config Release
```

#### Build the CTS

```sh
cmake -S OpenCL-CTS -B OpenCL-CTS/build -G "Ninja" \
      -DCL_INCLUDE_DIR=$PWD/OpenCL-Headers \
      -DCL_LIB_DIR=$PWD/OpenCL-ICD-Loader/build \
      -DSPIRV_INCLUDE_DIR=$PWD/SPIRV-Headers \
      -DSPIRV_TOOLS_DIR=$PWD/SPIRV-Tools/build/tools \
      -DOPENCL_LIBRARIES=OpenCL
cmake --build OpenCL-CTS/build --config Release
```

#### Running Tests

The compiled executables must be run from a Windows Command Prompt (cmd.exe) or PowerShell session.
Running them directly from the MSYS2 shell is not supported, as MSYS2 Bash may fail to launch the
executables (exit code 127) due to a known interoperability issue with PE binaries that depend on
system DLLs such as OpenCL.dll.

From cmd.exe:

```cmd
set PATH=C:\msys64\mingw64\bin;%PATH%
cd OpenCL-CTS\build\test_conformance\basic
test_basic.exe
```

Alternatively, launch from the MSYS2 shell via cmd.exe:

```sh
cmd.exe //c "set PATH=C:\\msys64\\mingw64\\bin;%PATH% && test_conformance\\basic\\test_basic.exe"
```

## Generating a Conformance Report

The Khronos [Conformance Process Document](https://members.khronos.org/document/dl/911)
details the steps required for a conformance submission.
In this repository [opencl_conformance_tests_full.csv](test_conformance/opencl_conformance_tests_full.csv)
defines the full list of tests which must be run for conformance. The output log
of which must be included alongside a filled in
[submission details template](test_conformance/submission_details_template.txt).

Utility script [run_conformance.py](test_conformance/run_conformance.py) can be
used to help generating the submission log, although it is not required.

Git [tags](https://github.com/KhronosGroup/OpenCL-CTS/tags) are used to define
the version of the repository conformance submissions are made against.

## Contributing

Contributions are welcome to the project from Khronos members and non-members
alike via GitHub Pull Requests (PR). Alternatively, if you've found a bug or have
a question please file an issue in the GitHub project. First time contributors
will be required to sign the Khronos Contributor License Agreement (CLA) before
their PR can be merged.

PRs to the repository are required to be `clang-format` clean to pass CI.
Developers can either use the `git-clang-format` tool locally to verify this
before contributing, or update their PR based on the diff provided by a failing
CI job.

## Running Targeted CI Tests on Pull Requests

To help verify fixes or check for regressions without running the entire
conformance test suite, our continuous integration pipeline allows contributor
to trigger specific tests on Pull Requests against the `pocl` implementation.

### How to Trigger Tests

Testing is triggered by adding a special tag to either your
**Pull Request description** or in any of your **commit messages**.

The CI parses the text for the following syntax:
`[run-test: <command>]`

Multiples tags for a single Pull Request is supported.

### Examples

```text
[run-test: test_bruteforce exp -w -1]
