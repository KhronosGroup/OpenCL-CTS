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
* `CL_LIB_DIR` Directory containing the OpenCL library to build against.
* `OPENCL_LIBRARIES` Name of the OpenCL library to link.

It is advised that the [OpenCL ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)
is used as the OpenCL library to build against. Where `CL_LIB_DIR` points to a
build of the ICD loader and `OPENCL_LIBRARIES` is "OpenCL".

### Example Build

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

## Generating a Conformance Report

The Khronos [Conformance Process Document](https://members.khronos.org/document/dl/911)
details the steps required for a conformance submission.
In this repository [opencl_conformance_tests_full.csv](test_conformance/submission_details_template.txt)
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
