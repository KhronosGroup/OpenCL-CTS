# Running the CTS

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

## Common test interface

All test share the same command-line interface, which they inherit from test harness. Running any test executable with `--help` prints something similar:

```
.\test_cpp_basic.exe --help
Common options:
    -h, --help
        This help
    --compilation-mode <mode>
        Specify a compilation mode.  Mode can be:
            online     Use online compilation (default)
            binary     Use binary offline compilation
            spir-v     Use SPIR-V offline compilation

For offline compilation (binary and spir-v modes) only:
    --compilation-cache-mode <cache-mode>
        Specify a compilation caching mode:
            compile-if-absent
                Read from cache if already populated, or else perform
                offline compilation (default)
            force-read
                Force reading from the cache
            overwrite
                Disable reading from the cache
            dump-cl-files
                Dumps the .cl and build .options files used by the test suite
    --compilation-cache-path <path>
        Path for offline compiler output and CL source
    --compilation-program <prog>
        Program to use for offline compilation, defaults to:
            cl_offline_compiler

For spir-v mode only:
    --disable-spirv-validation
        Disable validation of SPIR-V using the SPIR-V validator
    --spirv-validator
        Path for SPIR-V validator, defaults to spirv-val

Usage: C:\Users\mate\Source\Repos\OpenCL-CTS\.vscode\build\test_conformance\clcpp\basic\Release\test_cpp_basic.exe [<test name>*] [pid<num>] [id<num>] [<device type>]
        <test name>     One or more of: (wildcard character '*') (default *)
        pid<num>        Indicates platform at index <num> should be used (default 0).
        id<num>         Indicates device at index <num> should be used (default 0).
        <device_type>   cpu|gpu|accelerator|<CL_DEVICE_TYPE_*> (default CL_DEVICE_TYPE_DEFAULT)

        NOTE: You may pass environment variable CL_CONFORMANCE_RESULTS_FILENAME (currently '<undefined>')
              to save results to JSON file.

Test names:
        test_case_1
        test_case_2
        ...
```

Several test options can also be controlled using environmental variables. The list of such options are:

| Command-line option                 | Environment variable  |
|-------------------------------------|-----------------------|
| `--compilation-mode`                | `COMPILATION_MODE`    |
| `--compilation-program`             | `COMPILATION_PROGRAM` |

In reality, the trailing part of the sample command-line printed by help: `[pid<num>] [id<num>] [<device type>]` can also be controlled by environmental variables, but the format of these are aligned to CTest resource spec variables and are primarly used by the [CTest test driver](#CTest-test-driver).They aren't intended to be set by users manually, rather via a resource spec file.

## Offline Compilation

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

A [utility CMake script](../cmake/clang_offline_compiler.cmake) is provided for convenience which wraps an upstream Clang compatible executable with the aforementioned interface and hides any extra user-defined command-line fragments from test harness.

## Python test driver

The tests can be run via Python using `test_conformance/run_conformance.py` which launches each test one after the other and generates a conformance report.
## CTest test driver

The tests can also be run using CTest. While this driver can't generate conformance reports aligning to the Python scripts (yet), it can be used to:

- test multiple types of devices concurrently (for convenience)
- test multiple devices of the same type in parallel (to speed up testing)

Parallel test execution allows for tighter edit-build-test loops while developing both the CTS or ICDs.

> The CTest test driver makes use of CTest's [Resource Allocation](https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation) feature, and as such requires CMake 3.16 to use. (Building the CTS can still be done using older versions.) Both CMake and CTest silently ignore the unknown properties and command-line flags related to this feature and will run tests with default settings.

When using the CTest driver, one must specify `-D CTEST_RESOURCE_SPEC_FILE:FILEPATH=<path_to_file>` on the command-line to tell both CMake and CTest what OpenCL resources are available for testing. The resource spec file format can't encode all the information the CTest test driver will need, but CTest is forgiving to extending the resource spec file with fields it doesn't need.

### The resource spec file contents

The important part of an ordinary resource spec file is the list of resources. The test driver equates one resource to a single OpenCL device. Resource names are arbitrary, but the driver will append the names of the resources to the tests' names, so picking a descriptive name is advised. The mandatory fields of a resource entry (needed by CTest) are: `id` and `slots`. The `id` is supposed to be a string by which tests can identify the resource and `slots` is a numeric value indicating some capacity of the resource.

- The test driver will expect `id` to be a carefully formatted string from which the tests will extract the platform and device ids and the device type to use. (Essentially the `[pid<num>] [id<num>] [<device type>]` part of the command-line interface.)
  - The format of this string will be matched using the `pid([0-9]+)_id([0-9]+)_(default|cpu|gpu|accelerator|custom|all)` regular expression. For eg. `pid0_id0_cpu` is a valid entry.
- The driver will equate `slots` to be the available device memory in MBs. This value will be used to make sure that the device isn't oversubscribed.

_(Note: currently tests can only declare their device memory usage statically and globally as a scalar. Each test is assumed to consume the same amount of device memory. The default is set in the global `CMakeLists.txt` file and can be overriden using `-D OPENCL_CTS_TEST_SLOTS=<num>`. Later revisions of the test driver can improve on this.)_

Fields added for the test driver which are CTS specific are `languages` and `compilers`.

- `languages` is a list of objects inside each resource detailing how various device-side languages are consumed. The objects' fields are:
  - `lang` currently can only be `clc` denoting OpenCL C
  - `version` can be a `1.0`, `1.1`, `1.2`, `2.0`, `2.1`, `2.2`, `3.0` denoting matching OpenCL C versions
  - `mode` can be `online`, `binary` or `spir-v`
  - `program` as an optional entry for non-online compiled languages, so languages that are to be tested but the runtime doesn't provide a compiler as part of the runtime itself. The value of this entry shall match exactly one name in the `compilers` array.
- `compilers` is a list of objects in the json root, with two fields in each object:
  - `name` is any name given to the compiler. Must match a `program` from the list of resources.
  - `program` is the command-line test harness will execute assuming it satisfies the interface of offline compilers.

An example resource spec file may look like the following:

```json
{
    "version": {
      "major": 1,
      "minor": 0
    },
    "local": [
      {
        "gfx900": [
          {
            "id": "pid0_id0_gpu",
            "slots": 16384,
            "languages": [
              {
                "lang": "clc",
                "version": "2.0",
                "mode": "online"
              }
            ]
          },
          {
            "id": "pid0_id1_gpu",
            "slots": 16384,
            "languages": [
              {
                "lang": "clc",
                "version": "2.0",
                "mode": "online"
              }
            ]
          }
        ],
        "cpu": [
          {
            "id": "pid1_id0_cpu",
            "slots": 16384,
            "languages": [
              {
                "lang": "clc",
                "version": "3.0",
                "mode": "online"
              },
              {
                "lang": "clc",
                "version": "3.0",
                "mode": "spir-v",
                "program": "clang_spir-v"
              }
            ]
          }
        ]
      }
    ],
    "compilers": [
      {
          "name": "clang_spir-v",
          "program": "/opt/Kitware/CMake/3.24.2/bin/cmake -D CLANG_EXE=/usr/bin/clang-15 -D CLANG_ARGS=\"-O2\\;--target=spirv64\\;-c\" -P /builds/ci/OpenCL-CTS/cmake/clang_offline_compiler.cmake -- --compilation-cache-mode overwrite"
      }
    ]
  }

```

### Notes on the CTest driver

- Tests to run can be selected by regular CTest [command-line flags](https://cmake.org/cmake/help/latest/manual/ctest.1.html#options)
  - Test names are `<module_name>-<test_name>-<resource_name>`
  - Test are labeled by: `module_name`, `resource_name`

For eg. a system with a CPU and and IGP, named as `cpu` and `igp` in the json respectively when wanting to run the image tests only and only on the IGP, one may run:

- `ctest.exe --label-regex images --label-exclude cpu --parallel ${env:NUMBER_OF_PROCESSORS}` in PowerShell on Windows
- `` ctest --label-regex images --label-exclude cpu --parallel `nproc` `` in bash on Linux