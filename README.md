# OpenCL Conformance Test Suite (CTS)

This is the OpenCL CTS for all versions of the Khronos
[OpenCL](https://www.khronos.org/opencl/) standard.

## Building the CTS

For a detailed build guide, refer to the [relevant section of the docs](docs/BUILD.md).

## Generating a Conformance Report

The Khronos [Conformance Process Document](https://members.khronos.org/document/dl/911)
details the steps required for a conformance submissions.
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
a questions please file an issue in the GitHub project. First time contributors
will be required to sign the Khronos Contributor License Agreement (CLA) before
their PR can be merged.

PRs to the repository are required to be `clang-format` clean to pass CI.
Developers can either use the `git-clang-format` tool locally to verify this
before contributing, or update their PR based on the diff provided by a failing
CI job.
