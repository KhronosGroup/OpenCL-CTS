test_conformance/spirv_new README
==================================

The text versions of the spirv files are present in `conformance-tests/test_conformance/spriv_new/spirv_asm`.
These text files have been used to generate the binaries in `spirv_bin` using the assembler from `spirv-tools`.

The absolute path to `spirv_bin` needs to be passed after `--spirv-binaries-path` token for the test to find the SPIRV binaries.

An example invocation looks like the following:

```
./test_conformance/spirv_new/test_conformance_spirv_new --spirv-binaries-path /home/user/workspace/conformance-tests/test_conformance/spirv_new/spirv_bin/ [other options]
```
