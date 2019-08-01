To run the conformance tests you need SPIR-V binaries.

If you are using a conformance package then the binaries are included in the
package. If you are using conformance tests from gitlab repositories then the
binaries need to be picked up from Khronos SVN URL mentioned below:

https://cvs.khronos.org/svn/repos/OpenCL/trunk/Khronos/spirv/spirv10_2015.11.25.zip


Alternatively you can check out and build all of the below repositories.

1. SPIRV-LLVM
LLVM with support for SPIR-V (required by clang compiler)
Repository: https://gitlab.khronos.org/opencl/SPIRV-LLVM
Branch: spec_constants 
Notes: spirv-3.6.1 is a main branch with support for OpenCL C++ kernel language,
  spec_constants is based on it, but it adds support for specialization constants.

2. Clang 
Clang with support for OpenCL C++ kernel language
Repository: https://gitlab.khronos.org/opencl/clang
Branch: spec_constants 
Notes: spirv-1.1 is a main branch with support for OpenCL C++ kernel language, 
  spec_constants is based on it, but it adds support for specialization constants.

3. libclcxx
OpenCL C++ Standard Library
Repository: https://gitlab.khronos.org/opencl/libclcxx 
Branch: lit_tests_cl22   
Notes: lit_tests_cl22 branch includes both LIT tests and changes introduced in 
  spec_constants branch, that is, implementation of Specialization Constants Library.   

4. OpenCL 2.2 headers
OpenCL 2.2 headers
Repository: https://gitlab.khronos.org/opencl/headers 
Branch: opencl22 

5. OpenCL ICD (with 2.2 support)
OpenCL ICD 
Repository: https://gitlab.khronos.org/opencl/icd 
Branch: dev_cl22 
