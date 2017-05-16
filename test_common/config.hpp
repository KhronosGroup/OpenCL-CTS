//
// Copyright (c) 2017 The Khronos Group Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef TEST_COMMON_CONFIG_HPP
#define TEST_COMMON_CONFIG_HPP

// Enable development options for OpenCL C++ tests (test_conformance/clpp)
// #define DEVELOPMENT
#if defined(CLPP_DEVELOPMENT_OPTIONS) && !defined(DEVELOPMENT)
    #define DEVELOPMENT
#endif

#ifdef DEVELOPMENT
    // If defined OpenCL C++ tests only checks if OpenCL C++ kernels compiles correctly
    // #define ONLY_SPIRV_COMPILATION
    #if defined(CLPP_DEVELOPMENT_ONLY_SPIRV_COMPILATION) && !defined(ONLY_SPIRV_COMPILATION)
        #define ONLY_SPIRV_COMPILATION
    #endif

    #ifndef ONLY_SPIRV_COMPILATION
        // If defined OpenCL C++ tests are run using OpenCL C kernels
        // #define USE_OPENCLC_KERNELS
        #if defined(CLPP_DEVELOPMENT_USE_OPENCLC_KERNELS) && !defined(USE_OPENCLC_KERNELS)
            #define USE_OPENCLC_KERNELS
        #endif
    #endif    
#endif

#endif // TEST_COMMON_CONFIG_HPP