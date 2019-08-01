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
#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn    basefn_list[] = {
    test_load_program_source,
    test_load_multistring_source,
    test_load_two_kernel_source,
    test_load_null_terminated_source,
    test_load_null_terminated_multi_line_source,
    test_load_null_terminated_partial_multi_line_source,
    test_load_discreet_length_source,
    test_get_program_source,
    test_get_program_build_info,
    test_get_program_info,

    test_large_compile,
    test_async_build_pieces,

    test_options_optimizations,
    test_options_build_macro,
    test_options_build_macro_existence,
    test_options_include_directory,
    test_options_denorm_cache,

    test_preprocessor_define_udef,
    test_preprocessor_include,
    test_preprocessor_line_error,
    test_preprocessor_pragma,

    test_compiler_defines_for_extensions,
    test_image_macro,

    test_simple_compile_only,
    test_simple_static_compile_only,
    test_simple_extern_compile_only,
    test_simple_compile_with_callback,
    test_simple_embedded_header_compile,
    test_simple_link_only,
    test_two_file_regular_variable_access,
    test_two_file_regular_struct_access,
    test_two_file_regular_function_access,
    test_simple_link_with_callback,
    test_simple_embedded_header_link,
    test_execute_after_simple_compile_and_link,
    test_execute_after_simple_compile_and_link_no_device_info,
    test_execute_after_simple_compile_and_link_with_defines,
    test_execute_after_simple_compile_and_link_with_callbacks,
    test_execute_after_simple_library_with_link,
    test_execute_after_two_file_link,
    test_execute_after_embedded_header_link,
    test_execute_after_included_header_link,
    test_execute_after_serialize_reload_object,
    test_execute_after_serialize_reload_library,
    test_simple_library_only,
    test_simple_library_with_callback,
    test_simple_library_with_link,
    test_two_file_link,
    test_multi_file_libraries,
    test_multiple_files,
    test_multiple_libraries,
    test_multiple_files_multiple_libraries,
    test_multiple_embedded_headers,

    test_program_binary_type,
    test_compile_and_link_status_options_log
};


const char    *basefn_names[] = {
    "load_program_source",
    "load_multistring_source",
    "load_two_kernel_source",
    "load_null_terminated_source",
    "load_null_terminated_multi_line_source",
    "load_null_terminated_partial_multi_line_source",
    "load_discreet_length_source",
    "get_program_source",
    "get_program_build_info",
    "get_program_info",

    "large_compile",
    "async_build",

    "options_build_optimizations",
    "options_build_macro",
    "options_build_macro_existence",
    "options_include_directory",
    "options_denorm_cache",

    "preprocessor_define_udef",
    "preprocessor_include",
    "preprocessor_line_error",
    "preprocessor_pragma",

    "compiler_defines_for_extensions",
    "image_macro",

    "simple_compile_only",
    "simple_static_compile_only",
    "simple_extern_compile_only",
    "simple_compile_with_callback",
    "simple_embedded_header_compile",
    "simple_link_only",
    "two_file_regular_variable_access",
    "two_file_regular_struct_access",
    "two_file_regular_function_access",
    "simple_link_with_callback",
    "simple_embedded_header_link",
    "execute_after_simple_compile_and_link",
    "execute_after_simple_compile_and_link_no_device_info",
    "execute_after_simple_compile_and_link_with_defines",
    "execute_after_simple_compile_and_link_with_callbacks",
    "execute_after_simple_library_with_link",
    "execute_after_two_file_link",
    "execute_after_embedded_header_link",
    "execute_after_included_header_link",
    "execute_after_serialize_reload_object",
    "execute_after_serialize_reload_library",
    "simple_library_only",
    "simple_library_with_callback",
    "simple_library_with_link",
    "two_file_link",
    "multi_file_libraries",
    "multiple_files",
    "multiple_libraries",
    "multiple_files_multiple_libraries",
    "multiple_embedded_headers",
    "program_binary_type",
    "compile_and_link_status_options_log",

    "all"
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


