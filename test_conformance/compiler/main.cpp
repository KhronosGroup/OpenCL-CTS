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
#include "harness/compat.h"

#include "harness/testHarness.h"
#include "procs.h"
#include <stdio.h>
#include <string.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST(load_program_source),
    ADD_TEST(load_multistring_source),
    ADD_TEST(load_two_kernel_source),
    ADD_TEST(load_null_terminated_source),
    ADD_TEST(load_null_terminated_multi_line_source),
    ADD_TEST(load_null_terminated_partial_multi_line_source),
    ADD_TEST(load_discreet_length_source),
    ADD_TEST(get_program_source),
    ADD_TEST(get_program_build_info),
    ADD_TEST(get_program_info),

    ADD_TEST(large_compile),
    ADD_TEST(async_build),

    ADD_TEST(options_build_optimizations),
    ADD_TEST(options_build_macro),
    ADD_TEST(options_build_macro_existence),
    ADD_TEST(options_include_directory),
    ADD_TEST(options_denorm_cache),

    ADD_TEST(preprocessor_define_udef),
    ADD_TEST(preprocessor_include),
    ADD_TEST(preprocessor_line_error),
    ADD_TEST(preprocessor_pragma),

    ADD_TEST(opencl_c_versions),
    ADD_TEST(compiler_defines_for_extensions),
    ADD_TEST(image_macro),

    ADD_TEST(simple_compile_only),
    ADD_TEST(simple_static_compile_only),
    ADD_TEST(simple_extern_compile_only),
    ADD_TEST(simple_compile_with_callback),
    ADD_TEST(simple_embedded_header_compile),
    ADD_TEST(simple_link_only),
    ADD_TEST(two_file_regular_variable_access),
    ADD_TEST(two_file_regular_struct_access),
    ADD_TEST(two_file_regular_function_access),
    ADD_TEST(simple_link_with_callback),
    ADD_TEST(simple_embedded_header_link),
    ADD_TEST(execute_after_simple_compile_and_link),
    ADD_TEST(execute_after_simple_compile_and_link_no_device_info),
    ADD_TEST(execute_after_simple_compile_and_link_with_defines),
    ADD_TEST(execute_after_simple_compile_and_link_with_callbacks),
    ADD_TEST(execute_after_simple_library_with_link),
    ADD_TEST(execute_after_two_file_link),
    ADD_TEST(execute_after_embedded_header_link),
    ADD_TEST(execute_after_included_header_link),
    ADD_TEST(execute_after_serialize_reload_object),
    ADD_TEST(execute_after_serialize_reload_library),
    ADD_TEST(simple_library_only),
    ADD_TEST(simple_library_with_callback),
    ADD_TEST(simple_library_with_link),
    ADD_TEST(two_file_link),
    ADD_TEST(multi_file_libraries),
    ADD_TEST(multiple_files),
    ADD_TEST(multiple_libraries),
    ADD_TEST(multiple_files_multiple_libraries),
    ADD_TEST(multiple_embedded_headers),

    ADD_TEST(program_binary_type),
    ADD_TEST(compile_and_link_status_options_log),

    ADD_TEST_VERSION(pragma_unroll, Version(2, 0)),

    ADD_TEST_VERSION(features_macro, Version(3, 0)),
    ADD_TEST(features_macro_coupling),

    ADD_TEST(unload_valid),
    // ADD_TEST(unload_invalid), // disabling temporarily, see GitHub #977
    ADD_TEST(unload_repeated),
    ADD_TEST(unload_compile_unload_link),
    ADD_TEST(unload_build_unload_create_kernel),
    ADD_TEST(unload_link_different),
    ADD_TEST(unload_build_threaded),
    ADD_TEST(unload_build_info),
    ADD_TEST(unload_program_binaries),

};

const int test_num = ARRAY_SIZE(test_list);

int main(int argc, const char *argv[])
{
    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}
