set(MODULE_NAME COMPILER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_build_helpers.cpp
    test_compile.cpp
    test_async_build.cpp
    test_build_options.cpp
    test_preprocessor.cpp
    test_opencl_c_versions.cpp
    test_image_macro.cpp
    test_compiler_defines_for_extensions.cpp
    test_pragma_unroll.cpp
    test_unload_platform_compiler.cpp
    test_feature_macro.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
load_program_source
load_multistring_source
load_two_kernel_source
load_null_terminated_source
load_null_terminated_multi_line_source
load_null_terminated_partial_multi_line_source
load_discreet_length_source
get_program_source
get_program_build_info
get_program_info
large_compile
async_build
options_build_optimizations
options_build_macro
options_build_macro_existence
options_include_directory
options_denorm_cache
preprocessor_define_udef
preprocessor_include
preprocessor_line_error
preprocessor_pragma
opencl_c_versions
compiler_defines_for_extensions
image_macro
simple_compile_only
simple_static_compile_only
simple_extern_compile_only
simple_compile_with_callback
simple_embedded_header_compile
simple_link_only
two_file_regular_variable_access
two_file_regular_struct_access
two_file_regular_function_access
simple_link_with_callback
simple_embedded_header_link
execute_after_simple_compile_and_link
execute_after_simple_compile_and_link_no_device_info
execute_after_simple_compile_and_link_with_defines
execute_after_simple_compile_and_link_with_callbacks
execute_after_simple_library_with_link
execute_after_two_file_link
execute_after_embedded_header_link
execute_after_included_header_link
execute_after_serialize_reload_object
execute_after_serialize_reload_library
simple_library_only
simple_library_with_callback
simple_library_with_link
two_file_link
multi_file_libraries
multiple_files
multiple_libraries
multiple_files_multiple_libraries
multiple_embedded_headers
program_binary_type
compile_and_link_status_options_log
pragma_unroll
features_macro
unload_valid
unload_repeated
unload_compile_unload_link
unload_build_unload_create_kernel
unload_link_different
unload_build_threaded
unload_build_info
unload_program_binaries
)

include(../CMakeCommon.cmake)

# Copy the required test include directories into the build directory.
if(NOT DEFINED COMPILER_TEST_RESOURCES)
    set(COMPILER_TEST_RESOURCES $<TARGET_FILE_DIR:${${MODULE_NAME}_OUT}>)
endif()
add_custom_command(
        COMMENT "Copying compiler test resources..."
        TARGET ${${MODULE_NAME}_OUT}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory
        ${COMPILER_TEST_RESOURCES}/includeTestDirectory
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory
        ${COMPILER_TEST_RESOURCES}/secondIncludeTestDirectory)
