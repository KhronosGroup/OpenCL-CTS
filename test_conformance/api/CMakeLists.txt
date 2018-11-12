add_executable(conformance_test_api
   main.c
   test_retain_program.c
   test_queries.cpp
   test_create_kernels.c
   test_kernels.c
   test_kernel_arg_info.c
   test_api_min_max.c
   test_kernel_arg_changes.cpp
   test_kernel_arg_multi_setup.cpp
   test_binary.cpp
   test_native_kernel.cpp
   test_create_context_from_type.cpp
   test_platform.cpp
   test_retain.cpp
   test_device_min_data_type_align_size_alignment.cpp
   test_mem_objects.cpp
   test_bool.c
   test_null_buffer_arg.c
   test_mem_object_info.cpp
   ../../test_common/harness/errorHelpers.c
   ../../test_common/harness/threadTesting.c
   ../../test_common/harness/testHarness.c
   ../../test_common/harness/imageHelpers.cpp
   ../../test_common/harness/kernelHelpers.c
   ../../test_common/harness/typeWrappers.cpp
   ../../test_common/harness/msvc9.c
   ../../test_common/harness/mt19937.c
   ../../test_common/harness/conversions.c
)

set_source_files_properties(
   main.c
   test_retain_program.c
   test_queries.cpp
   test_create_kernels.c
   test_kernels.c
   test_kernel_arg_info.c
   test_api_min_max.c
   test_kernel_arg_changes.cpp
   test_kernel_arg_multi_setup.cpp
   test_binary.cpp
   test_native_kernel.cpp
   test_create_context_from_type.cpp
   test_platform.cpp
   test_retain.cpp
   test_device_min_data_type_align_size_alignment.cpp
   test_queue_properties.cpp
   test_mem_objects.cpp
   test_bool.c
   test_null_buffer_arg.c
   test_mem_object_info.cpp
   ../../test_common/harness/errorHelpers.c
   ../../test_common/harness/threadTesting.c
   ../../test_common/harness/testHarness.c
   ../../test_common/harness/imageHelpers.cpp
   ../../test_common/harness/kernelHelpers.c
   ../../test_common/harness/typeWrappers.cpp
   ../../test_common/harness/msvc9.c
   ../../test_common/harness/mt19937.c
   ../../test_common/harness/conversions.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_api
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_api)
install(TARGETS conformance_test_api
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
