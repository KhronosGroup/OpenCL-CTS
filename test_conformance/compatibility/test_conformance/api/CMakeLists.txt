set(MODULE_NAME COMPATIBILITY_API)

set(${MODULE_NAME}_SOURCES
         main.c
         test_bool.c
         test_retain.cpp
         test_retain_program.c
         test_queries.cpp
         test_create_kernels.c
         test_kernels.c
         test_api_min_max.c
         test_kernel_arg_changes.cpp
         test_kernel_arg_multi_setup.cpp
         test_binary.cpp
         test_native_kernel.cpp
         test_mem_objects.cpp
         test_create_context_from_type.cpp
         test_device_min_data_type_align_size_alignment.cpp
         test_platform.cpp
         test_mem_object_info.cpp
         test_null_buffer_arg.c
         test_kernel_arg_info.c
         test_queue_properties.cpp
)

set(${MODULE_NAME}_LIBS harness-compat)

include(../../../CMakeCommon.txt)
