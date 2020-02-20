set(MODULE_NAME API)

set(${MODULE_NAME}_SOURCES
         main.cpp
         test_bool.cpp
         test_retain.cpp
         test_retain_program.cpp
         test_queries.cpp
         test_queries_compatibility.cpp
         test_create_kernels.cpp
         test_kernels.cpp
         test_api_min_max.cpp
         test_kernel_arg_changes.cpp
         test_kernel_arg_multi_setup.cpp
         test_binary.cpp
         test_native_kernel.cpp
         test_mem_objects.cpp
         test_create_context_from_type.cpp
         test_device_min_data_type_align_size_alignment.cpp
         test_platform.cpp
         test_kernel_arg_info.cpp
         test_kernel_arg_info_compatibility.cpp
         test_null_buffer_arg.cpp
         test_mem_object_info.cpp
         test_queue_hint.cpp
         test_queue_properties.cpp
         test_sub_group_dispatch.cpp
         test_clone_kernel.cpp
         test_zero_sized_enqueue.cpp
)

include(../CMakeCommon.txt)
