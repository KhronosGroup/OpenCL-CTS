set(MODULE_NAME API)

find_package(Python3 COMPONENTS Interpreter QUIET)

if (SPIRV_INCLUDE_DIR)
    set(SPIRV_CAPABILITY_FILE
            ${CMAKE_CURRENT_BINARY_DIR}/spirv_capability_deps.def)
endif (SPIRV_INCLUDE_DIR)

set(${MODULE_NAME}_SOURCES
         main.cpp
         negative_platform.cpp
         negative_queue.cpp
         negative_enqueue_map_image.cpp
         test_api_consistency.cpp
         test_bool.cpp
         test_retain.cpp
         test_retain_program.cpp
         test_queries.cpp
         test_create_kernels.cpp
         test_kernels.cpp
         test_kernel_local_memory_size.cpp
         test_kernel_private_memory_size.cpp
         test_api_min_max.cpp
         test_kernel_arg_changes.cpp
         test_kernel_arg_multi_setup.cpp
         test_kernel_attributes.cpp
         test_binary.cpp
         test_native_kernel.cpp
         test_mem_objects.cpp
         test_create_context_from_type.cpp
         test_device_min_data_type_align_size_alignment.cpp
         test_platform.cpp
         test_kernel_arg_info.cpp
         test_null_buffer_arg.cpp
         test_mem_object_info.cpp
         test_min_image_formats.cpp
         test_queue.cpp
         test_queue_hint.cpp
         test_queue_properties.cpp
         test_sub_group_dispatch.cpp
         test_clone_kernel.cpp
         test_zero_sized_enqueue.cpp
         test_context_destructor_callback.cpp
         test_mem_object_properties_queries.cpp
         test_queue_properties_queries.cpp
         test_pipe_properties_queries.cpp
         test_wg_suggested_local_work_size.cpp
         test_device_command_queue.cpp
         test_spirv_queries.cpp
         ${SPIRV_CAPABILITY_FILE}
)

if (SPIRV_INCLUDE_DIR)
    set_gnulike_module_compile_flags("-DSPIRV_FILES_AVAILABLE")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spirv_capability_deps.def
        COMMENT "Generating spirv_capability_deps.def..."
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_spirv_capability_deps.py
                --grammar "${SPIRV_INCLUDE_DIR}/include/spirv/unified1/spirv.core.grammar.json"
                --output "${CMAKE_CURRENT_BINARY_DIR}/spirv_capability_deps.def"
        DEPENDS generate_spirv_capability_deps.py "${SPIRV_INCLUDE_DIR}/include/spirv/unified1/spirv.core.grammar.json"
        USES_TERMINAL
        VERBATIM)
endif (SPIRV_INCLUDE_DIR)

include(../CMakeCommon.txt)

target_include_directories(${${MODULE_NAME}_OUT} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
