if(WIN32)
list(APPEND CLConform_LIBRARIES d3d9.lib dxva2.lib )

set (MEDIA_SOURCES
        main.cpp
        wrappers.cpp
        utils.cpp
        test_create_context.cpp
        test_functions_api.cpp
        test_functions_kernel.cpp
        test_get_device_ids.cpp
        test_interop_sync.cpp
        test_memory_access.cpp
        test_other_data_types.cpp
    )

add_executable(conformance_test_media_surface_sharing
        ${MEDIA_SOURCES})

set_source_files_properties(
        ${MEDIA_SOURCES}
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_media_surface_sharing harness
        ${CLConform_LIBRARIES})

endif(WIN32)
