if(WIN32)
list(APPEND CLConform_LIBRARIES d3d9.lib dxva2.lib )
endif(WIN32)

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
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/typeWrappers.cpp
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/crc32.c
    )

add_executable(conformance_test_media_surface_sharing
        ${MEDIA_SOURCES})

set_source_files_properties(
        ${MEDIA_SOURCES}
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_media_surface_sharing
        ${CLConform_LIBRARIES})
