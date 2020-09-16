if(WIN32)
list(APPEND CLConform_LIBRARIES d3d9.lib dxva2.lib )
set(MODULE_NAME MEDIA_SURFACE_SHARING)

set(${MODULE_NAME}_SOURCES
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

set_source_files_properties(
        ${MODULE_NAME}_SOURCES
        PROPERTIES LANGUAGE CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(../../test_conformance/CMakeCommon.txt)
endif(WIN32)
