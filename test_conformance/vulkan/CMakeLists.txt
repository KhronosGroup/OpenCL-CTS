set (MODULE_NAME VULKAN)

set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (${MODULE_NAME}_SOURCES
        main.cpp
        test_vulkan_interop_buffer.cpp
        test_vulkan_interop_image.cpp
        test_vulkan_api_consistency.cpp
        test_vulkan_api_consistency_for_3dimages.cpp
        test_vulkan_api_consistency_for_1dimages.cpp
        test_vulkan_platform_device_info.cpp
        vulkan_interop_common.cpp
        vulkan_test_base.h
    )

if(VULKAN_IS_SUPPORTED)
list(APPEND CLConform_LIBRARIES vulkan_wrapper)
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
endif(WIN32)

set (CLConform_VULKAN_LIBRARIES_DIR "${VULKAN_LIB_DIR}")

link_directories(${CLConform_VULKAN_LIBRARIES_DIR})

list(APPEND CLConform_INCLUDE_DIR ${VULKAN_INCLUDE_DIR})

include_directories (${CLConform_INCLUDE_DIR})

include_directories("../common/vulkan_wrapper")

add_subdirectory(shaders)
add_compile_definitions(VULKAN_IS_SUPPORTED)

endif()

include(../CMakeCommon.txt)

if(VULKAN_IS_SUPPORTED)
    add_dependencies(${${MODULE_NAME}_OUT} vulkan_shaders)
endif()
