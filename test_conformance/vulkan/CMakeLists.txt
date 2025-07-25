set (MODULE_NAME VULKAN)

list(APPEND CLConform_LIBRARIES vulkan_wrapper)
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
endif(WIN32)

set (CLConform_VULKAN_LIBRARIES_DIR "${VULKAN_LIB_DIR}")

link_directories(${CLConform_VULKAN_LIBRARIES_DIR})

list(APPEND CLConform_INCLUDE_DIR ${VULKAN_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories (${CLConform_INCLUDE_DIR})

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

include_directories("../common/vulkan_wrapper")

add_subdirectory(shaders)

include(../CMakeCommon.txt)

add_dependencies(${${MODULE_NAME}_OUT} vulkan_shaders)
