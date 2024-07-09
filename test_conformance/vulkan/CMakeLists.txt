set (MODULE_NAME VULKAN)

list(APPEND CLConform_LIBRARIES vulkan_wrapper)
set(CMAKE_CXX_FLAGS "-fpermissive")
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
        ../../test_common/harness/genericThread.cpp
        ../../test_common/harness/errorHelpers.cpp
        ../../test_common/harness/testHarness.cpp
        ../../test_common/harness/kernelHelpers.cpp
        ../../test_common/harness/mt19937.cpp
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/deviceInfo.cpp
        ../../test_common/harness/crc32.cpp
    )

set_source_files_properties(
    ${${MODULE_NAME}_SOURCES}
    PROPERTIES LANGUAGE CXX)
include_directories("../common/vulkan_wrapper")

add_subdirectory(shaders)

include(../CMakeCommon.txt)
