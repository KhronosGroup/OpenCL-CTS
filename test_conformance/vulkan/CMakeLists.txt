set (MODULE_NAME VULKAN)

if(WIN32)
    list(APPEND CLConform_LIBRARIES vulkan-1)
else(WIN32)
    list(APPEND CLConform_LIBRARIES vulkan dl)
endif(WIN32)
set(CMAKE_CXX_FLAGS "-fpermissive")
if(WIN32) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
endif(WIN32)
if(MSVC)
    set (CLConform_VULKAN_LIBRARIES_DIR "${VULKAN_LIB_DIR}/${ARCH}_win32")
elseif(MINGW)
    set (CLConform_VULKAN_LIBRARIES_DIR "${VULKAN_LIB_DIR}/${ARCH}_mingw")
else(MSVC)
    set (CLConform_VULKAN_LIBRARIES_DIR "${VULKAN_LIB_DIR}/${ARCH}_Linux")
endif(MSVC)

link_directories(${CLConform_VULKAN_LIBRARIES_DIR})

list(APPEND CLConform_INCLUDE_DIR ${VULKAN_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories (${CLConform_INCLUDE_DIR})

set (${MODULE_NAME}_SOURCES
        main.cpp
        test_vulkan_interop_buffer.cpp
        test_vulkan_interop_image.cpp
        test_vulkan_api_consistency.cpp
        vulkan_interop_common/vulkan_wrapper.cpp
        vulkan_interop_common/vulkan_interop_common.cpp
        vulkan_interop_common/opencl_vulkan_wrapper.cpp
        vulkan_interop_common/vulkan_utility.cpp
        vulkan_interop_common/vulkan_list_map.cpp
        ../../test_common/harness/genericThread.cpp
        ../../test_common/harness/errorHelpers.cpp
        ../../test_common/harness/threadTesting.cpp
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
include_directories("./vulkan_interop_common/")

include(../CMakeCommon.txt)