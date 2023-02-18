set(MODULE_NAME CL_KHR_EXTERNAL_SEMAPHORE)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_external_semaphore.cpp
    vulkan_wrapper/vulkan_wrapper.cpp
    vulkan_wrapper/vulkan_interop_common.cpp
    vulkan_wrapper/opencl_vulkan_wrapper.cpp
    vulkan_wrapper/vulkan_utility.cpp
    vulkan_wrapper/vulkan_list_map.cpp
)

# needed by Vulkan wrapper to link
if(WIN32)
    list(APPEND CLConform_LIBRARIES vulkan-1)
else(WIN32)
    list(APPEND CLConform_LIBRARIES vulkan dl)
endif(WIN32)
set(CMAKE_CXX_FLAGS "-fpermissive")
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
endif(WIN32)

# needed by Vulkan wrapper to compile
add_cxx_flag_if_supported(-Wno-unused-variable)
add_cxx_flag_if_supported(-Wno-error=terminate)
add_cxx_flag_if_supported(-Wno-error=unused-function)
add_cxx_flag_if_supported(-Wno-error=return-type)

include(../../CMakeCommon.txt)
