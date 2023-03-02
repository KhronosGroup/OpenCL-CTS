set(VULKAN_WRAPPER_SOURCES
    vulkan_wrapper.cpp
    opencl_vulkan_wrapper.cpp
    vulkan_utility.cpp
    vulkan_list_map.cpp
)

# needed by Vulkan wrapper to compile
add_cxx_flag_if_supported(-Wno-unused-variable)
add_cxx_flag_if_supported(-Wno-error=terminate)
add_cxx_flag_if_supported(-Wno-error=unused-function)
add_cxx_flag_if_supported(-Wno-error=return-type)

link_directories(${CLConform_VULKAN_LIBRARIES_DIR})

list(APPEND CLConform_INCLUDE_DIR ${VULKAN_INCLUDE_DIR})

add_library(vulkan_wrapper SHARED ${VULKAN_WRAPPER_SOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories (${CLConform_INCLUDE_DIR})
