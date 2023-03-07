set(VULKAN_WRAPPER_SOURCES
    vulkan_wrapper.cpp
    opencl_vulkan_wrapper.cpp
    vulkan_utility.cpp
    vulkan_list_map.cpp
)

# needed by Vulkan wrapper to compile
add_cxx_flag_if_supported(-Wmisleading-indentation)
add_cxx_flag_if_supported(-Wno-narrowing)
add_cxx_flag_if_supported(-Wno-format)
add_cxx_flag_if_supported(-Wno-error)
add_cxx_flag_if_supported(-Wno-error=cpp) # Allow #warning directive
add_cxx_flag_if_supported(-Wno-error=unknown-pragmas) # Issue #785
add_cxx_flag_if_supported(-Wno-error=asm-operand-widths) # Issue #784
add_cxx_flag_if_supported(-Wno-unused-variable)
add_cxx_flag_if_supported(-Wno-error=terminate)
add_cxx_flag_if_supported(-Wno-error=unused-function)
add_cxx_flag_if_supported(-Wno-error=return-type)

link_directories(${CLConform_VULKAN_LIBRARIES_DIR})

list(APPEND CLConform_INCLUDE_DIR ${VULKAN_INCLUDE_DIR})

add_library(vulkan_wrapper STATIC ${VULKAN_WRAPPER_SOURCES})

if(ANDROID)
    target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
elseif(WIN32)
    target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_METAL_EXT)
elseif(UNIX)
    # Variable taken from Vulkan samples, commented out due to lack of WSI
    # Choose WSI based on VKB_WSI_SELECTION
    #if (VKB_WSI_SELECTION STREQUAL XCB OR VKB_WSI_SELECTION STREQUAL XLIB OR VKB_WSI_SELECTION STREQUAL WAYLAND)
    #    find_package(PkgConfig REQUIRED)
    #endif()
    #if (VKB_WSI_SELECTION STREQUAL XCB)
    #    pkg_check_modules(XCB xcb REQUIRED)
    #    if (XCB_FOUND)
    #        target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_XCB_KHR)
    #    endif()
    #elseif (VKB_WSI_SELECTION STREQUAL XLIB)
    #    pkg_check_modules(X11 x11 REQUIRED)
    #    if (X11_FOUND)
    #        target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_XLIB_KHR)
    #    endif()
    #elseif (VKB_WSI_SELECTION STREQUAL WAYLAND)
    #    pkg_check_modules(WAYLAND wayland-client REQUIRED)
    #    if (WAYLAND_FOUND)
    #        target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_WAYLAND_KHR)
    #    endif()
    #elseif (VKB_WSI_SELECTION STREQUAL D2D)
    #    set(DIRECT_TO_DISPLAY TRUE)
    #    set(DIRECT_TO_DISPLAY TRUE PARENT_SCOPE)
    #    target_compile_definitions(vulkan_wrapper PUBLIC VK_USE_PLATFORM_DISPLAY_KHR)
    #else()
    #    message(FATAL_ERROR "Unknown WSI")
    #endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories (${CLConform_INCLUDE_DIR})

if (NOT WIN32)
target_link_libraries(vulkan_wrapper dl)
endif()
