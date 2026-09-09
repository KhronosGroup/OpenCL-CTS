# Prefer CMake package targets because they preserve library locations when
# cross-compiling. Older SPIRV-Tools installations may only provide pkg-config.
set(SPIRV_TOOLS_TARGET_DIR "" CACHE PATH
    "Directory containing the target SPIRV-Tools CMake package")
if(SPIRV_TOOLS_TARGET_DIR)
    set("SPIRV-Tools_DIR" "${SPIRV_TOOLS_TARGET_DIR}")
endif()
find_package(SPIRV-Tools CONFIG QUIET)

if(TARGET SPIRV-Tools-static)
    set(CLConform_SPIRV_TOOLS_TARGET SPIRV-Tools-static)
elseif(TARGET SPIRV-Tools)
    set(CLConform_SPIRV_TOOLS_TARGET SPIRV-Tools)
elseif(TARGET SPIRV-Tools-shared)
    set(CLConform_SPIRV_TOOLS_TARGET SPIRV-Tools-shared)
else()
    include(FindPkgConfig)
    pkg_search_module(SPIRV_TOOLS QUIET IMPORTED_TARGET SPIRV-Tools)

    if(TARGET PkgConfig::SPIRV_TOOLS)
        set(CLConform_SPIRV_TOOLS_TARGET PkgConfig::SPIRV_TOOLS)
    endif()
endif()

function(target_link_spirv_tools target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Cannot link SPIRV-Tools to unknown target '${target}'")
    endif()

    if(NOT CLConform_SPIRV_TOOLS_TARGET)
        message(FATAL_ERROR
            "SPIRV-Tools was not found. Install its pkg-config or CMake package.")
    endif()

    if(TARGET spirv_tools_harness)
        target_link_libraries(${target} spirv_tools_harness)
    else()
        target_link_libraries(${target} ${CLConform_SPIRV_TOOLS_TARGET})
    endif()
endfunction()
