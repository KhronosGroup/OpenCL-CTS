set(MODULE_NAME CONVERSIONS)

set (${MODULE_NAME}_SOURCES
      Sleep.cpp test_conversions.cpp basic_test_conversions.cpp
)

if(CMAKE_COMPILER_IS_GNUCXX AND "${CLConform_TARGET_ARCH}" STREQUAL "ARM")
    list(APPEND ${MODULE_NAME}_SOURCES fplib.cpp)
endif()

if(NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)
# -march is needed for CPU atomics, default arch on gcc is i386
# that does not support atomics.
set_source_files_properties(
        ${MODULE_NAME}_SOURCES
        COMPILE_FLAGS -march=i686)
endif(NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)

include(../CMakeCommon.txt)
