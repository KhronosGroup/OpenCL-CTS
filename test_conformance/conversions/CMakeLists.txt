set(MODULE_NAME CONVERSIONS)

# VS2005 AMD64 WAR (basic_test_conversions.cpp causes internal compiler error)
if(MSVC)
set(CMAKE_C_FLAGS_RELEASE "/Od /Ob0")
set(CMAKE_CXX_FLAGS_RELEASE "/Od /Ob0")
else(MSVC)
set(CMAKE_C_FLAGS_RELEASE "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
endif(MSVC)

set (${MODULE_NAME}_SOURCES
      Sleep.cpp test_conversions.cpp basic_test_conversions.cpp
)

if(CMAKE_COMPILER_IS_GNUCC AND ${CLConform_TARGET_ARCH} EQUAL "ARM")
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
