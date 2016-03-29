set(MODULE_NAME CONVERSIONS)

# VS2005 AMD64 WAR (basic_test_conversions.c causes internal compiler error)
if(MSVC)
set(CMAKE_C_FLAGS_RELEASE "/Od /Ob0")
set(CMAKE_CXX_FLAGS_RELEASE "/Od /Ob0")
else(MSVC)
set(CMAKE_C_FLAGS_RELEASE "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
endif(MSVC)

set (${MODULE_NAME}_SOURCES
      Sleep.c test_conversions.c basic_test_conversions.c
      ../../test_common/harness/ThreadPool.c
      ../../test_common/harness/rounding_mode.c
      ../../test_common/harness/mt19937.c
      ../../test_common/harness/msvc9.c
      ../../test_common/harness/mingw_compat.c
      ../../test_common/harness/errorHelpers.c
      ../../test_common/harness/parseParameters.cpp
      ../../test_common/harness/kernelHelpers.c
      ../../test_common/harness/testHarness.c
)

if(ANDROID)
    list(APPEND CONVERSIONS_SOURCES fplib.c)
endif(ANDROID)

if(WIN32)
set_source_files_properties(
    ${MODULE_NAME}_SOURCES
        PROPERTIES LANGUAGE CXX)
endif(WIN32)

if(NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)
# -march is needed for CPU atomics, default arch on gcc is i386
# that does not support atomics.
set_source_files_properties(
        ${MODULE_NAME}_SOURCES
        COMPILE_FLAGS -march=i686)
endif(NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)

if(NOT ANDROID)
set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        COMPILE_FLAGS -msse2)
endif(NOT ANDROID)

include(../CMakeCommon.txt)
