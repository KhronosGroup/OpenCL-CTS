# VS2005 AMD64 WAR (basic_test_conversions.c causes internal compiler error)
if(MSVC)
set(CMAKE_C_FLAGS_RELEASE "/Od /Ob0")
set(CMAKE_CXX_FLAGS_RELEASE "/Od /Ob0")
else(MSVC)
set(CMAKE_C_FLAGS_RELEASE "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
endif(MSVC)

add_executable(conformance_test_conversions
        Sleep.c test_conversions.c basic_test_conversions.c
        ../../test_common/harness/ThreadPool.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
)

if(WIN32)
set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        ../../test_common/harness/ThreadPool.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/errorHelpers.c
        PROPERTIES LANGUAGE CXX)
endif(WIN32)

if(NOT CMAKE_CL_64 AND NOT MSVC)  
# -march is needed for CPU atomics, default arch on gcc is i386 
# that does not support atomics. 
set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        ../../test_common/harness/ThreadPool.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/errorHelpers.c
        COMPILE_FLAGS -march=i686)
endif(NOT CMAKE_CL_64 AND NOT MSVC)            

set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        COMPILE_FLAGS -std=c99)


set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        COMPILE_FLAGS -msse2)

TARGET_LINK_LIBRARIES(conformance_test_conversions
        ${CLConform_LIBRARIES})
