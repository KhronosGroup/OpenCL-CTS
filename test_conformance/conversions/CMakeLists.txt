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
        ../../test_common/harness/parseParameters.cpp
)

set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/ThreadPool.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/errorHelpers.c
        PROPERTIES LANGUAGE CXX)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86$)|(X86$)")
if(NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)
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
endif()
endif()

if(NOT MSVC)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        COMPILE_FLAGS -msse2)
endif()
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
set_source_files_properties(
        Sleep.c test_conversions.c basic_test_conversions.c
        COMPILE_FLAGS "/arch:IA32")
endif()

TARGET_LINK_LIBRARIES(conformance_test_conversions
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_conversions)
install(TARGETS conformance_test_conversions
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
