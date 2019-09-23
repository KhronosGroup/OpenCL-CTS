
set(HARNESS_COMMON_SOURCES
    harness/threadTesting.c
    harness/typeWrappers.cpp
    harness/mt19937.c
    harness/conversions.c
    harness/rounding_mode.c
    harness/msvc9.c
)

set_source_files_properties(${HARNESS_COMMON_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(harness-common STATIC ${HARNESS_COMMON_SOURCES})

set(HARNESS_SOURCES
    harness/crc32.c
    harness/errorHelpers.c
    harness/genericThread.cpp
    harness/imageHelpers.cpp
    harness/kernelHelpers.c
    harness/deviceInfo.cpp
    harness/os_helpers.cpp
    harness/parseParameters.cpp
    harness/testHarness.c
    harness/ThreadPool.c
    miniz/miniz.c
)    

set_source_files_properties(${HARNESS_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(harness STATIC ${HARNESS_SOURCES})

target_link_libraries(harness harness-common)
