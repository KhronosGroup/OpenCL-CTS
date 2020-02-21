
set(HARNESS_COMMON_SOURCES
    harness/threadTesting.cpp
    harness/typeWrappers.cpp
    harness/mt19937.c
    harness/conversions.cpp
    harness/rounding_mode.cpp
    harness/msvc9.c
)

add_library(harness-common STATIC ${HARNESS_COMMON_SOURCES})

set(HARNESS_SOURCES
    harness/crc32.c
    harness/errorHelpers.cpp
    harness/genericThread.cpp
    harness/imageHelpers.cpp
    harness/kernelHelpers.cpp
    harness/deviceInfo.cpp
    harness/os_helpers.cpp
    harness/parseParameters.cpp
    harness/testHarness.cpp
    harness/ThreadPool.cpp
    miniz/miniz.c
)

add_library(harness STATIC ${HARNESS_SOURCES})

target_link_libraries(harness harness-common)
