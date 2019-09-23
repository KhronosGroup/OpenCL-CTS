
set(HARNESS_COMPAT_SOURCES
    harness/errorHelpers.c
    harness/imageHelpers.cpp
    harness/kernelHelpers.c
    harness/testHarness.c
)

set_source_files_properties(${HARNESS_COMPAT_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(harness-compat STATIC ${HARNESS_COMPAT_SOURCES})

target_link_libraries(harness-compat harness-common)

