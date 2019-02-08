add_executable(conformance_test_allocations
        main.cpp
        allocation_execute.cpp
        allocation_fill.cpp
        allocation_functions.cpp
        allocation_utils.cpp
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/threadTesting.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/typeWrappers.cpp
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/crc32.c
)

set_source_files_properties(
        main.cpp
        allocation_execute.cpp
        allocation_fill.cpp
        allocation_functions.cpp
        allocation_utils.cpp
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/threadTesting.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_allocations
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_allocations)
install(TARGETS conformance_test_allocations
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
