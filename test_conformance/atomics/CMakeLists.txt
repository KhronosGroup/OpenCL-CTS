add_executable(test_atomics
        main.c
        test_atomics.cpp
        test_indexed_cases.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/threadTesting.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/conversions.c
        ../../test_common/harness/msvc9.c
)

set_source_files_properties(
        main.c
        test_atomics.cpp
        test_indexed_cases.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/threadTesting.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/conversions.c
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(test_atomics
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_atomics)
install(TARGETS test_atomics
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
