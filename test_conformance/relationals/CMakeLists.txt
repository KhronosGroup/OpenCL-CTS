add_executable(conformance_test_relationals 
        main.c
	test_relationals.cpp
    test_comparisons_float.cpp
    test_comparisons_double.cpp
	test_shuffles.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/mt19937.c
	../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
)


set_source_files_properties(
        COMPILE_FLAGS -msse2)

set_source_files_properties(
        main.c
	test_relationals.cpp
    test_comparisons_float.cpp
    test_comparisons_double.cpp
	test_shuffles.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_relationals
        ${CLConform_LIBRARIES})
