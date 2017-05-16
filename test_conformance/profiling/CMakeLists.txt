add_executable(conformance_test_profiling
        main.c readArray.c writeArray.c readImage.c writeImage.c copy.c
        execute.c execute_multipass.c
        ../../test_common/harness/testHarness.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/typeWrappers.cpp
	../../test_common/harness/imageHelpers.cpp
	../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
)


set_source_files_properties(
        COMPILE_FLAGS -msse2)

set_source_files_properties(
        main.c readArray.c writeArray.c readImage.c writeImage.c copy.c
        execute.c execute_multipass.c
        ../../test_common/harness/testHarness.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_profiling
        ${CLConform_LIBRARIES})
