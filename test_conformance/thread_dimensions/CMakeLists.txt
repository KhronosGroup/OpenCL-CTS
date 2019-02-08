add_executable(conformance_test_thread_dimensions
        main.c
	test_thread_dimensions.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/typeWrappers.cpp
        ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
  ../../test_common/harness/crc32.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

set_source_files_properties(
        main.c
	test_thread_dimensions.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_thread_dimensions
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_thread_dimensions)
install(TARGETS conformance_test_thread_dimensions
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
