add_executable(conformance_test_samplerless_reads
	main.cpp
	test_iterations.cpp
	test_loops.cpp
	test_read_1D.cpp
	test_read_1D_buffer.cpp
	test_read_1D_array.cpp
	test_read_2D_array.cpp
	test_read_3D.cpp
	../image_helpers.cpp 
	../../../test_common/harness/errorHelpers.c
	../../../test_common/harness/threadTesting.c
	../../../test_common/harness/kernelHelpers.c
	../../../test_common/harness/imageHelpers.cpp
	../../../test_common/harness/conversions.c
	../../../test_common/harness/testHarness.c
	../../../test_common/harness/mt19937.c
	../../../test_common/harness/msvc9.c
	../../../test_common/harness/typeWrappers.cpp
)

set_source_files_properties(
	main.cpp
	test_iterations.cpp
	test_loops.cpp
	test_read_1D.cpp
	test_read_1D_buffer.cpp
	test_read_1D_array.cpp
	test_read_2D_array.cpp
	test_read_3D.cpp
	../image_helpers.cpp 
	../../../test_common/harness/errorHelpers.c
	../../../test_common/harness/threadTesting.c
	../../../test_common/harness/kernelHelpers.c
	../../../test_common/harness/imageHelpers.cpp
	../../../test_common/harness/conversions.c
	../../../test_common/harness/testHarness.c
	../../../test_common/harness/mt19937.c
	../../../test_common/harness/msvc9.c
	../../../test_common/harness/typeWrappers.cpp 
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_samplerless_reads
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_samplerless_reads)
install(TARGETS conformance_test_samplerless_reads
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
