add_executable(conformance_test_image_streams
	main.cpp
	test_iterations.cpp
	../image_helpers.cpp 
	test_loops.cpp
	test_write_image.cpp
	test_read_1D.cpp
	test_read_3D.cpp
	test_read_1D_array.cpp
	test_read_2D_array.cpp
	test_write_1D.cpp
	test_write_3D.cpp
	test_write_1D_array.cpp
	test_write_2D_array.cpp
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
	../image_helpers.cpp 
	test_loops.cpp
	test_write_image.cpp
	test_read_1D.cpp
	test_read_3D.cpp
	test_read_1D_array.cpp
	test_read_2D_array.cpp
	test_write_1D.cpp
	test_write_3D.cpp
	test_write_1D_array.cpp
	test_write_2D_array.cpp
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

TARGET_LINK_LIBRARIES(conformance_test_image_streams
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_image_streams)
install(TARGETS conformance_test_image_streams
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
