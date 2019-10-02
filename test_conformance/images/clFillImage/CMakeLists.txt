add_executable(test_cl_fill_images
	main.cpp
	test_fill_generic.cpp
	test_fill_1D.cpp
	test_fill_2D.cpp
	test_fill_3D.cpp
	test_fill_1D_array.cpp
	test_fill_2D_array.cpp
	test_loops.cpp
	../image_helpers.cpp 
	../../../test_common/harness/errorHelpers.c
	../../../test_common/harness/threadTesting.c
	../../../test_common/harness/kernelHelpers.c
	../../../test_common/harness/imageHelpers.cpp
	../../../test_common/harness/conversions.c
	../../../test_common/harness/testHarness.c
	../../../test_common/harness/typeWrappers.cpp
	../../../test_common/harness/mt19937.c
	../../../test_common/harness/msvc9.c)

set_source_files_properties(
	main.cpp
	test_fill_generic.cpp
	test_fill_1D.cpp
	test_fill_2D.cpp
	test_fill_3D.cpp
	test_fill_1D_array.cpp
	test_fill_2D_array.cpp
	test_loops.cpp
	../image_helpers.cpp 
	../../../test_common/harness/errorHelpers.c
	../../../test_common/harness/threadTesting.c
	../../../test_common/harness/kernelHelpers.c
	../../../test_common/harness/imageHelpers.cpp
	../../../test_common/harness/conversions.c
	../../../test_common/harness/testHarness.c
	../../../test_common/harness/typeWrappers.cpp
	../../../test_common/harness/mt19937.c
	../../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(test_cl_fill_images
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_cl_fill_images)
install(TARGETS test_cl_fill_images
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
