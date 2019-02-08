add_executable(conformance_test_kernel_image_methods
	main.cpp
	test_1D.cpp
	test_1D_array.cpp
	test_2D.cpp
	test_2D_array.cpp
	../image_helpers.cpp 
	test_loops.cpp
	test_3D.cpp
	../../../test_common/harness/errorHelpers.c
	../../../test_common/harness/threadTesting.c
	../../../test_common/harness/kernelHelpers.c
	../../../test_common/harness/imageHelpers.cpp
	../../../test_common/harness/conversions.c
	../../../test_common/harness/testHarness.c
        ../../../test_common/harness/parseParameters.cpp
	../../../test_common/harness/mt19937.c
	../../../test_common/harness/msvc9.c
	../../../test_common/harness/typeWrappers.cpp
  ../../../test_common/harness/crc32.c
)

set_source_files_properties(
	main.cpp
	test_1D.cpp
	test_1D_array.cpp
	test_2D.cpp
	test_2D_array.cpp
	../image_helpers.cpp 
	test_loops.cpp
	test_3D.cpp
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

TARGET_LINK_LIBRARIES(conformance_test_kernel_image_methods
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_kernel_image_methods)
install(TARGETS conformance_test_kernel_image_methods
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
