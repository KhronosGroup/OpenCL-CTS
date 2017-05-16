add_executable(conformance_test_cl_copy_images
        main.cpp 
		test_copy_2D.cpp 
		../image_helpers.cpp  
		test_loops.cpp 
		test_copy_3D.cpp 
		test_copy_2D_3D.cpp 
		  ../../../test_common/harness/testHarness.c 
		  ../../../test_common/harness/errorHelpers.c 
		  ../../../test_common/harness/threadTesting.c 
		  ../../../test_common/harness/kernelHelpers.c 
		  ../../../test_common/harness/imageHelpers.cpp 
		  ../../../test_common/harness/mt19937.c
		  ../../../test_common/harness/conversions.c 
		  ../../../test_common/harness/typeWrappers.cpp
          ../../../test_common/harness/msvc9.c)

set_source_files_properties(
        main.cpp 
		test_copy_2D.cpp 
		../image_helpers.cpp  
		test_loops.cpp 
		test_copy_3D.cpp 
		test_copy_2D_3D.cpp 
		  ../../../test_common/harness/testHarness.c 
		  ../../../test_common/harness/errorHelpers.c 
		  ../../../test_common/harness/threadTesting.c 
		  ../../../test_common/harness/kernelHelpers.c 
		  ../../../test_common/harness/imageHelpers.cpp 
		  ../../../test_common/harness/conversions.c 
		 ../../../test_common/harness/typeWrappers.cpp
         ../../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_cl_copy_images
        ${CLConform_LIBRARIES})
