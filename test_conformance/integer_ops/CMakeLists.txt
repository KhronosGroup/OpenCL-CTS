add_executable(conformance_test_integer_ops
	main.c
	test_popcount.c
	test_int_basic_ops.c
	test_integers.cpp
	test_upsample.cpp
	test_intmul24.c test_intmad24.c
	test_sub_sat.c test_add_sat.c
	test_abs.c test_absdiff.c 
	test_unary_ops.cpp
	verification_and_generation_functions.c
	../../test_common/harness/conversions.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/os_helpers.cpp
        ../../test_common/harness/parseParameters.cpp
	../../test_common/harness/mt19937.c
	../../test_common/harness/msvc9.c
	../../test_common/harness/ThreadPool.c
	../../test_common/harness/kernelHelpers.c
  ../../test_common/harness/crc32.c
)

set_source_files_properties(
	main.c
	test_popcount.c
	test_int_basic_ops.c
	test_integers.cpp
	test_upsample.cpp
	test_intmul24.c test_intmad24.c
	test_sub_sat.c test_add_sat.c
	test_abs.c test_absdiff.c 
	test_unary_ops.cpp
	verification_and_generation_functions.c
	../../test_common/harness/conversions.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/mt19937.c
	../../test_common/harness/msvc9.c
	../../test_common/harness/ThreadPool.c
	../../test_common/harness/kernelHelpers.c
        PROPERTIES LANGUAGE CXX)
        
if (NOT CMAKE_CL_64 AND NOT MSVC)                   
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)" AND NOT MSVC)
set_source_files_properties(
	main.c
	test_popcount.c
	test_int_basic_ops.c
	test_integers.cpp
	test_upsample.cpp
	test_intmul24.c test_intmad24.c
	test_sub_sat.c test_add_sat.c
	test_abs.c test_absdiff.c 
	test_unary_ops.cpp
	verification_and_generation_functions.c
	../../test_common/harness/conversions.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/mt19937.c
	../../test_common/harness/ThreadPool.c
	../../test_common/harness/kernelHelpers.c
         COMPILE_FLAGS -march=i686)
endif()
endif (NOT CMAKE_CL_64 AND NOT MSVC)

TARGET_LINK_LIBRARIES(conformance_test_integer_ops
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_integer_ops)
install(TARGETS conformance_test_integer_ops
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
