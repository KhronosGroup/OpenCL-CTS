add_executable(test_bruteforce
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c 	binary_two_results_i.c 	i_unary.c
    macro_binary.c 	macro_unary.c 	mad.c
    main.c 	reference_math.c 
    ternary.c 	unary.c 	unary_two_results.c
    unary_two_results_i.c unary_u.c
    
    ../../test_common/harness/rounding_mode.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c 	binary_two_results_i.c 	i_unary.c
    macro_binary.c 	macro_unary.c 	mad.c
    main.c 	reference_math.c 
    ternary.c 	unary.c 	unary_two_results.c
    unary_two_results_i.c unary_u.c
    COMPILE_FLAGS -msse2)
endif()

if (MSVC)
set_source_files_properties(
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c 	binary_two_results_i.c 	i_unary.c
    macro_binary.c 	macro_unary.c 	mad.c
    main.c 	reference_math.c
    ternary.c 	unary.c 	unary_two_results.c
    unary_two_results_i.c unary_u.c
    ../../test_common/harness/rounding_mode.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/mt19937.c
	PROPERTIES LANGUAGE CXX)
endif(MSVC)

if (NOT CMAKE_CL_64 AND NOT MSVC)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86$)|(X86$)" AND NOT MSVC)
set_source_files_properties(
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c 	binary_two_results_i.c 	i_unary.c
    macro_binary.c 	macro_unary.c 	mad.c
    main.c 	reference_math.c
    ternary.c 	unary.c 	unary_two_results.c
    unary_two_results_i.c unary_u.c
    ../../test_common/harness/rounding_mode.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/msvc9.c
	COMPILE_FLAGS -march=i686)
endif()
endif(NOT CMAKE_CL_64 AND NOT MSVC)

TARGET_LINK_LIBRARIES(test_bruteforce
    ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_bruteforce)
install(TARGETS test_bruteforce
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
