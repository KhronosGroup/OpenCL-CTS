add_executable(conformance_test_vecalign
	globals.c
        main.c
        structs.c
        test_vec_align.c
        type_replacer.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/conversions.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

set_source_files_properties(
	globals.c
        main.c
        structs.c
        test_vec_align.c
        type_replacer.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/conversions.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_vecalign
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_vecalign)
install(TARGETS conformance_test_vecalign
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
