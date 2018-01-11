add_executable(conformance_test_vecstep
	globals.c
        test_step.c
        main.c
        structs.c
        type_replacer.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

set_source_files_properties(
	globals.c
        test_step.c
        main.c
        structs.c
        type_replacer.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_vecstep
        ${CLConform_LIBRARIES})

install(TARGETS conformance_test_vecstep
    DESTINATION "${CLConf_OUT_DIR}")
