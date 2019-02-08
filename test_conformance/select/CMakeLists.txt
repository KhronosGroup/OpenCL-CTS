add_executable(conformance_test_select
        test_select.c
        util_select.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/crc32.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

set_source_files_properties(
        test_select.c
        util_select.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/testHarness.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_select
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_select)
install(TARGETS conformance_test_select
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
