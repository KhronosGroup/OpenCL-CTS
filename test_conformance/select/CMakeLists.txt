add_executable(conformance_test_select
        test_select.c
        util_select.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

if(WIN32)
set_source_files_properties(
        test_select.c
        util_select.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        PROPERTIES LANGUAGE CXX)
endif(WIN32)

TARGET_LINK_LIBRARIES(conformance_test_select
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_select)
install(TARGETS conformance_test_select
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
