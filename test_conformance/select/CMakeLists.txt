add_executable(conformance_test_select
        test_select.c
        util_select.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
)


set_source_files_properties(
        COMPILE_FLAGS -msse2)

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
