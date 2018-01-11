add_executable(conformance_test_computeinfo
        main.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/msvc9.c
)

set_source_files_properties(
        main.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)
        
TARGET_LINK_LIBRARIES(conformance_test_computeinfo
        ${CLConform_LIBRARIES})

install(TARGETS conformance_test_computeinfo
    DESTINATION "${CLConf_OUT_DIR}")
