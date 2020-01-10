add_executable(test_computeinfo
        main.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/msvc9.c
)

set_source_files_properties(
        main.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)
        
TARGET_LINK_LIBRARIES(test_computeinfo
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_computeinfo)
install(TARGETS test_computeinfo
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
