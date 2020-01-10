add_executable(test_half
        cl_utils.c Test_vLoadHalf.c Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/parseParameters.cpp
)

set_source_files_properties(
          cl_utils.c Test_vLoadHalf.c Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/mingw_compat.c  
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)
TARGET_LINK_LIBRARIES(test_half
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_half)
install(TARGETS test_half
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
