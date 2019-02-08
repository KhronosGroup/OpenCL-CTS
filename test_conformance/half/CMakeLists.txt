add_executable(conformance_test_half
        cl_utils.c Test_vLoadHalf.c Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/crc32.c
)

set_source_files_properties(
          cl_utils.c Test_vLoadHalf.c Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/mingw_compat.c  
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)
TARGET_LINK_LIBRARIES(conformance_test_half
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_half)
install(TARGETS conformance_test_half
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
