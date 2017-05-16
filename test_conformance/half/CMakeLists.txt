add_executable(conformance_Test_half
        cl_utils.c Test_vLoadHalf.c Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
)

if(MSVC)
set_source_files_properties(
          cl_utils.c Test_vLoadHalf.c Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/mingw_compat.c  
        ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)
endif(MSVC)
TARGET_LINK_LIBRARIES(conformance_Test_half
        ${CLConform_LIBRARIES})
