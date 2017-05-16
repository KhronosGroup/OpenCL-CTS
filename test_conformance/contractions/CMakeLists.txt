add_executable(conformance_contractions
        contractions.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/kernelHelpers.c
)
set_source_files_properties(
        contractions.c
        COMPILE_FLAGS -msse)
if(WIN32)
set_source_files_properties(
        contractions.c    
        ../../test_common/harness/msvc9.c        
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/kernelHelpers.c
        PROPERTIES LANGUAGE CXX)
endif(WIN32)

TARGET_LINK_LIBRARIES(conformance_contractions
        ${CLConform_LIBRARIES})
