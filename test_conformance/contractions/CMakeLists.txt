add_executable(test_contractions
        contractions.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/kernelHelpers.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        contractions.c
        COMPILE_FLAGS -msse)
endif()

if(WIN32)
set_source_files_properties(
        contractions.c
        ../../test_common/harness/mt19937.c        
        ../../test_common/harness/msvc9.c        
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/kernelHelpers.c
        PROPERTIES LANGUAGE CXX)
endif(WIN32)

TARGET_LINK_LIBRARIES(test_contractions
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_contractions)
install(TARGETS test_contractions
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
