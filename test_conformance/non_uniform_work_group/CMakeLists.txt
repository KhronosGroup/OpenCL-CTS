set(MODULE_NAME NON_UNIFORM_WORK_GROUP)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    add_cxx_flag_if_supported(-Wno-narrowing)
endif()

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_advanced_2d.cpp
    test_advanced_3d.cpp
    test_advanced_other.cpp
    test_basic.cpp
    TestNonUniformWorkGroup.cpp
    tools.cpp
)

include(../CMakeCommon.txt)

# end of file #
