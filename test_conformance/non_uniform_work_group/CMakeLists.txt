set(MODULE_NAME NON_UNIFORM_WORK_GROUP)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_advanced_2d.cpp
    test_advanced_3d.cpp
    test_advanced_other.cpp
    test_basic.cpp
    TestNonUniformWorkGroup.cpp
    tools.cpp
)

set_gnulike_module_compile_flags("-Wno-unused-but-set-variable")

include(../CMakeCommon.txt)

# end of file #
