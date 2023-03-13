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

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
non_uniform_1d_basic
non_uniform_1d_atomics
non_uniform_1d_barriers
non_uniform_2d_basic
non_uniform_2d_atomics
non_uniform_2d_barriers
non_uniform_3d_basic
non_uniform_3d_atomics
non_uniform_3d_barriers
non_uniform_other_basic
non_uniform_other_atomics
non_uniform_other_barriers
)

include(../CMakeCommon.cmake)

# end of file #
