set(MODULE_NAME THREAD_DIMENSIONS)

set(${MODULE_NAME}_SOURCES
        main.cpp
    test_thread_dimensions.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
quick_1d_explicit_local
quick_2d_explicit_local
quick_3d_explicit_local
quick_1d_implicit_local
quick_2d_implicit_local
quick_3d_implicit_local
full_1d_explicit_local
full_2d_explicit_local
full_3d_explicit_local
full_1d_implicit_local
full_2d_implicit_local
full_3d_implicit_local
)

include(../CMakeCommon.cmake)
