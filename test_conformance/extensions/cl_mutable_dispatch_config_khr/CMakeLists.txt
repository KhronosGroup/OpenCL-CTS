set(MODULE_NAME CL_KHR_MUTABLE_DISPATCH)

set(${MODULE_NAME}_SOURCES
    main.cpp
    command_buffer_test_mutable_dispatch.cpp
)

add_cxx_flag_if_supported(-std=c++14)

include(../../CMakeCommon.txt)
