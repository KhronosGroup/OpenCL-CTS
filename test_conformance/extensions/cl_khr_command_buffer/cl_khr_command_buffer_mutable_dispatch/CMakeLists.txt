set(MODULE_NAME CL_KHR_MUTABLE_DISPATCH)

set(${MODULE_NAME}_SOURCES
    main.cpp
    mutable_command_info.cpp
    mutable_command_image_arguments.cpp
    mutable_command_arguments.cpp
    mutable_command_out_of_order.cpp
    mutable_command_global_size.cpp
    mutable_command_local_size.cpp
    mutable_command_global_offset.cpp
    ../basic_command_buffer.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../../../CMakeCommon.txt)
