set(MODULE_NAME CL_KHR_MUTABLE_DISPATCH)

set(${MODULE_NAME}_SOURCES
    main.cpp
    mutable_command_info.cpp
    mutable_command_image_arguments.cpp
    mutable_command_arguments.cpp
    mutable_command_simultaneous.cpp
    mutable_command_global_size.cpp
    mutable_command_local_size.cpp
    mutable_command_global_offset.cpp
    mutable_command_full_dispatch.cpp
    mutable_command_overwrite_update.cpp
    mutable_command_multiple_dispatches.cpp
    mutable_command_iterative_arg_update.cpp
    mutable_command_work_groups.cpp
    ../basic_command_buffer.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../../../CMakeCommon.txt)
