set(MODULE_NAME pipes)

set(${MODULE_NAME}_SOURCES
    main.c
    test_pipe_read_write.c
    test_pipe_info.c
    test_pipe_limits.c
    test_pipe_query_functions.c
    test_pipe_readwrite_errors.c
    test_pipe_subgroups.c
)

include(../CMakeCommon.txt)
