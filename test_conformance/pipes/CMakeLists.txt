set(MODULE_NAME pipes)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_pipe_read_write.cpp
    test_pipe_info.cpp
    test_pipe_limits.cpp
    test_pipe_query_functions.cpp
    test_pipe_readwrite_errors.cpp
    test_pipe_subgroups.cpp
)

include(../CMakeCommon.txt)
