set(MODULE_NAME ALLOCATIONS)

set(${MODULE_NAME}_SOURCES
        main.cpp
        allocation_execute.cpp
        allocation_fill.cpp
        allocation_functions.cpp
        allocation_utils.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../CMakeCommon.txt)
