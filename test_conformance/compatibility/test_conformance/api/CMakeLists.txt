set(MODULE_NAME COMPATIBILITY_API)

set(${MODULE_NAME}_SOURCES
         main.c
         test_api_min_max.c
         test_mem_object_info.cpp
         test_kernel_arg_info.c
         test_queue_properties.cpp
)

set(${MODULE_NAME}_LIBS harness-compat)

include(../../../CMakeCommon.txt)
