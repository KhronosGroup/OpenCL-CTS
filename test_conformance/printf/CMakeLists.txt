set(MODULE_NAME PRINTF)

set(CMAKE_CXX_STANDARD 11)

set(${MODULE_NAME}_SOURCES
         test_printf.c
         util_printf.c
         )

include(../CMakeCommon.txt)
