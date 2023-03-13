set(MODULE_NAME DEVICE_TIMER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_device_timer.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
timer_resolution_queries
device_and_host_timers
)

include(../CMakeCommon.cmake)
