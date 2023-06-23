set(MODULE_NAME DEVICE_TIMER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_device_timer.cpp
)

set_gnulike_module_compile_flags("-Wno-unused-but-set-variable")

include(../CMakeCommon.txt)
