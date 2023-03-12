set(MODULE_NAME WORKGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_wg_all.cpp
    test_wg_any.cpp
    test_wg_broadcast.cpp
    test_wg_scan_reduce.cpp
    test_wg_suggested_local_work_size.cpp
)

include(../CMakeCommon.txt)
