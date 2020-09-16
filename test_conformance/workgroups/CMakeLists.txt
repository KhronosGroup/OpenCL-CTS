set(MODULE_NAME WORKGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_wg_all.cpp
    test_wg_any.cpp
    test_wg_broadcast.cpp
    test_wg_reduce.cpp
    test_wg_reduce_max.cpp
    test_wg_reduce_min.cpp
    test_wg_scan_exclusive_add.cpp
    test_wg_scan_exclusive_min.cpp
    test_wg_scan_exclusive_max.cpp
    test_wg_scan_inclusive_add.cpp
    test_wg_scan_inclusive_min.cpp
    test_wg_scan_inclusive_max.cpp
)

include(../CMakeCommon.txt)
