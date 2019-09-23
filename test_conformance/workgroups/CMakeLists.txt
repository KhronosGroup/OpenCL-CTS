set(MODULE_NAME WORKGROUPS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_wg_all.c
    test_wg_any.c
    test_wg_broadcast.c
    test_wg_reduce.c
    test_wg_reduce_max.c
    test_wg_reduce_min.c
    test_wg_scan_exclusive_add.c
    test_wg_scan_exclusive_min.c
    test_wg_scan_exclusive_max.c
    test_wg_scan_inclusive_add.c
    test_wg_scan_inclusive_min.c
    test_wg_scan_inclusive_max.c
)

include(../CMakeCommon.txt)
