set(MODULE_NAME WORKGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_wg_all.cpp
    test_wg_any.cpp
    test_wg_broadcast.cpp
    test_wg_scan_reduce.cpp
    test_wg_suggested_local_work_size.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
work_group_all
work_group_any
work_group_reduce_add
work_group_reduce_min
work_group_reduce_max
work_group_scan_inclusive_add
work_group_scan_inclusive_min
work_group_scan_inclusive_max
work_group_scan_exclusive_add
work_group_scan_exclusive_min
work_group_scan_exclusive_max
work_group_broadcast_1D
work_group_broadcast_2D
work_group_broadcast_3D
work_group_suggested_local_size_1D
work_group_suggested_local_size_2D
work_group_suggested_local_size_3D
)

include(../CMakeCommon.cmake)
