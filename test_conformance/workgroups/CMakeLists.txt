set(MODULE_NAME WORKGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_wg_all.cpp
    test_wg_any.cpp
    test_wg_broadcast.cpp
    test_wg_scan_reduce.cpp
    test_wg_suggested_local_work_size.cpp
    test_advanced_2d.cpp
    test_advanced_3d.cpp
    test_advanced_other.cpp
    test_basic.cpp
    TestNonUniformWorkGroup.cpp
    tools.cpp
)

include(../CMakeCommon.txt)
