set(MODULE_NAME SUBGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_barrier.cpp
    test_queries.cpp
    test_workitem.cpp
    test_workgroup.cpp
    test_ifp.cpp
    test_workgroup_extended_types.cpp
    workgroup_common_kernels.cpp
    test_workgroup_non_uniform_vote.cpp
    test_workgroup_non_uniform_arithmetic.cpp
    test_workgroup_ballot.cpp
)

include(../CMakeCommon.txt)
