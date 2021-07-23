set(MODULE_NAME SUBGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_barrier.cpp
    test_queries.cpp
    test_workitem.cpp
    test_subgroup.cpp
    test_ifp.cpp
    subgroup_common_kernels.cpp
)
if(USE_CL_EXPERIMENTAL)
    list(APPEND ${MODULE_NAME}_SOURCES
         test_subgroup_extended_types.cpp
         test_subgroup_non_uniform_vote.cpp
         test_subgroup_non_uniform_arithmetic.cpp
         test_subgroup_ballot.cpp
         test_subgroup_clustered_reduce.cpp
         test_subgroup_shuffle.cpp
         test_subgroup_shuffle_relative.cpp
    )
endif(USE_CL_EXPERIMENTAL)

include(../CMakeCommon.txt)
