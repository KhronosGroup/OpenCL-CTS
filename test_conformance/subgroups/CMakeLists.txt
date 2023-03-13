set(MODULE_NAME SUBGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    subhelpers.cpp
    test_barrier.cpp
    test_queries.cpp
    test_workitem.cpp
    test_subgroup.cpp
    test_ifp.cpp
    test_subgroup_extended_types.cpp
    subgroup_common_kernels.cpp
    test_subgroup_non_uniform_vote.cpp
    test_subgroup_non_uniform_arithmetic.cpp
    test_subgroup_ballot.cpp
    test_subgroup_clustered_reduce.cpp
    test_subgroup_shuffle.cpp
    test_subgroup_shuffle_relative.cpp
    test_subgroup_rotate.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
sub_group_info_ext
sub_group_info_core
work_item_functions_ext
work_item_functions_core
subgroup_functions_ext
subgroup_functions_core
barrier_functions_ext
barrier_functions_core
ifp_ext
ifp_core
subgroup_functions_extended_types
subgroup_functions_non_uniform_vote
subgroup_functions_non_uniform_arithmetic
subgroup_functions_ballot
subgroup_functions_clustered_reduce
subgroup_functions_shuffle
subgroup_functions_shuffle_relative
subgroup_functions_rotate
)

include(../CMakeCommon.cmake)
