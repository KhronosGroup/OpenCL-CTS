set(MODULE_NAME DEVICE_PARTITION)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_device_partition.cpp
)
set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
partition_equally
partition_by_counts
partition_by_affinity_domain_numa
partition_by_affinity_domain_l4_cache
partition_by_affinity_domain_l3_cache
partition_by_affinity_domain_l2_cache
partition_by_affinity_domain_l1_cache
partition_by_affinity_domain_next_partitionable
partition_all
)

include(../CMakeCommon.cmake)
