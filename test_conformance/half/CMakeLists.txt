set(MODULE_NAME HALF)

set(${MODULE_NAME}_SOURCES
        cl_utils.cpp
        Test_vLoadHalf.cpp
        Test_roundTrip.cpp
        Test_vStoreHalf.cpp main.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
vload_half
vloada_half
vstore_half
vstorea_half
vstore_half_rte
vstorea_half_rte
vstore_half_rtz
vstorea_half_rtz
vstore_half_rtp
vstorea_half_rtp
vstore_half_rtn
vstorea_half_rtn
roundTrip
)

include(../CMakeCommon.cmake)

