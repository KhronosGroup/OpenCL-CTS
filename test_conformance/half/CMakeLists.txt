set(MODULE_NAME HALF)

set(${MODULE_NAME}_SOURCES
        cl_utils.c
        Test_vLoadHalf.c
        Test_roundTrip.c
        Test_vStoreHalf.c main.c
)

include(../CMakeCommon.txt)

