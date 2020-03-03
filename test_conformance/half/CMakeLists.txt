set(MODULE_NAME HALF)

set(${MODULE_NAME}_SOURCES
        cl_utils.cpp
        Test_vLoadHalf.cpp
        Test_roundTrip.cpp
        Test_vStoreHalf.cpp main.cpp
)

include(../CMakeCommon.txt)

