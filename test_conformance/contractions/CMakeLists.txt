set(MODULE_NAME CONTRACTIONS)

set(${MODULE_NAME}_SOURCES
        contractions.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
contractions_float_0
contractions_float_1
contractions_float_2
contractions_float_3
contractions_float_4
contractions_float_5
contractions_float_6
contractions_float_7
contractions_double_0
contractions_double_1
contractions_double_2
contractions_double_3
contractions_double_4
contractions_double_5
contractions_double_6
contractions_double_7
)

include(../CMakeCommon.cmake)
