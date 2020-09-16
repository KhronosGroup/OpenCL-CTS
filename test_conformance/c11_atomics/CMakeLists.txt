set(MODULE_NAME C11_ATOMICS)

set(${MODULE_NAME}_SOURCES
    common.cpp
    host_atomics.cpp
    main.cpp
    test_atomics.cpp
)

include(../CMakeCommon.txt)
