set(MODULE_NAME GEOMETRICS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_geometrics_double.cpp
    test_geometrics.cpp
)

# VS2005 optimization WAR for geom_length)
if(MSVC)
set(CMAKE_C_FLAGS_RELEASE "/Od /Ob0")
set(CMAKE_CXX_FLAGS_RELEASE "/Od /Ob0")
endif(MSVC)

include(../CMakeCommon.txt)

