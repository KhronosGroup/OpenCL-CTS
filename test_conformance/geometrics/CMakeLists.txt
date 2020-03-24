set(MODULE_NAME GEOMETRICS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_geometrics_double.cpp
    test_geometrics.cpp
)

# VS2005 optimization WAR for geom_length)
if(MSVC)
string(REGEX REPLACE "/O. /Ob." "/Od /Ob0" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
string(REGEX REPLACE "/O. /Ob." "/Od /Ob0" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
endif(MSVC)

include(../CMakeCommon.txt)

