set(MODULE_NAME CL_KHR_EXTERNAL_MEMORY_AHB)

set(${MODULE_NAME}_SOURCES
         main.cpp
        test_ahb.cpp
        debug_ahb.cpp
)

link_libraries(android OpenCL)

include(../../CMakeCommon.txt)
