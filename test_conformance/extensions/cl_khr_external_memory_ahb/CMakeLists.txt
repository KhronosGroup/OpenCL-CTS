set(MODULE_NAME CL_KHR_EXTERNAL_MEMORY_AHB)

if(DEFINED ANDROID_PLATFORM AND ANDROID_PLATFORM GREATER 28)
set(${MODULE_NAME}_SOURCES
    main.cpp
    test_ahb.cpp
    test_ahb_negative.cpp
    debug_ahb.cpp
)

link_libraries(OpenCL nativewindow)

else()
build_as_stub(${MODULE_NAME})
endif()

include(../../CMakeCommon.txt)
