set(MODULE_NAME CL_KHR_EXTERNAL_MEMORY_AHB)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_ahb.cpp
    test_ahb_negative.cpp
    debug_ahb.cpp
)

if(DEFINED ANDROID_PLATFORM AND ANDROID_PLATFORM GREATER 28)
  link_libraries(OpenCL nativewindow)
else()
  build_with_stub(${MODULE_NAME})
endif()

include(../../CMakeCommon.txt)
