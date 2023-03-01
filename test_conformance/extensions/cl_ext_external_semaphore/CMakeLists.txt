set(MODULE_NAME CL_KHR_EXTERNAL_SEMAPHORE)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_external_semaphore.cpp
)

# needed by Vulkan wrapper to link
if(WIN32)
    list(APPEND CLConform_LIBRARIES vulkan-1 vulkan_wrapper)
else(WIN32)
    list(APPEND CLConform_LIBRARIES vulkan dl vulkan_wrapper)
endif(WIN32)
set(CMAKE_CXX_FLAGS "-fpermissive")
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
endif(WIN32)

include_directories("../../common/vulkan_wrapper")

include(../../CMakeCommon.txt)
