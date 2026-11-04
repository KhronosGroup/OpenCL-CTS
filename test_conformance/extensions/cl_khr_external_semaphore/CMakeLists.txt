set(MODULE_NAME CL_KHR_EXTERNAL_SEMAPHORE)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_external_semaphore.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

if(VULKAN_IS_SUPPORTED)
  set (CLConform_VULKAN_LIBRARIES_DIR "${VULKAN_LIB_DIR}")

  link_directories(${CLConform_VULKAN_LIBRARIES_DIR})

  list(APPEND CLConform_INCLUDE_DIR ${VULKAN_INCLUDE_DIR})

  include_directories (${CLConform_INCLUDE_DIR})

  list(APPEND CLConform_LIBRARIES vulkan_wrapper)

  include_directories("../../common/vulkan_wrapper")

  add_compile_definitions(VULKAN_IS_SUPPORTED)
endif()

include(../../CMakeCommon.txt)
