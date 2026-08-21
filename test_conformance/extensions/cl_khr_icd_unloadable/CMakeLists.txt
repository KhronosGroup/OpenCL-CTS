# Test Executable:

set(MODULE_NAME cl_khr_icd_unloadable)
set(${MODULE_NAME}_OUT ${CONFORMANCE_PREFIX}${MODULE_NAME}${CONFORMANCE_SUFFIX})

add_executable(${${MODULE_NAME}_OUT} main.cpp)
set_property(TARGET ${${MODULE_NAME}_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")
target_link_libraries(${${MODULE_NAME}_OUT} ${CMAKE_DL_LIBS})

# Test Plugin (dynamically loaded and unloaded):

set(PLUGIN_NAME LoadUnloadPlugin)
set(${PLUGIN_NAME}_OUT ${PLUGIN_NAME}${CONFORMANCE_SUFFIX})

add_library(${${PLUGIN_NAME}_OUT} MODULE load_unload_plugin.cpp)
set_property(TARGET ${${PLUGIN_NAME}_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")
target_link_libraries(${${PLUGIN_NAME}_OUT} ${HARNESS_LIB} ${CLConform_LIBRARIES})

# Dependency
add_dependencies(${${MODULE_NAME}_OUT} ${${PLUGIN_NAME}_OUT})

# Install Targets:

include(GNUInstallDirs)

install(TARGETS ${${MODULE_NAME}_OUT}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
install(TARGETS ${${PLUGIN_NAME}_OUT}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
