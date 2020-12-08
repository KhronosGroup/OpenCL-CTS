set(MODULE_NAME computeinfo)

set(${MODULE_NAME}_SOURCES
        main.cpp
        device_uuid.cpp
        extended_versioning.cpp
        conforming_version.cpp
        pci_bus_info.cpp
)

include(../CMakeCommon.txt)
