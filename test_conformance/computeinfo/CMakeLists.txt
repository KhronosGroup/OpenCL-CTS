set(MODULE_NAME computeinfo)

set(${MODULE_NAME}_SOURCES
        main.cpp
        device_uuid.cpp
        extended_versioning.cpp
        conforming_version.cpp
        pci_bus_info.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
computeinfo
extended_versioning
device_uuid
conformance_version
pci_bus_info
)

include(../CMakeCommon.cmake)
