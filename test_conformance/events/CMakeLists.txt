set(MODULE_NAME EVENTS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_events.cpp
    test_event_dependencies.cpp
    test_waitlists.cpp
    test_userevents.cpp
    test_userevents_multithreaded.cpp
    action_classes.cpp
    test_callbacks.cpp
)

include(../CMakeCommon.txt)
