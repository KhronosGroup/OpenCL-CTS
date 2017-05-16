set(MODULE_NAME EVENTS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_events.cpp
    test_event_dependencies.cpp
    test_waitlists.cpp
    test_userevents.cpp
    test_userevents_multithreaded.cpp
    action_classes.cpp
    test_callbacks.cpp
    ../../test_common/harness/genericThread.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/ThreadPool.c
)

include(../CMakeCommon.txt)
