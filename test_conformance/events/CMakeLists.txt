add_executable(conformance_test_events
        main.c
	test_events.cpp
	test_event_dependencies.cpp
	test_waitlists.cpp
    test_userevents.cpp
    test_callbacks.cpp
    test_userevents_multithreaded.cpp
    action_classes.cpp
	../../test_common/harness/genericThread.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/typeWrappers.cpp
	../../test_common/harness/mt19937.c
	../../test_common/harness/conversions.c
   	../../test_common/harness/msvc9.c
)

set_source_files_properties(
        main.c
    test_userevents.cpp
    test_callbacks.cpp
    test_userevents_multithreaded.cpp
    action_classes.cpp
	../../test_common/harness/genericThread.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/conversions.c
    	../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_events
        ${CLConform_LIBRARIES})
