add_executable(conformance_test_device_partition
	main.c
	test_device_partition.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/genericThread.cpp
	../../test_common/harness/mt19937.c
	../../test_common/harness/conversions.c
	../../test_common/harness/typeWrappers.cpp
	../../test_common/harness/msvc9.c
)

set_source_files_properties(
	main.c
	test_device_partition.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/genericThread.cpp
	../../test_common/harness/mt19937.c
	../../test_common/harness/conversions.c
	../../test_common/harness/typeWrappers.cpp
	../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

target_link_libraries(conformance_test_device_partition
        ${CLConform_LIBRARIES})

install(TARGETS conformance_test_device_partition
    DESTINATION "${CLConf_OUT_DIR}")
