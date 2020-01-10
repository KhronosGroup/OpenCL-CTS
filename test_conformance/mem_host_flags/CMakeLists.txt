add_executable(test_mem_host_flags
	main.cpp
	mem_host_buffer.cpp
	mem_host_image.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/genericThread.cpp
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/typeWrappers.cpp
	../../test_common/harness/mt19937.c
	../../test_common/harness/msvc9.c
	../../test_common/harness/conversions.c
	../../test_common/harness/ThreadPool.c
)

set_source_files_properties(
	main.cpp
	mem_host_buffer.cpp
	mem_host_image.cpp
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/threadTesting.c
	../../test_common/harness/testHarness.c
	../../test_common/harness/genericThread.cpp
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/typeWrappers.cpp
	../../test_common/harness/mt19937.c
	../../test_common/harness/msvc9.c
	../../test_common/harness/conversions.c
	../../test_common/harness/ThreadPool.c
        PROPERTIES LANGUAGE CXX)

target_link_libraries(test_mem_host_flags
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_mem_host_flags)
install(TARGETS test_mem_host_flags
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
