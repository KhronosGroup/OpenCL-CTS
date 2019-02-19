add_compile_options(-std=c++11)

add_executable(conformance_test_printf
	test_printf.c
	util_printf.c
	../../test_common/harness/mt19937.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/msvc9.c
  ../../test_common/harness/crc32.c
        ../../test_common/harness/parseParameters.cpp
	../../test_common/harness/os_helpers.cpp
)

set_source_files_properties(
	test_printf.c
	util_printf.c
	../../test_common/harness/mt19937.c
	../../test_common/harness/kernelHelpers.c
	../../test_common/harness/errorHelpers.c
	../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

target_link_libraries(conformance_test_printf
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_printf)
install(TARGETS conformance_test_printf
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
