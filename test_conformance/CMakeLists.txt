cmake_minimum_required(VERSION 3.0)

if(MSVC)
  # The CTS will not link on MSVC 32-bit builds. To workaround this, add the
  # /LARGEADDRESSAWARE option as a linker flag. This is automatically set for
  # MSVC 64-bit so it does no harm to enable it on all MSVC builds.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
else()
  # Allowing the compiler to contract floating-point math causes wrong results
  # to be output from the reference math of bruteforce tests, and for the
  # host rounding code in conversions and image_streams tests.
  set(CTS_EXTRA_FLAGS "-ffp-contract=off")

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    # We set '-mfpmath=sse' because it is the default on the x86-64 compiler,
    # and we want consistent results for x86 code too.
    # We set '-msse2' as most (but not all) of the sub CMakeLists.txt we include
    # set it, so setting it here once means we have consistent compiliation
    # flags across the entire conformance_test_* executables.
    set(CTS_EXTRA_FLAGS "${CTS_EXTRA_FLAGS} -mfpmath=sse -msse2")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(ARM)|(ARM64)")
    # The OpenCL CTS assumes that the char type is signed, which is not the
    # default on ARM compilers, so we need to set it.
    set(CTS_EXTRA_FLAGS "${CTS_EXTRA_FLAGS} -fsigned-char")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CTS_EXTRA_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CTS_EXTRA_FLAGS}")
endif()

include_directories(BEFORE SYSTEM ${CLConform_HEADERS})
link_directories(${CLConform_LIB_DIR})

add_definitions(-DCL_TARGET_OPENCL_VERSION=120)

add_custom_target(OpenCLCTS)

add_subdirectory(allocations)
add_subdirectory(api)
add_subdirectory(atomics)
add_subdirectory(basic)
add_subdirectory(buffers)
add_subdirectory(commonfns)
add_subdirectory(compiler)
add_subdirectory(computeinfo)
add_subdirectory(contractions)
add_subdirectory(conversions)
add_subdirectory(device_partition)
add_subdirectory(events)
add_subdirectory(geometrics)
if(GL_IS_SUPPORTED)
	add_subdirectory(gl)
endif(GL_IS_SUPPORTED)
add_subdirectory(half)
add_subdirectory(headers)
add_subdirectory(images)
add_subdirectory(integer_ops)
add_subdirectory(math_brute_force)
add_subdirectory(mem_host_flags)
add_subdirectory(multiple_device_context)
add_subdirectory(printf)
add_subdirectory(profiling)
add_subdirectory(relationals)
add_subdirectory(select)
add_subdirectory(thread_dimensions)
add_subdirectory(vec_align)
add_subdirectory(vec_step)
