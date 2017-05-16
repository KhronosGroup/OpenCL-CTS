if(NOT MSVC)
# IGL: Force no optimizations
set(CMAKE_C_FLAGS_RELEASE "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
endif(NOT MSVC)

add_subdirectory(clCopyImage)
add_subdirectory(clFillImage)
add_subdirectory(clGetInfo)
add_subdirectory(clReadWriteImage)
add_subdirectory(kernel_image_methods)
add_subdirectory(kernel_read_write)
add_subdirectory(samplerlessReads)


