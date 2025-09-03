#include <array>

static const char write_kernel_source[] = R"(
	kernel void write_kernel(global unsigned int *p) {
		*p = 42;
	})";
