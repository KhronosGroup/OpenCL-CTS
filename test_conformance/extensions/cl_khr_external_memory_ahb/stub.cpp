#include "harness/deviceInfo.h"

int main(int argc, const char* argv[])
{
    return run_extension_stub(
        argc, argv, { "cl_khr_external_memory_android_hardware_buffer" });
}
