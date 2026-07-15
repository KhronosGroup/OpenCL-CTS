#include "harness/deviceInfo.h"

int main(int argc, const char* argv[])
{
    return run_extension_stub(argc, argv, { "cl_khr_d3d10_sharing" });
}
