#include "harness/deviceInfo.h"

int main(int argc, const char* argv[])
{
    return run_extension_stub(argc, argv,
                              { "cl_khr_gl_event", "cl_khr_gl_msaa_sharing",
                                "cl_khr_gl_depth_images",
                                "cl_khr_gl_msaa_sharing" });
}
