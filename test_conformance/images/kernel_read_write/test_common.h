
#include "../testBase.h"

#define ABS_ERROR(result, expected) (fabs(expected - result))
#define CLAMP(_val, _min, _max)                                                \
    ((_val) < (_min) ? (_min) : (_val) > (_max) ? (_max) : (_val))

#define MAX_ERR 0.005f
#define MAX_TRIES 1
#define MAX_CLAMPED 1

extern cl_sampler create_sampler(cl_context context, image_sampler_data *sdata, bool test_mipmaps, cl_int *error);

extern bool gExtraValidateInfo;
extern bool gDisableOffsets;
extern bool gUseKernelSamplers;
