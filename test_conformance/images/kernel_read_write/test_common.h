
#include "../testBase.h"

#define ABS_ERROR(result, expected) (fabs(expected - result))

extern cl_sampler create_sampler(cl_context context, image_sampler_data *sdata, bool test_mipmaps, cl_int *error);

