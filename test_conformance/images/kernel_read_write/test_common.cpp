
#include "test_common.h"

cl_sampler create_sampler(cl_context context, image_sampler_data *sdata, bool test_mipmaps, cl_int *error) {
    cl_sampler sampler = nullptr;
    if (test_mipmaps) {
        cl_sampler_properties properties[] = {
            CL_SAMPLER_NORMALIZED_COORDS, sdata->normalized_coords,
            CL_SAMPLER_ADDRESSING_MODE, sdata->addressing_mode,
            CL_SAMPLER_FILTER_MODE, sdata->filter_mode,
            CL_SAMPLER_MIP_FILTER_MODE, sdata->filter_mode,
            0};
        sampler = clCreateSamplerWithProperties(context, properties, error);
    } else {
        sampler = clCreateSampler(context, sdata->normalized_coords, sdata->addressing_mode, sdata->filter_mode, error);
    }
    return sampler;
}

