//
// Copyright (c) 2020 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef _featureHelpers_h
#define _featureHelpers_h

#include "compat.h"
#include "testHarness.h"

struct OpenCLCFeatures
{
    bool supports__opencl_c_3d_image_writes;
    bool supports__opencl_c_atomic_order_acq_rel;
    bool supports__opencl_c_atomic_order_seq_cst;
    bool supports__opencl_c_atomic_scope_device;
    bool supports__opencl_c_atomic_scope_all_devices;
    bool supports__opencl_c_device_enqueue;
    bool supports__opencl_c_generic_address_space;
    bool supports__opencl_c_fp64;
    bool supports__opencl_c_images;
    bool supports__opencl_c_int64;
    bool supports__opencl_c_pipes;
    bool supports__opencl_c_program_scope_global_variables;
    bool supports__opencl_c_read_write_images;
    bool supports__opencl_c_subgroups;
    bool supports__opencl_c_work_group_collective_functions;
};

int get_device_cl_c_features(cl_device_id device, OpenCLCFeatures& features);

#endif // _featureHelpers_h
