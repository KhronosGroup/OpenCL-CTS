//
// Copyright (c) 2019 The Khronos Group Inc.
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


#ifndef __MEDIA_SHARING_PROCS_H__
#define __MEDIA_SHARING_PROCS_H__


extern int test_context_create(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_get_device_ids(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_api(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_kernel(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_other_data_types(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_memory_access(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_interop_user_sync(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);


#endif    // #ifndef __MEDIA_SHARING_PROCS_H__ 