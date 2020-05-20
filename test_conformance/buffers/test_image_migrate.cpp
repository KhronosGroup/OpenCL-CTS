//
// Copyright (c) 2017 The Khronos Group Inc.
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
#include <stdio.h>
#include <stdlib.h>

#include "procs.h"
#include "harness/errorHelpers.h"

#define MAX_SUB_DEVICES        16        // Limit the sub-devices to ensure no out of resource errors.
#define MEM_OBJ_SIZE          1024
#define IMAGE_DIM         16

// Kernel source code
static const char *image_migrate_kernel_code =
"__kernel void test_image_migrate(write_only image2d_t dst, read_only image2d_t src1,\n"
"                                 read_only image2d_t src2, sampler_t sampler, uint x)\n"
"{\n"
"  int tidX = get_global_id(0), tidY = get_global_id(1);\n"
"  int2 coords = (int2) {tidX, tidY};\n"
"  uint4 val = read_imageui(src1, sampler, coords) ^\n"
"              read_imageui(src2, sampler, coords) ^\n"
"              x;\n"
"  write_imageui(dst, coords, val);\n"
"}\n";

enum migrations { MIGRATE_PREFERRED,           // migrate to the preferred sub-device
                  MIGRATE_NON_PREFERRED,     // migrate to a randomly chosen non-preferred sub-device
                  MIGRATE_RANDOM,              // migrate to a randomly chosen sub-device with randomly chosen flags
                  NUMBER_OF_MIGRATIONS };

static cl_mem init_image(cl_command_queue cmd_q, cl_mem image, cl_uint *data)
{
  cl_int err;

  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {IMAGE_DIM, IMAGE_DIM, 1};

  if (image) {
    if ((err = clEnqueueWriteImage(cmd_q, image, CL_TRUE,
                                   origin, region, 0, 0, data, 0, NULL, NULL)) != CL_SUCCESS) {
      print_error(err, "Failed on enqueue write of image data.");
    }
  }

  return image;
}

static cl_int migrateMemObject(enum migrations migrate, cl_command_queue *queues, cl_mem *mem_objects,
                               cl_uint num_devices, cl_mem_migration_flags *flags, MTdata d)
{
  cl_uint i, j;
  cl_int  err = CL_SUCCESS;

  for (i=0; i<num_devices; i++) {
    j = genrand_int32(d) % num_devices;
    flags[i] = 0;
    switch (migrate) {
      case MIGRATE_PREFERRED:
        // Force the device to be preferred
        j = i;
        break;
      case MIGRATE_NON_PREFERRED:
        // Coerce the device to be non-preferred
        if ((j == i) && (num_devices > 1)) j = (j+1) % num_devices;
        break;
      case MIGRATE_RANDOM:
        // Choose a random set of flags
        flags[i] = (cl_mem_migration_flags)(genrand_int32(d) & (CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));
        break;
      default: log_error("Unhandled migration type: %d\n", migrate); return -1;
    }
    if ((err = clEnqueueMigrateMemObjects(queues[j], 1, (const cl_mem *)(&mem_objects[i]),
                                          flags[i], 0, NULL, NULL)) != CL_SUCCESS) {
      print_error(err, "Failed migrating memory object.");
    }
  }
  return err;
}

static cl_int restoreImage(cl_command_queue *queues, cl_mem *mem_objects, cl_uint num_devices,
                           cl_mem_migration_flags *flags, cl_uint *buffer)
{
  cl_uint i;
  cl_int  err;

  const size_t origin[3] = {0, 0, 0};
  const size_t region[3] = {IMAGE_DIM, IMAGE_DIM, 1};

  // If the image was previously migrated with undefined content, reload the content.

  for (i=0; i<num_devices; i++) {
    if (flags[i] & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
      if ((err = clEnqueueWriteImage(queues[i], mem_objects[i], CL_TRUE,
                                     origin, region, 0, 0, buffer, 0, NULL, NULL)) != CL_SUCCESS) {
        print_error(err, "Failed on restoration enqueue write of image data.");
        return err;
      }
    }
  }
  return CL_SUCCESS;
}

int test_image_migrate(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  int failed = 0;
  cl_uint i, j;
  cl_int err;
  cl_uint max_sub_devices = 0;
  cl_uint num_devices, num_devices_limited;
  cl_uint A[MEM_OBJ_SIZE], B[MEM_OBJ_SIZE], C[MEM_OBJ_SIZE];
  cl_uint test_number = 1;
  cl_device_affinity_domain domain, domains;
  cl_device_id *devices;
  cl_command_queue *queues;
  cl_mem_migration_flags *flagsA, *flagsB, *flagsC;
  cl_device_partition_property property[] = {CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, 0, 0};
  cl_mem *imageA, *imageB, *imageC;
  cl_mem_flags flags;
  cl_image_format format;
  cl_sampler sampler = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_context ctx = NULL;
  enum migrations migrateA, migrateB, migrateC;
  MTdata d = init_genrand(gRandomSeed);
  const size_t wgs[2] = {IMAGE_DIM, IMAGE_DIM};
  const size_t wls[2] = {1, 1};

  // Check for image support.
  if(checkForImageSupport(deviceID) == CL_IMAGE_FORMAT_NOT_SUPPORTED) {
    log_info("Device does not support images. Skipping test.\n");
    return 0;
  }

  // Allocate arrays whose size varies according to the maximum number of sub-devices.
  if ((err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_sub_devices), &max_sub_devices, NULL)) != CL_SUCCESS) {
    print_error(err, "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS) failed");
    return -1;
  }
  if (max_sub_devices < 1) {
    log_error("ERROR: Invalid number of compute units returned.\n");
    return -1;
  }

  devices = (cl_device_id *)malloc(max_sub_devices * sizeof(cl_device_id));
  queues = (cl_command_queue *)malloc(max_sub_devices * sizeof(cl_command_queue));
  flagsA = (cl_mem_migration_flags *)malloc(max_sub_devices * sizeof(cl_mem_migration_flags));
  flagsB = (cl_mem_migration_flags *)malloc(max_sub_devices * sizeof(cl_mem_migration_flags));
  flagsC = (cl_mem_migration_flags *)malloc(max_sub_devices * sizeof(cl_mem_migration_flags));
  imageA = (cl_mem *)malloc(max_sub_devices * sizeof(cl_mem));
  imageB = (cl_mem *)malloc(max_sub_devices * sizeof(cl_mem));
  imageC = (cl_mem *)malloc(max_sub_devices * sizeof(cl_mem));

  if ((devices == NULL) || (queues  == NULL) ||
      (flagsA  == NULL) || (flagsB  == NULL) || (flagsC == NULL) ||
      (imageA  == NULL) || (imageB == NULL)  || (imageC == NULL)) {
    log_error("ERROR: Failed to successfully allocate required local buffers.\n");
    failed = -1;
    goto cleanup_allocations;
  }

  for (i=0; i<max_sub_devices; i++) {
    devices[i] = NULL;
    queues [i] = NULL;
    imageA[i] = imageB[i] = imageC[i] = NULL;
  }

  for (i=0; i<MEM_OBJ_SIZE; i++) {
    A[i] = genrand_int32(d);
    B[i] = genrand_int32(d);
  }

  // Set image format.
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_UNSIGNED_INT32;


  // Attempt to partition the device along each of the allowed affinity domain.
  if ((err = clGetDeviceInfo(deviceID, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, sizeof(domains), &domains, NULL)) != CL_SUCCESS) {
    print_error(err, "clGetDeviceInfo(CL_PARTITION_AFFINITY_DOMAIN) failed");
    return -1;
  }

  domains &= (CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE | CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE |
              CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE | CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE | CL_DEVICE_AFFINITY_DOMAIN_NUMA);

  do {
    if (domains) {
      for (domain = 1; (domain & domains) == 0; domain <<= 1) {};
      domains &= ~domain;
    } else {
      domain = 0;
    }

    // Determine the number of partitions for the device given the specific domain.
    if (domain) {
      property[1] = domain;
      err = clCreateSubDevices(deviceID, (const cl_device_partition_property *)property, -1, NULL, &num_devices);
      if ((err != CL_SUCCESS) || (num_devices == 0)) {
        print_error(err, "Obtaining the number of partions by affinity failed.");
        failed = 1;
        goto cleanup;
      }
    } else {
      num_devices = 1;
    }

    if (num_devices > 1) {
      // Create each of the sub-devices and a corresponding context.
      if ((err = clCreateSubDevices(deviceID, (const cl_device_partition_property *)property, num_devices, devices, &num_devices)) != CL_SUCCESS) {
        print_error(err, "Failed creating sub devices.");
        failed = 1;
        goto cleanup;
      }

      // Create a context containing all the sub-devices
      ctx = clCreateContext(NULL, num_devices, devices, notify_callback, NULL, &err);
      if (ctx == NULL) {
    print_error(err, "Failed creating context containing the sub-devices.");
    failed = 1;
    goto cleanup;
      }

      // Create a command queue for each sub-device
      for (i=0; i<num_devices; i++) {
        if (devices[i]) {
          if ((queues[i] = clCreateCommandQueue(ctx, devices[i], 0, &err)) == NULL) {
            print_error(err, "Failed creating command queues.");
            failed = 1;
            goto cleanup;
          }
        }
      }
    } else {
      // No partitioning available. Just exercise the APIs on a single device.
      devices[0] = deviceID;
      queues[0] = queue;
      ctx = context;
    }

    // Build the kernel program.
    if ((err = create_single_kernel_helper(ctx, &program, &kernel, 1,
                                           &image_migrate_kernel_code,
                                           "test_image_migrate")))
    {
        print_error(err, "Failed creating kernel.");
        failed = 1;
        goto cleanup;
    }

    // Create sampler.
    sampler = clCreateSampler(ctx, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &err );
    if ((err != CL_SUCCESS) || !sampler) {
      print_error(err, "Failed to create a sampler.");
      failed = 1;
      goto cleanup;
    }

    num_devices_limited = num_devices;

    // Allocate memory buffers. 3 buffers (2 input, 1 output) for each sub-device.
    // If we run out of memory, then restrict the number of sub-devices to be tested.
    for (i=0; i<num_devices; i++) {
      imageA[i] = init_image(queues[i], create_image_2d(ctx, (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR),
                                                        &format, IMAGE_DIM, IMAGE_DIM, 0, NULL, &err), A);
      imageB[i] = init_image(queues[i], create_image_2d(ctx, (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR),
                                                        &format, IMAGE_DIM, IMAGE_DIM, 0, NULL, &err), B);
      imageC[i] = create_image_2d(ctx, (CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR),
                                  &format, IMAGE_DIM, IMAGE_DIM, 0, NULL, &err);

      if ((imageA[i] == NULL) || (imageB[i] == NULL) || (imageC[i] == NULL)) {
        if (i == 0) {
          log_error("Failed to allocate even 1 set of buffers.\n");
          failed = 1;
          goto cleanup;
        }
        num_devices_limited = i;
        break;
      }
    }

    // For each partition, we will execute the test kernel with each of the 3 buffers migrated to one of the migrate options
    for (migrateA=(enum migrations)(0); migrateA<NUMBER_OF_MIGRATIONS; migrateA = (enum migrations)((int)migrateA + 1)) {
      if (migrateMemObject(migrateA, queues, imageA, num_devices_limited, flagsA, d) != CL_SUCCESS) {
        failed = 1;
        goto cleanup;
      }
      for (migrateC=(enum migrations)(0); migrateC<NUMBER_OF_MIGRATIONS; migrateC = (enum migrations)((int)migrateC + 1)) {
        if (migrateMemObject(migrateC, queues, imageC, num_devices_limited, flagsC, d) != CL_SUCCESS) {
          failed = 1;
          goto cleanup;
        }
        for (migrateB=(enum migrations)(0); migrateB<NUMBER_OF_MIGRATIONS; migrateB = (enum migrations)((int)migrateB + 1)) {
          if (migrateMemObject(migrateB, queues, imageB, num_devices_limited, flagsB, d) != CL_SUCCESS) {
            failed = 1;
            goto cleanup;
          }
          // Run the test on each of the partitions.
          for (i=0; i<num_devices_limited; i++) {
            cl_uint x;

            x = i + test_number;

            if ((err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void *)&imageC[i])) != CL_SUCCESS) {
              print_error(err, "Failed set kernel argument 0.");
              failed = 1;
              goto cleanup;
            }

            if ((err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *)&imageA[i])) != CL_SUCCESS) {
              print_error(err, "Failed set kernel argument 1.");
              failed = 1;
              goto cleanup;
            }

            if ((err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *)&imageB[i])) != CL_SUCCESS) {
              print_error(err, "Failed set kernel argument 2.");
              failed = 1;
              goto cleanup;
            }

            if ((err = clSetKernelArg(kernel, 3, sizeof(cl_sampler), (const void *)&sampler)) != CL_SUCCESS) {
              print_error(err, "Failed set kernel argument 3.");
              failed = 1;
              goto cleanup;
            }

            if ((err = clSetKernelArg(kernel, 4, sizeof(cl_uint), (const void *)&x)) != CL_SUCCESS) {
              print_error(err, "Failed set kernel argument 4.");
              failed = 1;
              goto cleanup;
            }

            if ((err = clEnqueueNDRangeKernel(queues[i], kernel, 2, NULL, wgs, wls, 0, NULL, NULL)) != CL_SUCCESS) {
              print_error(err, "Failed enqueueing the NDRange kernel.");
              failed = 1;
              goto cleanup;
            }
          }
          // Verify the results as long as neither input is an undefined migration
          const size_t origin[3] = {0, 0, 0};
          const size_t region[3] = {IMAGE_DIM, IMAGE_DIM, 1};

          for (i=0; i<num_devices_limited; i++, test_number++) {
            if (((flagsA[i] | flagsB[i]) & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) == 0) {
              if ((err = clEnqueueReadImage(queues[i], imageC[i], CL_TRUE,
                                            origin, region, 0, 0, C, 0, NULL, NULL)) != CL_SUCCESS) {
                print_error(err, "Failed reading output buffer.");
                failed = 1;
                goto cleanup;
              }
              for (j=0; j<MEM_OBJ_SIZE; j++) {
                cl_uint expected;

                expected = A[j] ^ B[j] ^ test_number;
                if (C[j] != expected) {
                  log_error("Failed on device %d,  work item %4d,  expected 0x%08x got 0x%08x (0x%08x ^ 0x%08x ^ 0x%08x)\n", i, j, expected, C[j], A[j], B[j], test_number);
                  failed = 1;
                }
              }
              if (failed) goto cleanup;
            }
          }

          if (restoreImage(queues, imageB, num_devices_limited, flagsB, B) != CL_SUCCESS) {
            failed = 1;
            goto cleanup;
          }
        }
      }
      if (restoreImage(queues, imageA, num_devices_limited, flagsA, A) != CL_SUCCESS) {
        failed = 1;
        goto cleanup;
      }
    }

  cleanup:
    // Clean up all the allocted resources create by the test. This includes sub-devices,
    // command queues, and memory buffers.

    for (i=0; i<max_sub_devices; i++) {
      // Memory buffer cleanup
      if (imageA[i]) {
        if ((err = clReleaseMemObject(imageA[i])) != CL_SUCCESS) {
          print_error(err, "Failed releasing memory object.");
          failed = 1;
        }
      }
      if (imageB[i]) {
        if ((err = clReleaseMemObject(imageB[i])) != CL_SUCCESS) {
          print_error(err, "Failed releasing memory object.");
          failed = 1;
        }
      }
      if (imageC[i]) {
        if ((err = clReleaseMemObject(imageC[i])) != CL_SUCCESS) {
          print_error(err, "Failed releasing memory object.");
          failed = 1;
        }
      }

      if (num_devices > 1) {
        // Command queue cleanup
        if (queues[i]) {
          if ((err = clReleaseCommandQueue(queues[i])) != CL_SUCCESS) {
            print_error(err, "Failed releasing command queue.");
            failed = 1;
          }
        }

        // Sub-device cleanup
        if (devices[i]) {
          if ((err = clReleaseDevice(devices[i])) != CL_SUCCESS) {
            print_error(err, "Failed releasing sub device.");
            failed = 1;
          }
        }
        devices[i] = 0;
      }
    }

    // Sampler cleanup
    if (sampler) {
      if ((err = clReleaseSampler(sampler)) != CL_SUCCESS) {
    print_error(err, "Failed releasing sampler.");
    failed = 1;
      }
      sampler = NULL;
    }

    // Context, program, and kernel cleanup
    if (program) {
      if ((err = clReleaseProgram(program)) != CL_SUCCESS) {
    print_error(err, "Failed releasing program.");
    failed = 1;
      }
      program = NULL;
    }

    if (kernel) {
      if ((err = clReleaseKernel(kernel)) != CL_SUCCESS) {
    print_error(err, "Failed releasing kernel.");
    failed = 1;
      }
      kernel = NULL;
    }

    if (ctx && (ctx != context)) {
      if ((err = clReleaseContext(ctx)) != CL_SUCCESS) {
    print_error(err, "Failed releasing context.");
    failed = 1;
      }
    }
    ctx = NULL;

    if (failed) goto cleanup_allocations;
  } while (domains);

cleanup_allocations:
  if (devices) free(devices);
  if (queues)  free(queues);
  if (flagsA)  free(flagsA);
  if (flagsB)  free(flagsB);
  if (flagsC)  free(flagsC);
  if (imageA)  free(imageA);
  if (imageB)  free(imageB);
  if (imageC)  free(imageC);

  return ((failed) ? -1 : 0);
}
