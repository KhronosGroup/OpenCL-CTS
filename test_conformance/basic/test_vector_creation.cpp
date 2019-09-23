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
#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"




#define DEBUG 0
#define DEPTH 16
// Limit the maximum code size for any given kernel.
#define MAX_CODE_SIZE (1024*32)

const int sizes[] = {1, 2, 3, 4, 8, 16, -1, -1, -1, -1};
const char *size_names[] = {"", "2", "3", "4", "8", "16" , "!!a", "!!b", "!!c", "!!d"};

// Creates a kernel by enumerating all possible ways of building the vector out of vloads
// skip_to_results will skip results up to a given number. If the amount of code generated
// is greater than MAX_CODE_SIZE, this function will return the number of results used,
// which can then be used as the skip_to_result value to continue where it left off.
int create_kernel(ExplicitType type, int output_size, char *program, int *number_of_results, int skip_to_result) {

    int number_of_sizes;

    switch (output_size) {
        case 1:
            number_of_sizes = 1;
            break;
        case 2:
            number_of_sizes = 2;
            break;
        case 3:
            number_of_sizes = 3;
            break;
        case 4:
            number_of_sizes = 4;
            break;
        case 8:
            number_of_sizes = 5;
            break;
        case 16:
            number_of_sizes = 6;
            break;
        default:
            log_error("Invalid size: %d\n", output_size);
            return -1;
    }

    int total_results = 0;
    int current_result = 0;
    int total_vloads = 0;
    int total_program_length = 0;
    int aborted_due_to_size = 0;

    if (skip_to_result < 0)
        skip_to_result = 0;

    // The line of code for the vector creation
    char line[1024];
    // Keep track of what size vector we are using in each position so we can iterate through all fo them
    int pos[DEPTH];
    int max_size = output_size;
    if (DEBUG > 1) log_info("max_size: %d\n", max_size);

    program[0] = '\0';
    sprintf(program, "%s\n__kernel void test_vector_creation(__global %s *src, __global %s%s *result) {\n",
            type == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
            get_explicit_type_name(type), get_explicit_type_name(type), ( number_of_sizes == 3 ) ? "" : size_names[number_of_sizes-1]);
    total_program_length += (int)strlen(program);

    char storePrefix[ 128 ], storeSuffix[ 128 ];

    // Start out trying sizes 1,1,1,1,1...
    for (int i=0; i<DEPTH; i++)
        pos[i] = 0;

    int done = 0;
    while (!done) {
        if (DEBUG > 1) {
            log_info("pos size[] = [");
            for (int k=0; k<DEPTH; k++)
                log_info(" %d ", pos[k]);
            log_info("]\n");
        }

        // Go through the selected vector sizes and see if the first n of them fit the
        //  required size exactly.
        int size_so_far = 0;
        int vloads;
        for ( vloads=0; vloads<DEPTH; vloads++) {
            if (size_so_far + sizes[pos[vloads]] <= max_size) {
                size_so_far += sizes[pos[vloads]];
            } else {
                break;
            }
        }
        if (DEBUG > 1)  log_info("vloads: %d, size_so_far:%d\n", vloads, size_so_far);

        // If they did not fit the required size exactly it is too long, so there is no point in checking any other combinations
        //  of the sizes to the right. Prune them from the search.
        if (size_so_far != max_size) {
            // Zero all the sizes to the right
            for (int k=vloads+1; k<DEPTH; k++) {
                pos[k] = 0;
            }
            // Increment this current size and propagate the values up if needed
            for (int d=vloads; d>=0; d--) {
                pos[d]++;
                if (pos[d] >= number_of_sizes) {
                    pos[d] = 0;
                    if (d == 0) {
                        // If we rolled over then we are done
                        done = 1;
                        break;
                    }
                } else {
                    break;
                }
            }
            // Go on to the next size since this one (and all others "under" it) didn't fit
            continue;
        }


        // Generate the actual load line if we are building this part
        line[0]= '\0';
        if (skip_to_result == 0 || total_results >= skip_to_result) {
            if( number_of_sizes == 3 )
            {
                sprintf( storePrefix, "vstore3( " );
                sprintf( storeSuffix, ", %d, result )", current_result );
            }
            else
            {
                sprintf( storePrefix, "result[%d] = ", current_result );
                storeSuffix[ 0 ] = 0;
            }

            sprintf(line, "\t%s(%s%d)(", storePrefix, get_explicit_type_name(type), output_size);
            current_result++;

            int offset = 0;
            for (int i=0; i<vloads; i++) {
                if (pos[i] == 0)
                    sprintf(line + strlen(line), "src[%d]", offset);
                else
                    sprintf(line + strlen(line), "vload%s(0,src+%d)", size_names[pos[i]], offset);
                offset += sizes[pos[i]];
                if (i<(vloads-1))
                    sprintf(line + strlen(line), ",");
            }
            sprintf(line + strlen(line), ")%s;\n", storeSuffix);

            strcat(program, line);
            total_vloads += vloads;
        }
        total_results++;
        total_program_length += (int)strlen(line);
        if (total_program_length > MAX_CODE_SIZE) {
            aborted_due_to_size = 1;
            done = 1;
        }


        if (DEBUG) log_info("line is: %s", line);

        // If we did not use all of them, then we ignore any changes further to the right.
        // We do this by causing those loops to skip on the next iteration.
        if (vloads < DEPTH) {
            if (DEBUG > 1) log_info("done with this depth\n");
            for (int k=vloads; k<DEPTH; k++)
                pos[k] = number_of_sizes;
        }

        // Increment the far right size by 1, rolling over as needed
        for (int d=DEPTH-1; d>=0; d--) {
            pos[d]++;
            if (pos[d] >= number_of_sizes) {
                pos[d] = 0;
                if (d == 0) {
                    // If we rolled over at the far-left then we are done
                    done = 1;
                    break;
                }
            } else {
                break;
            }
        }
        if (done)
            break;

        // Continue until we are done.
    }
    strcat(program, "}\n\n"); //log_info("%s\n", program);
    total_program_length += 3;
    if (DEBUG) log_info("\t\t(Program for vector type %s%s contains %d vector creations, of total program length %gkB, with a total of %d vloads.)\n",
                        get_explicit_type_name(type), size_names[number_of_sizes-1], total_results, total_program_length/1024.0, total_vloads);
    *number_of_results = current_result;
    if (aborted_due_to_size)
        return total_results;
    return 0;
}




int test_vector_creation(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16};

    char *program_source;
    int error;
    int total_errors = 0;

    cl_int input_data_int[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    cl_double input_data_double[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    void *input_data_converted;
    void *output_data;

    int number_of_results;;

    input_data_converted = malloc(sizeof(cl_double)*16);
    program_source = (char*)malloc(sizeof(char)*1024*1024*4);

    // Iterate over all the types
    for (int type_index=0; type_index<10; type_index++) {
    if(!gHasLong && ((vecType[type_index] == kLong)  || (vecType[type_index] == kULong)))
    {
      log_info("Long/ULong data type not supported on this device\n");
      continue;
    }

        clMemWrapper input;

        if (vecType[type_index] == kDouble) {
            if (!is_extension_available(deviceID, "cl_khr_fp64")) {
                log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
                continue;
            }
            log_info("Testing doubles.\n");
        }

        // Convert the data to the right format for the test.
        memset(input_data_converted, 0xff, sizeof(cl_double)*16);
        if (vecType[type_index] != kDouble) {
            for (int j=0; j<16; j++) {
                convert_explicit_value(&input_data_int[j], ((char*)input_data_converted)+get_explicit_type_size(vecType[type_index])*j,
                                       kInt, 0, kRoundToEven, vecType[type_index]);
            }
        } else {
            memcpy(input_data_converted, &input_data_double, sizeof(cl_double)*16);
        }

        input = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, get_explicit_type_size(vecType[type_index])*16,
                               (vecType[type_index] != kDouble) ? input_data_converted : input_data_double, &error);
        if (error) {
            print_error(error, "clCreateBuffer failed");
            total_errors++;
            continue;
        }

        // Iterate over all the vector sizes.
        for (int size_index=1; size_index< 5; size_index++) {
            size_t global[] = {1,1,1};
            int number_generated = -1;
            int previous_number_generated = 0;

            log_info("Testing %s%s...\n", get_explicit_type_name(vecType[type_index]), size_names[size_index]);
            while (number_generated != 0) {
                clMemWrapper output;
                clKernelWrapper kernel;
                clProgramWrapper program;

                number_generated = create_kernel(vecType[type_index], vecSizes[size_index], program_source, &number_of_results, number_generated);
                if (number_generated != 0) {
                    if (previous_number_generated == 0)
                        log_info("Code size greater than %gkB; splitting test into multiple kernels.\n", MAX_CODE_SIZE/1024.0);
                    log_info("\tExecuting vector permutations %d to %d...\n", previous_number_generated, number_generated-1);
                }

                error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&program_source, "test_vector_creation");
                if (error) {
                    log_error("create_single_kernel_helper failed.\n");
                    total_errors++;
                    break;
                }

                output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        number_of_results*get_explicit_type_size(vecType[type_index])*vecSizes[size_index],
                                        NULL, &error);
                if (error) {
                    print_error(error, "clCreateBuffer failed");
                    total_errors++;
                    break;
                }

                error = clSetKernelArg(kernel, 0, sizeof(input), &input);
                error |= clSetKernelArg(kernel, 1, sizeof(output), &output);
                if (error) {
                    print_error(error, "clSetKernelArg failed");
                    total_errors++;
                    break;
                }

                error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
                if (error) {
                    print_error(error, "clEnqueueNDRangeKernel failed");
                    total_errors++;
                    break;
                }

                error = clFinish(queue);
                if (error) {
                    print_error(error, "clFinish failed");
                    total_errors++;
                    break;
                }

                output_data = malloc(number_of_results*get_explicit_type_size(vecType[type_index])*vecSizes[size_index]);
                if (output_data == NULL) {
                    log_error("Failed to allocate memory for output data.\n");
                    total_errors++;
                    break;
                }
                memset(output_data, 0xff, number_of_results*get_explicit_type_size(vecType[type_index])*vecSizes[size_index]);
                error = clEnqueueReadBuffer(queue, output, CL_TRUE, 0,
                                            number_of_results*get_explicit_type_size(vecType[type_index])*vecSizes[size_index],
                                            output_data, 0, NULL, NULL);
                if (error) {
                    print_error(error, "clEnqueueReadBuffer failed");
                    total_errors++;
                    free(output_data);
                    break;
                }

                // Check the results
                char *res = (char *)output_data;
                char *exp = (char *)input_data_converted;
                for (int i=0; i<number_of_results; i++) {
                    // If they do not match, then print out why
                    if (memcmp(input_data_converted,
                               res + i*(get_explicit_type_size(vecType[type_index])*vecSizes[size_index]),
                               get_explicit_type_size(vecType[type_index])*vecSizes[size_index])
                        ) {
                        log_error("Data failed to validate for result %d\n", i);

                        // Find the line in the program that failed. This is ugly.
                        char search[32];
                        char found_line[1024];
                        found_line[0]='\0';
                        search[0]='\0';
                        sprintf(search, "result[%d] = (", i);
                        char *start_loc = strstr(program_source, search);
                        if (start_loc == NULL)
                            log_error("Failed to find program source for failure for %s in \n%s", search, program_source);
                        else {
                          char *end_loc = strstr(start_loc, "\n");
                          memcpy(&found_line, start_loc, (end_loc-start_loc));
                          found_line[end_loc-start_loc]='\0';
                          log_error("Failed vector line: %s\n", found_line);
                        }

                        for (int j=0; j<(int)vecSizes[size_index]; j++) {
                            char expected_value[64];
                            char returned_value[64];
                            expected_value[0]='\0';
                            returned_value[0]='\0';
                            print_type_to_string(vecType[type_index], (void*)(res+get_explicit_type_size(vecType[type_index])*(i*vecSizes[size_index]+j)), returned_value);
                            print_type_to_string(vecType[type_index], (void*)(exp+get_explicit_type_size(vecType[type_index])*j), expected_value);
                            log_error("index [%d, component %d]: got: %s expected: %s\n", i, j,
                                      returned_value, expected_value);
                        }

                        total_errors++;
                    }
                }
                free(output_data);
                previous_number_generated = number_generated;
            } // number_generated != 0

        } // vector sizes
    } // vector types

    free(input_data_converted);
    free(program_source);

    return total_errors;
}


