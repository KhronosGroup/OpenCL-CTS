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
#include "tools.h"

#include "TestNonUniformWorkGroup.h"

int
  test_non_uniform_2d_basic(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::BASIC);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_2d_max_wg_size_plus_1_basic
  {
    size_t globalSize[] = {maxWgSize+1, maxWgSize};
    size_t localSize[] = {maxWgSize, 1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_prime_number_basic
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber, maxWgSize};
    size_t localSize[] = {maxWgSize/2, 2};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_two_prime_numbers_basic
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    int primeNumber2 = 1759;
    size_t globalSize[] = {primeNumber2, primeNumber};
    size_t localSize[] = {16, maxWgSize/16};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_prime_number_basic_2
  {
    int primeNumber = 1327;
    size_t globalSize[] = {primeNumber, primeNumber};
    size_t localSize[] = {maxWgSize/32, 32};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_combination_of_max_wg_size_basic
  {
    size_t globalSize[] = {maxWgSize + 2, maxWgSize + 4};
    size_t localSize[] = {maxWgSize/32, 32};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_two_prime_numbers_and_ls_null_basic
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    unsigned int primeNumber2 = 1669;
    size_t globalSize[] = {primeNumber, primeNumber2};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_prime_number_and_ls_null_basic
  {
    unsigned int primeNumber = 1249;
    size_t globalSize[] = {primeNumber, primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_four_prime_numbers_basic
  {
    unsigned int primeNumber = 1951;
    unsigned int primeNumber2 = 911;
    unsigned int primeNumber3 = 13;
    unsigned int primeNumber4 = 17;

    PrimeNumbers::Result2d fit2dResult;
    fit2dResult = PrimeNumbers::fitMaxPrime2d(primeNumber3, primeNumber4, maxWgSize);

    size_t globalSize[] = {primeNumber, primeNumber2};
    size_t localSize[] =  {fit2dResult.Val1, fit2dResult.Val2};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_2d_three_prime_numbers_basic
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize/2, maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    unsigned int primeNumber2 = 42967;
    unsigned int primeNumber3 = 13;
    size_t globalSize[] = {primeNumber2, primeNumber3};
    size_t localSize[] = {primeNumber, 1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  return exec.status();
}

int
  test_non_uniform_2d_atomics(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::ATOMICS);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_2d_max_wg_size_plus_1_atomics
  {
    size_t globalSize[] = {maxWgSize+1, maxWgSize};
    size_t localSize[] = {maxWgSize, 1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_prime_number_atomics
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber, maxWgSize};
    size_t localSize[] = {maxWgSize/2, 2};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_two_prime_numbers_atomics
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    int primeNumber2 = 1759;
    size_t globalSize[] = {primeNumber2, primeNumber};
    size_t localSize[] = {16, maxWgSize/16};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_prime_number_atomics_2
  {
    int primeNumber = 1327;
    size_t globalSize[] = {primeNumber, primeNumber};
    size_t localSize[] = {maxWgSize/32, 32};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_combination_of_max_wg_size_atomics
  {
    size_t globalSize[] = {maxWgSize + 2, maxWgSize + 4};
    size_t localSize[] = {maxWgSize/32, 32};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_two_prime_numbers_and_ls_null_atomics
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    unsigned int primeNumber2 = 1669;
    size_t globalSize[] = {primeNumber, primeNumber2};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_prime_number_and_ls_null_atomics
  {
    unsigned int primeNumber = 1249;
    size_t globalSize[] = {primeNumber, primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_four_prime_numbers_atomics
  {
    unsigned int primeNumber = 1951;
    unsigned int primeNumber2 = 911;
    unsigned int primeNumber3 = 13;
    unsigned int primeNumber4 = 17;

    PrimeNumbers::Result2d fit2dResult;
    fit2dResult = PrimeNumbers::fitMaxPrime2d(primeNumber3, primeNumber4, maxWgSize);

    size_t globalSize[] = {primeNumber, primeNumber2};
    size_t localSize[] = {fit2dResult.Val1, fit2dResult.Val2};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_2d_three_prime_numbers_atomics
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize/2, maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    unsigned int primeNumber2 = 42967;
    unsigned int primeNumber3 = 13;
    size_t globalSize[] = {primeNumber2, primeNumber3};
    size_t localSize[] = {primeNumber, 1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  return exec.status();
}

int
  test_non_uniform_2d_barriers(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::BARRIERS);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_2d_max_wg_size_plus_1_barriers
  {
    size_t globalSize[] = {maxWgSize+1, maxWgSize};
    size_t localSize[] = {maxWgSize, 1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_prime_number_barriers
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber, maxWgSize};
    size_t localSize[] = {maxWgSize/2, 2};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_two_prime_numbers_barriers
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    int primeNumber2 = 1759;
    size_t globalSize[] = {primeNumber2, primeNumber};
    size_t localSize[] = {16, maxWgSize/16};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_prime_number_barriers_2
  {
    int primeNumber = 1327;
    size_t globalSize[] = {primeNumber, primeNumber};
    size_t localSize[] = {maxWgSize/32, 32};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_combination_of_max_wg_size_barriers
  {
    size_t globalSize[] = {maxWgSize + 2, maxWgSize + 4};
    size_t localSize[] = {maxWgSize/32, 32};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_two_prime_numbers_and_ls_null_barriers
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    unsigned int primeNumber2 = 1669;
    size_t globalSize[] = {primeNumber, primeNumber2};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_prime_number_and_ls_null_barriers
  {
    unsigned int primeNumber = 1249;
    size_t globalSize[] = {primeNumber, primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_four_prime_numbers_barriers
  {
    unsigned int primeNumber = 1951;
    unsigned int primeNumber2 = 911;
    unsigned int primeNumber3 = 13;
    unsigned int primeNumber4 = 17;
    PrimeNumbers::Result2d fit2dResult;
    fit2dResult = PrimeNumbers::fitMaxPrime2d(primeNumber3, primeNumber4, maxWgSize);
    size_t globalSize[] = {primeNumber, primeNumber2};
    size_t localSize[] = {fit2dResult.Val1, fit2dResult.Val2};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_2d_three_prime_numbers_barriers
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize/2, maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    unsigned int primeNumber2 = 42967;
    unsigned int primeNumber3 = 13;
    size_t globalSize[] = {primeNumber2, primeNumber3};
    size_t localSize[] = {primeNumber, 1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  return exec.status();
}
