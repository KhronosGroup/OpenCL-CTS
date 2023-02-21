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
  test_non_uniform_1d_basic(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::BASIC);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_1d_max_wg_size_plus_1_basic
  {
    size_t globalSize[] = {maxWgSize+1};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_prime_number_basic
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_max_wg_size_plus_prime_number_basic
  {
    int primeNumber = 11;
    size_t globalSize[] = {maxWgSize+primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_max_wg_size_plus_prime_number_basic_2
  {
    int primeNumber = 53;
    size_t globalSize[] = {maxWgSize+primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_2max_wg_size_minus_1_basic
  {
    size_t globalSize[] = {2*maxWgSize - 1};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_prime_number_basic_2
  {
    unsigned int primeNumber = 20101;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_prime_number_basic_3
  {
    unsigned int primeNumber = 42967;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_prime_number_basic_4
  {
    unsigned int primeNumber = 65521;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_prime_number_and_ls_null_basic_2
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_prime_number_and_ls_null_basic_3
  {
    unsigned int primeNumber = 65521;
    size_t globalSize[] = {primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  // non_uniform_1d_two_prime_numbers_basic
  {
    unsigned int primeNumber = 42967;
    unsigned int primeNumber2 = 113;
    PrimeNumbers::Result1d fit1dResult;

    fit1dResult = PrimeNumbers::fitMaxPrime1d(primeNumber2, maxWgSize );

    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {fit1dResult.Val1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BASIC);
  }

  return exec.status();
}

int
  test_non_uniform_1d_atomics(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::ATOMICS);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_1d_max_wg_size_plus_1_atomics
  {
    size_t globalSize[] = {maxWgSize+1};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_prime_number_atomics
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_max_wg_size_plus_prime_number_atomics
  {
    int primeNumber = 11;
    size_t globalSize[] = {maxWgSize+primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_max_wg_size_plus_prime_number_atomics_2
  {
    int primeNumber = 53;
    size_t globalSize[] = {maxWgSize+primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_2max_wg_size_minus_1_atomics
  {
    size_t globalSize[] = {2*maxWgSize - 1};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_prime_number_atomics_2
  {
    unsigned int primeNumber = 20101;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_prime_number_atomics_3
  {
    unsigned int primeNumber = 42967;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_prime_number_atomics_4
  {
    unsigned int primeNumber = 65521;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_prime_number_and_ls_null_atomics_2
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_prime_number_and_ls_null_atomics_3
  {
    unsigned int primeNumber = 65521;
    size_t globalSize[] = {primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  // non_uniform_1d_two_prime_numbers_atomics
  {
    unsigned int primeNumber = 42967;
    unsigned int primeNumber2 = 113;
    PrimeNumbers::Result1d fit1dResult;

    fit1dResult = PrimeNumbers::fitMaxPrime1d(primeNumber2, maxWgSize );

    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {fit1dResult.Val1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::ATOMICS);
  }

  return exec.status();
}

int
  test_non_uniform_1d_barriers(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::BARRIERS);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_1d_max_wg_size_plus_1_barriers
  {
    size_t globalSize[] = {maxWgSize+1};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_prime_number_barriers
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_max_wg_size_plus_prime_number_barriers
  {
    int primeNumber = 11;
    size_t globalSize[] = {maxWgSize+primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_max_wg_size_plus_prime_number_barriers_2
  {
    int primeNumber = 53;
    size_t globalSize[] = {maxWgSize+primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_2max_wg_size_minus_1_barriers
  {
    size_t globalSize[] = {2*maxWgSize - 1};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_prime_number_barriers_2
  {
    unsigned int primeNumber = 20101;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_prime_number_barriers_3
  {
    unsigned int primeNumber = 42967;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_prime_number_barriers_4
  {
    unsigned int primeNumber = 65521;
    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {maxWgSize};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_prime_number_and_ls_null_barriers_2
  {
    int primeNumber = PrimeNumbers::getPrimeNumberInRange(maxWgSize, 2*maxWgSize);
    if (primeNumber < 1) {
      log_error ("Cannot find proper prime number.");
      return -1;
    }
    size_t globalSize[] = {primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_prime_number_and_ls_null_barriers_3
  {
    unsigned int primeNumber = 65521;
    size_t globalSize[] = {primeNumber};
    size_t *localSize = NULL;

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  // non_uniform_1d_two_prime_numbers_barriers
  {
    unsigned int primeNumber = 42967;
    unsigned int primeNumber2 = 113;

    PrimeNumbers::Result1d fit1dResult;

    fit1dResult = PrimeNumbers::fitMaxPrime1d(primeNumber2, maxWgSize );

    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {fit1dResult.Val1};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, Range::BARRIERS);
  }

  return exec.status();
}
