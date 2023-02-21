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
  test_non_uniform_other_basic(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::BASIC);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_1d_two_prime_numbers_offset_basic
  {
    unsigned int primeNumber = 42967;
    unsigned int primeNumber2 = 113;
    PrimeNumbers::Result1d fit1dResult;

    fit1dResult = PrimeNumbers::fitMaxPrime1d(primeNumber2, maxWgSize );

    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {fit1dResult.Val1};
    size_t offset[] = {23};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::BASIC);
  }

  // non_uniform_2d_three_prime_numbers_offset_basic
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
    size_t offset[] = {23, 17};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::BASIC);
  }

  // non_uniform_3d_six_prime_numbers_offset_basic
  {
    unsigned int primeNumber = 373;
    unsigned int primeNumber2 = 13;
    unsigned int primeNumber3 = 279;
    unsigned int primeNumber4 = 3;
    unsigned int primeNumber5 = 5;
    unsigned int primeNumber6 = 7;

    PrimeNumbers::Result3d fit3dResult;

    size_t globalSize[] = {primeNumber, primeNumber2, primeNumber3};

    fit3dResult = PrimeNumbers::fitMaxPrime3d(primeNumber4, primeNumber5, primeNumber6, maxWgSize );

    size_t localSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};
    size_t offset[] = {11, 23, 17};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::BASIC);
  }

  // non_uniform_3d_six_prime_numbers_rwgs_basic
  {
    unsigned int primeNumber = 373;
    unsigned int primeNumber2 = 13;
    unsigned int primeNumber3 = 279;
    unsigned int primeNumber4 = 3;
    unsigned int primeNumber5 = 5;
    unsigned int primeNumber6 = 7;
    PrimeNumbers::Result3d fit3dResult;

    fit3dResult = PrimeNumbers::fitMaxPrime3d(primeNumber4, primeNumber5, primeNumber6, maxWgSize );

    size_t globalSize[] = {primeNumber, primeNumber2, primeNumber3};
    size_t localSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};
    size_t reqdWorkGroupSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, NULL, reqdWorkGroupSize, Range::BASIC);
  }

  return exec.status();
}

int
  test_non_uniform_other_atomics(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::ATOMICS);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_1d_two_prime_numbers_offset_atomics
  {
    unsigned int primeNumber = 42967;
    unsigned int primeNumber2 = 113;
    PrimeNumbers::Result1d fit1dResult;

    fit1dResult = PrimeNumbers::fitMaxPrime1d(primeNumber2, maxWgSize );

    size_t globalSize[] = {primeNumber};
    size_t localSize[] = {fit1dResult.Val1};
    size_t offset[] = {23};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::ATOMICS);
  }

  // non_uniform_2d_three_prime_numbers_offset_atomics
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
    size_t offset[] = {23, 17};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::ATOMICS);
  }

  // non_uniform_3d_six_prime_numbers_offset_atomics
  {
    unsigned int primeNumber = 373;
    unsigned int primeNumber2 = 13;
    unsigned int primeNumber3 = 279;
    unsigned int primeNumber4 = 3;
    unsigned int primeNumber5 = 5;
    unsigned int primeNumber6 = 7;
    PrimeNumbers::Result3d fit3dResult;

    fit3dResult = PrimeNumbers::fitMaxPrime3d(primeNumber4, primeNumber5, primeNumber6, maxWgSize );

    size_t globalSize[] = {primeNumber, primeNumber2, primeNumber3};
    size_t localSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};
    size_t offset[] = {11, 23, 17};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::ATOMICS);
  }

  // non_uniform_3d_six_prime_numbers_rwgs_atomics
  {
    unsigned int primeNumber = 373;
    unsigned int primeNumber2 = 13;
    unsigned int primeNumber3 = 279;
    unsigned int primeNumber4 = 3;
    unsigned int primeNumber5 = 5;
    unsigned int primeNumber6 = 7;
    PrimeNumbers::Result3d fit3dResult;

    fit3dResult = PrimeNumbers::fitMaxPrime3d(primeNumber4, primeNumber5, primeNumber6, maxWgSize );

    size_t globalSize[] = {primeNumber, primeNumber2, primeNumber3};
    size_t localSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};
    size_t reqdWorkGroupSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, NULL, reqdWorkGroupSize, Range::ATOMICS);
  }

  return exec.status();
}

int
  test_non_uniform_other_barriers(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  SubTestExecutor exec(device, context, queue);

  size_t maxWgSize;
  int err;
  err = exec.calculateWorkGroupSize(maxWgSize, Range::BARRIERS);
  if (err) {
    log_error ("Cannot calculate work group size.");
    return -1;
  }

  // non_uniform_1d_two_prime_numbers_offset_barriers
  {
    unsigned int primeNumber = 42967;
    unsigned int primeNumber2 = 113;
    PrimeNumbers::Result1d fit1dResult;

    fit1dResult = PrimeNumbers::fitMaxPrime1d(primeNumber2, maxWgSize );

    size_t globalSize[] = {primeNumber};

    size_t localSize[] = {fit1dResult.Val1};
    size_t offset[] = {23};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::BARRIERS);
  }

  // non_uniform_2d_three_prime_numbers_offset_barriers
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
    size_t offset[] = {23, 17};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::BARRIERS);
  }

  // non_uniform_3d_six_prime_numbers_offset_barriers
  {
    unsigned int primeNumber = 373;
    unsigned int primeNumber2 = 13;
    unsigned int primeNumber3 = 279;
    unsigned int primeNumber4 = 3;
    unsigned int primeNumber5 = 5;
    unsigned int primeNumber6 = 7;
    PrimeNumbers::Result3d fit3dResult;

    fit3dResult = PrimeNumbers::fitMaxPrime3d(primeNumber4, primeNumber5, primeNumber6, maxWgSize );

    size_t globalSize[] = {primeNumber, primeNumber2, primeNumber3};

    size_t localSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};
    size_t offset[] = {11, 23, 17};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, offset, NULL, Range::BARRIERS);
  }

  // non_uniform_3d_six_prime_numbers_rwgs_barriers
  {
    unsigned int primeNumber = 373;
    unsigned int primeNumber2 = 13;
    unsigned int primeNumber3 = 279;
    unsigned int primeNumber4 = 3;
    unsigned int primeNumber5 = 5;
    unsigned int primeNumber6 = 7;
    PrimeNumbers::Result3d fit3dResult;

    fit3dResult = PrimeNumbers::fitMaxPrime3d(primeNumber4, primeNumber5, primeNumber6, maxWgSize );

    size_t globalSize[] = {primeNumber, primeNumber2, primeNumber3};

    size_t localSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};
    size_t reqdWorkGroupSize[] = {fit3dResult.Val1, fit3dResult.Val2, fit3dResult.Val3};

    exec.runTestNonUniformWorkGroup(sizeof(globalSize)/sizeof(globalSize[0]), globalSize, localSize, NULL, reqdWorkGroupSize, Range::BARRIERS);
  }

  return exec.status();
}
