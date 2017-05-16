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
#ifndef _TOOLS_H
#define _TOOLS_H

#include "procs.h"
#include <vector>
#include <map>
#include <string>

typedef std::vector<unsigned int> PrimeNumbersCollection;



// Class responsible for distributing prime numbers
class PrimeNumbers {

public:
  struct Result1d{
        size_t Val1;
  };

  struct Result2d{
        size_t Val1;
        size_t Val2;
  };

  struct Result3d{
        size_t Val1;
        size_t Val2;
        size_t Val3;
  };

  static void generatePrimeNumbers (unsigned int maxValue);
  static int getPrimeNumberInRange (size_t lowerValue, size_t higherValue);
  static int getNextLowerPrimeNumber (size_t upperValue);
  static Result1d fitMaxPrime1d(size_t Val1, size_t productMax);
  // Return val1 and Val2 which are largest prime numbers who's product is <= productMax
  static Result2d fitMaxPrime2d(size_t Val1, size_t Val2, size_t productMax);
  // Return val1, val2 and val3, which are largest prime numbers who's product is <= productMax
  static Result3d fitMaxPrime3d(size_t Val1, size_t Val2,  size_t Val3, size_t productMax);
private:
  static PrimeNumbersCollection primeNumbers;
  PrimeNumbers();
};

// Stores information about errors
namespace Error {
#define MAX_NUMBER_OF_PRINTED_ERRORS 10
  enum Type{
    ERR_GLOBAL_SIZE=0,
    ERR_GLOBAL_WORK_OFFSET,
    ERR_LOCAL_SIZE,
    ERR_GLOBAL_ID,
    ERR_LOCAL_ID,
    ERR_ENQUEUED_LOCAL_SIZE,
    ERR_NUM_GROUPS,
    ERR_GROUP_ID,
    ERR_WORK_DIM,
    ERR_GLOBAL_BARRIER,
    ERR_LOCAL_BARRIER,
    ERR_GLOBAL_ATOMIC,
    ERR_LOCAL_ATOMIC,

    ERR_STRICT_MODE,
    ERR_BUILD_STATUS,

    ERR_UNKNOWN,
    ERR_DIFFERENT,
    _LAST_ELEM
  };

  typedef std::map<Type, std::string> ErrorMap;
  typedef std::map<Type, unsigned int> ErrorStats;

  class ErrorClass {
  public:
    ErrorClass();
    void show(Type whatErr, std::string where="", std::string additionalInfo="");
    void show(Type whatErr, std::string where, cl_ulong valueIs, cl_ulong valueExpected);
    void show(std::string description);
    bool checkError();
    void showStats();
    void synchronizeStatsMap();
    cl_uint * errorArrayCounter() {return _errorArrayCounter;};
    size_t errorArrayCounterSize() {return sizeof(_errorArrayCounter);};
  private:
    cl_uint _errorArrayCounter[Error::_LAST_ELEM]; // this buffer is passed to kernel
    int _overallNumberOfErrors;
    ErrorStats _stats;
    void printError(std::string errString);

  };

}
#endif // _TOOLS_H
