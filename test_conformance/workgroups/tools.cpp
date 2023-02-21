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
#include "tools.h"
#include <sstream>
#include "harness/errorHelpers.h"

PrimeNumbersCollection PrimeNumbers::primeNumbers;
// Method generates prime numbers using Sieve of Eratosthenes algorithm
void PrimeNumbers::generatePrimeNumbers (unsigned int maxValue) {

  primeNumbers.clear();

  for (unsigned int i=2; i < maxValue; i++)
    primeNumbers.push_back(i);

  PrimeNumbersCollection::iterator it, it2;
  it = primeNumbers.begin();
  it2 = primeNumbers.begin();

  unsigned int maxValueSqrt = (unsigned int)sqrt((double)maxValue);

  for (; it != primeNumbers.end(); it++) {
    it2 = it;
    ++it2;
    if(*it>maxValueSqrt)
      break;
    for (;it2 != primeNumbers.end();)
      if (*it2 % *it == 0)
        it2 = primeNumbers.erase(it2);
      else
        ++it2;
  }
}

// Returns prime number for specified range
int PrimeNumbers::getPrimeNumberInRange (size_t lowerValue, size_t higherValue) {
  if(lowerValue >= higherValue)
    return -1;

  if(primeNumbers.back() < lowerValue)
    return -2;

  PrimeNumbersCollection::iterator it = primeNumbers.begin();

  for (; it != primeNumbers.end(); ++it) {
    if (lowerValue<*it) {
      if(higherValue>*it)
        return *it;
      else
        return -3;
    }
  }
  return -1;
}


int PrimeNumbers::getNextLowerPrimeNumber(size_t upperValue) {
    size_t retVal = 1;

    PrimeNumbersCollection::iterator it = primeNumbers.begin();

    for (; it != primeNumbers.end(); ++it) {
        if (upperValue > *it) {
            retVal = *it;
        } else {
            break;
        }
    }
    return retVal;
}

PrimeNumbers::Result1d PrimeNumbers::fitMaxPrime1d(size_t val1, size_t maxVal){

    PrimeNumbers::Result1d result;

    if (maxVal == 1) {
        result.Val1 = 1;
        return result;
    }

    while(val1 > maxVal)
    {
        val1 = PrimeNumbers::getNextLowerPrimeNumber(val1);
    }

    result.Val1 = val1;
    return result;
}

PrimeNumbers::Result2d PrimeNumbers::fitMaxPrime2d(size_t val1, size_t val2, size_t productMax) {

    PrimeNumbers::Result2d result;

    if (productMax == 1) {
        result.Val1 = 1;
        result.Val2 = 1;
        return result;
    }

    while ((val2 * val1) > productMax) {
        if ((val2 > val1) && (val2 > 1)) {
            val2 = PrimeNumbers::getNextLowerPrimeNumber(val2);
            continue;
        }
        if (val1 > 1) {
            val1 = PrimeNumbers::getNextLowerPrimeNumber(val1);
            continue;
        }
        break;
    }
    result.Val1 = val1;
    result.Val2 = val2;
    return result;
}


PrimeNumbers::Result3d PrimeNumbers::fitMaxPrime3d(size_t val1, size_t val2, size_t val3, size_t productMax) {

    Result3d result;

    if (productMax == 1) {
        result.Val1 = 1;
        result.Val2 = 1;
        result.Val3 = 1;
        return result;
    }

    while ((val3 * val2 * val1) > productMax) {
        if ((val3 > val2) && (val3 > val1) && (val3 > 1)) {
            val3 = PrimeNumbers::getNextLowerPrimeNumber(val3);
            continue;
        }
        if ((val2 > val1) && (val2 > 1)) {
            val2 = PrimeNumbers::getNextLowerPrimeNumber(val2);
            continue;
        }
        if (val1 > 1) {
            val1 = PrimeNumbers::getNextLowerPrimeNumber(val1);
            continue;
        }
        break;
    }
    result.Val1 = val1;
    result.Val2 = val2;
    result.Val3 = val3;
    return result;
}

namespace Error {
ErrorMap::value_type rawDataErrorString[] = {
  ErrorMap::value_type(ERR_GLOBAL_SIZE, "global size"),
  ErrorMap::value_type(ERR_GLOBAL_WORK_OFFSET, "global work offset"),
  ErrorMap::value_type(ERR_LOCAL_SIZE, "local size"),
  ErrorMap::value_type(ERR_GLOBAL_ID, "global id"),
  ErrorMap::value_type(ERR_LOCAL_ID, "local id"),
  ErrorMap::value_type(ERR_ENQUEUED_LOCAL_SIZE, "enqueued local size"),
  ErrorMap::value_type(ERR_LOCAL_SIZE, "local size"),
  ErrorMap::value_type(ERR_NUM_GROUPS, "num groups"),
  ErrorMap::value_type(ERR_GROUP_ID, "group id"),
  ErrorMap::value_type(ERR_WORK_DIM, "work dim"),
  ErrorMap::value_type(ERR_GLOBAL_BARRIER, "global barrier"),
  ErrorMap::value_type(ERR_LOCAL_BARRIER, "local barrier"),
  ErrorMap::value_type(ERR_GLOBAL_ATOMIC, "global atomic"),
  ErrorMap::value_type(ERR_LOCAL_ATOMIC, "local atomic"),
  ErrorMap::value_type(ERR_STRICT_MODE, "strict requirements failed. Wrong local work group size"),
  ErrorMap::value_type(ERR_BUILD_STATUS, "build status"),
  ErrorMap::value_type(ERR_UNKNOWN, "[unknown]"),
  ErrorMap::value_type(ERR_DIFFERENT, "[different]"),
};

const int numElems = sizeof(rawDataErrorString)/sizeof(rawDataErrorString[0]);
ErrorMap errorString (rawDataErrorString, rawDataErrorString+numElems);

ErrorClass::ErrorClass() {
  _overallNumberOfErrors = 0;
  _stats.clear();
  for (unsigned short i=0; i<sizeof(_errorArrayCounter)/sizeof(_errorArrayCounter[0]); i++) {
   _errorArrayCounter[i] = 0;
  }
}

void ErrorClass::show(Type err, std::string where, std::string additionalInfo) {
  ++_overallNumberOfErrors;

  err = (errorString.find(err) == errorString.end())?ERR_UNKNOWN:err;
  ++_stats[err];

  if (_overallNumberOfErrors == MAX_NUMBER_OF_PRINTED_ERRORS)
    printError("\t. . . Too many errors. Application will skip printing them.");

  if (_overallNumberOfErrors >= MAX_NUMBER_OF_PRINTED_ERRORS)
    return;

  std::string errString = "Error ";
  errString += errorString[err];
  errString += " appeared";

  if(where.compare("") != 0) {
    errString += " in ";
    errString += where;
  }

  if(additionalInfo.compare("") != 0) {
    errString += " ";
    errString += additionalInfo;
  }
  printError(errString);
}

void ErrorClass::show(Type whatErr, std::string where, cl_ulong valueIs, cl_ulong valueExpected) {
  std::ostringstream tmp;
  tmp << "(is: " << valueIs << ", expected: " << valueExpected << ")";
  show(whatErr, where, tmp.str());
}


void ErrorClass::show(std::string description) {
  ++_overallNumberOfErrors;
  ++_stats[ERR_DIFFERENT];
  if (_overallNumberOfErrors < MAX_NUMBER_OF_PRINTED_ERRORS)
    printError(description);

  if (_overallNumberOfErrors == MAX_NUMBER_OF_PRINTED_ERRORS)
    printError("\t. . . Too many errors. Application will skip printing them.");
}

void ErrorClass::printError(std::string errString) {
  log_error ("%s\n", errString.c_str());
}

void ErrorClass::showStats() {

  Type err;
  log_info ("T E S T  S U M M A R Y:\n");
  for (ErrorStats::iterator it = _stats.begin(); it != _stats.end(); it++) {
    err = (errorString.find(it->first) == errorString.end())?ERR_UNKNOWN:it->first;
    std::string errName = errorString[err];
    log_info("Error %s:\t%d\n", errName.c_str(), it->second);
  }

  log_info("Overall number of errors:\t%d\n", _overallNumberOfErrors);

}

bool ErrorClass::checkError() {
  return _overallNumberOfErrors > 0;
}

// This method is required to synchronize errors counters between kernel and host
void ErrorClass::synchronizeStatsMap() {
  for (unsigned short i=0; i<sizeof(_errorArrayCounter)/sizeof(_errorArrayCounter[0]); i++) {
    if(_errorArrayCounter[i] == 0)
      continue;

    _stats[static_cast<Type>(i)] += _errorArrayCounter[i];
    _overallNumberOfErrors += _errorArrayCounter[i];
  }

}

}
