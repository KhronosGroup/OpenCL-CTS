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
#include "../../test_common/harness/testHarness.h"
#include "TestNonUniformWorkGroup.h"

basefn    basefn_list[] = {
  test_non_uniform_1d_basic,
  test_non_uniform_1d_atomics,
  test_non_uniform_1d_barriers,

  test_non_uniform_2d_basic,
  test_non_uniform_2d_atomics,
  test_non_uniform_2d_barriers,

  test_non_uniform_3d_basic,
  test_non_uniform_3d_atomics,
  test_non_uniform_3d_barriers,

  test_non_uniform_other_basic,
  test_non_uniform_other_atomics,
  test_non_uniform_other_barriers
};

const char    *basefn_names[] = {
  "non_uniform_1d_basic",
  "non_uniform_1d_atomics",
  "non_uniform_1d_barriers",

  "non_uniform_2d_basic",
  "non_uniform_2d_atomics",
  "non_uniform_2d_barriers",

  "non_uniform_3d_basic",
  "non_uniform_3d_atomics",
  "non_uniform_3d_barriers",

  "non_uniform_other_basic",
  "non_uniform_other_atomics",
  "non_uniform_other_barriers",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
  typedef std::vector<const char *> ArgsVector;
  ArgsVector programArgs;
  programArgs.assign(argv, argv+argc);

  int numFns = num_fns;
  basefn *baseFnList = basefn_list;
  const char **baseFnNames = basefn_names;

  for (ArgsVector::iterator it = programArgs.begin(); it!=programArgs.end();) {

    if(*it == std::string("-strict")) {
      TestNonUniformWorkGroup::enableStrictMode(true);
      it=programArgs.erase(it);
    } else {
      ++it;
    }
  }

  PrimeNumbers::generatePrimeNumbers(100000);

  return runTestHarness(static_cast<int>(programArgs.size()), &programArgs.front(), numFns, baseFnList, baseFnNames, false /* image support required */, false /* force no context creation */, 0 );
}




