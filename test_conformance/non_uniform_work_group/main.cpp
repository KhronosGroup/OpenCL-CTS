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
#include "harness/testHarness.h"
#include "TestNonUniformWorkGroup.h"

test_status InitCL(cl_device_id device) {
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);
    if (version < expected_min_version)
    {
        version_expected_info("Test", "OpenCL",
                              expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }
    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
  typedef std::vector<const char *> ArgsVector;
  ArgsVector programArgs;
  programArgs.assign(argv, argv+argc);

  for (ArgsVector::iterator it = programArgs.begin(); it!=programArgs.end();) {

    if(*it == std::string("-strict")) {
      TestNonUniformWorkGroup::enableStrictMode(true);
      it=programArgs.erase(it);
    } else {
      ++it;
    }
  }

  PrimeNumbers::generatePrimeNumbers(100000);

  return runTestHarnessWithCheck(
      static_cast<int>(programArgs.size()), &programArgs.front(),
      test_registry::getInstance().num_tests(),
      test_registry::getInstance().definitions(), false, false, InitCL);
}

