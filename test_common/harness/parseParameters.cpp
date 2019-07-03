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
#include "parseParameters.h"
#include "errorHelpers.h"
#include <string.h>

bool is_power_of_two(int number)
{
    return number && !(number & (number - 1));
}

extern void parseWimpyReductionFactor(const char *&arg, int &wimpyReductionFactor)
{
    const char *arg_temp = strchr(&arg[1], ']');
    if (arg_temp != 0)
    {
        int new_factor = atoi(&arg[1]);
        arg = arg_temp; // Advance until ']'
        if (is_power_of_two(new_factor))
        {
            log_info("\n Wimpy reduction factor changed from %d to %d \n", wimpyReductionFactor, new_factor);
            wimpyReductionFactor = new_factor;
        }
        else
        {
            log_info("\n WARNING: Incorrect wimpy reduction factor %d, must be power of 2. The default value will be used.\n", new_factor);
        }
    }
}
