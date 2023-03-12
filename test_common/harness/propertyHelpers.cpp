//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "propertyHelpers.h"
#include "errorHelpers.h"

#include <assert.h>

#include <algorithm>
#include <cinttypes>
#include <vector>

static bool findProperty(const std::vector<cl_properties>& props,
                         cl_properties prop, cl_properties& value)
{
    // This function assumes properties are valid:
    assert(props.size() == 0 || props.back() == 0);
    assert(props.size() == 0 || props.size() % 2 == 1);

    for (cl_uint i = 0; i < props.size(); i = i + 2)
    {
        cl_properties check_prop = props[i];

        if (check_prop == 0)
        {
            break;
        }

        if (check_prop == prop)
        {
            value = props[i + 1];
            return true;
        }
    }

    return false;
}

int compareProperties(const std::vector<cl_properties>& queried,
                      const std::vector<cl_properties>& check)
{
    if (queried.size() != 0)
    {
        if (queried.back() != 0)
        {
            log_error("ERROR: queried properties do not end with 0!\n");
            return TEST_FAIL;
        }
        if (queried.size() % 2 != 1)
        {
            log_error("ERROR: queried properties does not consist of "
                      "property-value pairs!\n");
            return TEST_FAIL;
        }
    }
    if (check.size() != 0)
    {
        if (check.back() != 0)
        {
            log_error("ERROR: check properties do not end with 0!\n");
            return TEST_FAIL;
        }
        if (check.size() % 2 != 1)
        {
            log_error("ERROR: check properties does not consist of "
                      "property-value pairs!\n");
            return TEST_FAIL;
        }
    }

    if (queried != check)
    {
        for (cl_uint i = 0; i < check.size(); i = i + 2)
        {
            cl_properties check_prop = check[i];

            if (check_prop == 0)
            {
                break;
            }

            cl_properties check_value = check[i + 1];
            cl_properties queried_value = 0;

            bool found = findProperty(queried, check_prop, queried_value);

            if (!found)
            {
                log_error("ERROR: expected property 0x%" PRIx64 " not found!\n",
                          check_prop);
                return TEST_FAIL;
            }
            else if (check_value != queried_value)
            {
                log_error("ERROR: mis-matched value for property 0x%" PRIx64
                          ": wanted "
                          "0x%" PRIx64 ", got 0x%" PRIx64 "\n",
                          check_prop, check_value, queried_value);
                return TEST_FAIL;
            }
        }

        if (queried.size() > check.size())
        {
            log_error("ERROR: all properties found but there are extra "
                      "properties: expected %zu, got %zu.\n",
                      check.size(), queried.size());
            return TEST_FAIL;
        }

        log_error("ERROR: properties were returned in the wrong order.\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}
