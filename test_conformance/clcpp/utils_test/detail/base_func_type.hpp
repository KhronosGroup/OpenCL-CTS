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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_TEST_DETAIL_BASE_FUNC_TYPE_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_TEST_DETAIL_BASE_FUNC_TYPE_HPP

#include <random>
#include <limits>
#include <type_traits>
#include <algorithm>

#include <cmath>

#include "../../common.hpp"

#include "vec_helpers.hpp"

namespace detail
{

template<class OUT1>
struct base_func_type
{   
    virtual ~base_func_type() {};

    // Returns function name
    virtual std::string str() = 0;

    // Returns name of the test kernel for that function
    virtual std::string get_kernel_name()
    {
        std::string kn = this->str();
        replace_all(kn, "::", "_");
        return "test_" + kn;
    }

    // Returns required defines and pragmas.
    virtual std::string defs()
    {
        return "";
    }

    // Returns required OpenCL C++ headers.
    virtual std::string headers()
    {
        return "";
    }

    // Return true if OUT1 type in OpenCL kernel should be treated
    // as bool type; false otherwise.
    bool is_out_bool()
    {
        return false;
    }

    // Max ULP error, that is error should be raised when
    // if Ulp_Error(result, expected) > ulp()
    float ulp()
    {
        return 0.0f;
    }

    // Should we check ULP error when verifing if the result is
    // correct? 
    //
    // (This effects how are_equal() function works, 
    // it may not have effect if verify() method in derived
    // class does not use are_equal() function.)
    //
    // Only for FP numbers/vectors
    bool use_ulp()
    {
        return true;
    }

    // Max error. Error should be raised if
    // abs(result - expected) > delta(.., expected)
    //
    // Default value: 0.001 * expected
    //
    // (This effects how are_equal() function works, 
    // it may not have effect if verify() method in derived
    // class does not use are_equal() function.)
    //
    // Only for FP numbers/vectors
    template<class T>
    typename make_vector_type<cl_double, vector_size<T>::value>::type
    delta(const T& expected)
    {
        typedef 
            typename make_vector_type<cl_double, vector_size<T>::value>::type
            delta_vector_type;
        auto e = detail::make_value<delta_vector_type>(1e-3);
        return detail::multiply<delta_vector_type>(e, expected);
    }
};

} // detail namespace

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_DETAIL_BASE_FUNC_TYPE_HPP
