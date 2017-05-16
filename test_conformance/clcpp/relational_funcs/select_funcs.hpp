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
#ifndef TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_SELECT_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_SELECT_FUNCS_HPP

#include "common.hpp"

template <class IN1, cl_int N /* Vector size */>
struct select_func_select : public ternary_func<
                                    typename make_vector_type<IN1, N>::type, /* create IN1N type */
                                    typename make_vector_type<IN1, N>::type, /* create IN1N type */
                                    typename make_vector_type<cl_int, N>::type, /* create cl_intN type */
                                    typename make_vector_type<IN1, N>::type /* create IN1N type */
                                 >
{
    typedef typename make_vector_type<IN1, N>::type input1_type;
    typedef typename make_vector_type<IN1, N>::type input2_type;
    typedef typename make_vector_type<cl_int, N>::type input3_type;
    typedef typename make_vector_type<IN1, N>::type result_type;
   
    std::string str()
    {
        return "select";
    }
   
    std::string headers()
    {
        return "#include <opencl_relational>\n";
    }
   
    result_type operator()(const input1_type& x, const input2_type& y, const input3_type& z)
    {   
        typedef typename scalar_type<input1_type>::type SCALAR1;
        typedef typename scalar_type<input2_type>::type SCALAR2;
        typedef typename scalar_type<input3_type>::type SCALAR3;

        return perform_function<input1_type, input2_type, input3_type, result_type>(
            x, y, z,
            [](const SCALAR1& a, const SCALAR2& b, const SCALAR3& c)
            {
                    return (c != 0) ? b : a;
            }
        );
    }

    bool is_in3_bool()
    {
        return true;
    }
   
    std::vector<input3_type> in3_special_cases()
    {
        return { 
            detail::make_value<input3_type>(0),
            detail::make_value<input3_type>(1),
            detail::make_value<input3_type>(12),
            detail::make_value<input3_type>(-12)
        };
    }
};

template <class IN1, cl_int N /* Vector size */>
struct select_func_bitselect : public ternary_func<
                                    typename make_vector_type<IN1, N>::type, /* create IN1N type */
                                    typename make_vector_type<IN1, N>::type, /* create IN1N type */
                                    typename make_vector_type<IN1, N>::type, /* create cl_intN type */
                                    typename make_vector_type<IN1, N>::type /* create IN1N type */
                                 >
{
    typedef typename make_vector_type<IN1, N>::type input1_type;
    typedef typename make_vector_type<IN1, N>::type input2_type;
    typedef typename make_vector_type<IN1, N>::type input3_type;
    typedef typename make_vector_type<IN1, N>::type result_type;
   
    std::string str()
    {
        return "bitselect";
    }
   
    std::string headers()
    {
        return "#include <opencl_relational>\n";
    }
   
    result_type operator()(const input1_type& x, const input2_type& y, const input3_type& z)
    {  
        static_assert(
            std::is_integral<IN1>::value,
            "bitselect test is implemented only for integers."
        ); 
        static_assert(
            std::is_unsigned<IN1>::value,
            "IN1 type should be unsigned, bitwise operations on signed int may cause problems."
        );
        typedef typename scalar_type<input1_type>::type SCALAR1;
        typedef typename scalar_type<input2_type>::type SCALAR2;
        typedef typename scalar_type<input3_type>::type SCALAR3;

        return perform_function<input1_type, input2_type, input3_type, result_type>(
            x, y, z,
            [](const SCALAR1& a, const SCALAR2& b, const SCALAR3& c)
            {
                return (~c & a) | (c & b);
            }
        );
    }
};

AUTO_TEST_CASE(test_relational_select_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

// Tests for select(gentype a, gentype b, booln c) are not run in USE_OPENCLC_KERNELS 
// mode, because this functions in OpenCL C requires different reference functions on host
// compared to their equivalent in OpenCL C++.
// (In OpenCL C the result of select(), when gentype is vector type, is based on the most
// significant bits of c components)
#ifndef USE_OPENCLC_KERNELS
    // gentype select(gentype a, gentype b, booln c)
    TEST_TERNARY_FUNC_MACRO((select_func_select<cl_uint,  1>()))
    TEST_TERNARY_FUNC_MACRO((select_func_select<cl_float, 2>()))
    TEST_TERNARY_FUNC_MACRO((select_func_select<cl_short, 4>()))
    TEST_TERNARY_FUNC_MACRO((select_func_select<cl_uint,  8>()))
    TEST_TERNARY_FUNC_MACRO((select_func_select<cl_uint,  16>()))
#else
    log_info("WARNING:\n\tTests for select(gentype a, gentype b, booln c) are not run in USE_OPENCLC_KERNELS mode\n");
#endif

    // gentype bitselect(gentype a, gentype b, gentype c)
    TEST_TERNARY_FUNC_MACRO((select_func_bitselect<cl_uint, 1>()))
    TEST_TERNARY_FUNC_MACRO((select_func_bitselect<cl_ushort, 2>()))
    TEST_TERNARY_FUNC_MACRO((select_func_bitselect<cl_uchar, 4>()))
    TEST_TERNARY_FUNC_MACRO((select_func_bitselect<cl_ushort, 8>()))
    TEST_TERNARY_FUNC_MACRO((select_func_bitselect<cl_uint, 16>()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_SELECT_FUNCS_HPP
