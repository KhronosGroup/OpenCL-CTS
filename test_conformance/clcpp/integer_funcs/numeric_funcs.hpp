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
#ifndef TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_NUMERIC_HPP
#define TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_NUMERIC_HPP

#include "common.hpp"
#include <type_traits>

template<class IN1, class OUT1>
struct int_func_abs : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "abs";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x)
    {
        static_assert(
            std::is_unsigned<OUT1>::value,
            "OUT1 type must be unsigned"
        );
        if(x < IN1(0))
            return static_cast<OUT1>(-x);
        return static_cast<OUT1>(x);
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_abs_diff : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "abs_diff";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value,
            "IN1 must be IN2"
        );
        static_assert(
            std::is_unsigned<OUT1>::value,
            "OUT1 type must be unsigned"
        );
        if(x < y)
            return static_cast<OUT1>(y-x);
        return static_cast<OUT1>(x-y);
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_add_sat : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "add_sat";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value,
            "IN1 must be IN2"
        );
        static_assert(
            std::is_same<OUT1, IN2>::value,
            "OUT1 must be IN2"
        );
        // sat unsigned integers
        if(std::is_unsigned<OUT1>::value)
        {
            OUT1 z = x + y;
            if(z < x || z < y)
                return (std::numeric_limits<OUT1>::max)();
            return z;
        }
        // sat signed integers
        OUT1 z = x + y;
        if(y > 0)
        {
            if(z < x)
                return (std::numeric_limits<OUT1>::max)();
        }
        else
        {
            if(z > x)
                return (std::numeric_limits<OUT1>::min)();
        }
        return z;
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_hadd : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "hadd";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value,
            "IN1 must be IN2"
        );
        static_assert(
            std::is_same<OUT1, IN2>::value,
            "OUT1 must be IN2"
        );
        return (x >> OUT1(1)) + (y >> OUT1(1)) + (x & y & OUT1(1));
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_rhadd : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "rhadd";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value,
            "IN1 must be IN2"
        );
        static_assert(
            std::is_same<OUT1, IN2>::value,
            "OUT1 must be IN2"
        );
        return (x >> OUT1(1)) + (y >> OUT1(1)) + ((x | y) & OUT1(1));
    }
};

// clamp for scalars
template<class IN1, class IN2, class IN3, class OUT1, class Enable = void>
struct int_func_clamp : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "clamp";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& minval, const IN3& maxval)
    {
        static_assert(
            std::is_same<IN2, IN3>::value,
            "IN3 must be IN2"
        );
        static_assert(
            std::is_same<OUT1, IN1>::value,
            "OUT1 must be IN1"
        );
        return (std::min)((std::max)(x, minval), maxval);
    }

    IN2 min2()
    {
        return (std::numeric_limits<IN2>::min)();
    }

    IN2 max2()
    {
        return (std::numeric_limits<IN2>::max)() / IN2(2);
    }

    IN3 min3()
    {
        return IN3(1) + ((std::numeric_limits<IN3>::max)() / IN3(2));
    }

    IN3 max3()
    {
        return (std::numeric_limits<IN3>::max)();
    }
};

// gentype clamp(gentype x, scalar minval, scalar maxval);
template<class IN1, class IN2, class IN3, class OUT1>
struct int_func_clamp<IN1, IN2, IN3, OUT1, typename std::enable_if<is_vector_type<OUT1>::value>::type> : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "clamp";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& minval, const IN3& maxval)
    {
        static_assert(
            std::is_same<IN2, IN3>::value,
            "IN3 must be IN2"
        );
        static_assert(
            !is_vector_type<IN2>::value && !is_vector_type<IN3>::value,
            "IN3 and IN2 must be scalar"
        );
        static_assert(
            std::is_same<OUT1, IN1>::value,
            "OUT1 must be IN1"
        );
        OUT1 result;
        for(size_t i = 0; i < vector_size<OUT1>::value; i++)
        {
            result.s[i] = (std::min)((std::max)(x.s[i], minval), maxval);
        }
        return result;
    }

    IN1 min1()
    {
        typedef typename scalar_type<IN1>::type SCALAR1;
        IN1 min1;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            min1.s[i] = (std::numeric_limits<SCALAR1>::min)();
        }
        return min1;
    }

    IN1 max1()
    {
        typedef typename scalar_type<IN1>::type SCALAR1;
        IN1 max1;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            max1.s[i] = (std::numeric_limits<SCALAR1>::max)();
        }
        return max1;
    }

    IN2 min2()
    {
        return (std::numeric_limits<IN2>::min)();
    }

    IN2 max2()
    {
        return (std::numeric_limits<IN2>::max)() / IN2(2);
    }

    IN3 min3()
    {
        return IN3(1) + ((std::numeric_limits<IN3>::max)() / IN3(2));
    }

    IN3 max3()
    {
        return (std::numeric_limits<IN3>::max)();
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_mul_hi : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "mul_hi";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, OUT1>::value,
            "Types must be the same"
        );
        static_assert(
            !std::is_same<IN1, cl_long>::value && !std::is_same<IN1, cl_ulong>::value,
            "Operation unimplemented for 64-bit scalars"
        );  
        cl_long xl = static_cast<cl_long>(x);
        cl_long yl = static_cast<cl_long>(y);
        return static_cast<OUT1>((xl * yl) >> (8 * sizeof(OUT1)));
    }
};

template<class IN1, class IN2, class IN3, class OUT1>
struct int_func_mad_hi : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "mad_hi";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y, const IN3& z)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, IN3>::value
                && std::is_same<IN3, OUT1>::value,
            "Types must be the same"
        );   
        return int_func_mul_hi<IN1, IN2, OUT1>()(x, y) + z;
    }
};

// This test is implemented only for unsigned integers
template<class IN1, class IN2, class IN3, class OUT1>
struct int_func_mad_sat : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "mad_sat";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y, const IN3& z)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, IN3>::value
                && std::is_same<IN3, OUT1>::value,
            "Types must be the same"
        );
        static_assert(
            std::is_unsigned<OUT1>::value,
            "Test operation is not implemented for signed integers"
        );  
        // mad_sat unsigned integers
        OUT1 w1 = (x * y);
        if (x != 0 && w1 / x != y)
            return (std::numeric_limits<OUT1>::max)();
        OUT1 w2 = w1 + z;
        if(w2 < w1)
            return (std::numeric_limits<OUT1>::max)();
        return w2;
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_sub_sat : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "sub_sat";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value && std::is_same<IN2, OUT1>::value,
            "IN1, IN2 and OUT1 must be the same types"
        );
        // sat unsigned integers
        if(std::is_unsigned<OUT1>::value)
        {
            OUT1 z = x - y;
            if(x < y)
                return (std::numeric_limits<OUT1>::min)();
            return z;
        }
        // sat signed integers
        OUT1 z = x - y;
        if(y < 0)
        {
            if(z < x)
                return (std::numeric_limits<OUT1>::max)();
        }
        else
        {
            if(z > x)
                return (std::numeric_limits<OUT1>::min)();
        }
        return z;
    }
};

template<class IN1, class IN2, class OUT1, class Enable = void>
struct int_func_max : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "max";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value && std::is_same<IN2, OUT1>::value,
            "IN1, IN2 and OUT1 must be the same types"
        );
        return (std::max)(x, y);
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_max<IN1, IN2, OUT1, typename std::enable_if<is_vector_type<OUT1>::value>::type> : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "max";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    IN1 min1()
    {
        typedef typename scalar_type<IN1>::type SCALAR1;
        IN1 min1;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            min1.s[i] = (std::numeric_limits<SCALAR1>::min)();
        }
        return min1;
    }

    IN1 max1()
    {
        typedef typename scalar_type<IN1>::type SCALAR1;
        IN1 max1;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            max1.s[i] = (std::numeric_limits<SCALAR1>::max)();
        }
        return max1;
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, OUT1>::value,
            "IN1 and OUT1 must be the same types"
        );
        static_assert(
            !is_vector_type<IN2>::value,
            "IN2 must be scalar"
        );
        static_assert(
            std::is_same<typename scalar_type<OUT1>::type, IN2>::value,
            "IN2 must match with OUT1 and IN1"
        );
        IN1 result = x;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            result.s[i] = (std::max)(x.s[i], y);
        }
        return result;
    }
};

template<class IN1, class IN2, class OUT1, class Enable = void>
struct int_func_min : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "min";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value && std::is_same<IN2, OUT1>::value,
            "IN1, IN2 and OUT1 must be the same types"
        );
        return (std::min)(x, y);
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_min<IN1, IN2, OUT1, typename std::enable_if<is_vector_type<OUT1>::value>::type> : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "min";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    IN1 min1()
    {
        typedef typename scalar_type<IN1>::type SCALAR1;
        IN1 min1;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            min1.s[i] = (std::numeric_limits<SCALAR1>::min)();
        }
        return min1;
    }

    IN1 max1()
    {
        typedef typename scalar_type<IN1>::type SCALAR1;
        IN1 max1;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            max1.s[i] = (std::numeric_limits<SCALAR1>::max)();
        }
        return max1;
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, OUT1>::value,
            "IN1 and OUT1 must be the same types"
        );
        static_assert(
            !is_vector_type<IN2>::value,
            "IN2 must be scalar"
        );
        static_assert(
            std::is_same<typename scalar_type<OUT1>::type, IN2>::value,
            "IN2 must match with OUT1 and IN1"
        );
        IN1 result = x;
        for(size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            result.s[i] = (std::min)(x.s[i], y);
        }
        return result;
    }
};

AUTO_TEST_CASE(test_int_numeric_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // ugentype abs(gentype x);
    TEST_UNARY_FUNC_MACRO((int_func_abs<cl_int, cl_uint>()))
    TEST_UNARY_FUNC_MACRO((int_func_abs<cl_uint, cl_uint>()))
    TEST_UNARY_FUNC_MACRO((int_func_abs<cl_long, cl_ulong>()))
    TEST_UNARY_FUNC_MACRO((int_func_abs<cl_ulong, cl_ulong>()))

    // ugentype abs_diff(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_abs_diff<cl_int, cl_int, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_abs_diff<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_abs_diff<cl_long, cl_long, cl_ulong>()))
    TEST_BINARY_FUNC_MACRO((int_func_abs_diff<cl_ulong, cl_ulong, cl_ulong>()))

    // gentype add_sat(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_add_sat<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_add_sat<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_add_sat<cl_long, cl_long, cl_long>()))
    TEST_BINARY_FUNC_MACRO((int_func_add_sat<cl_ulong, cl_ulong, cl_ulong>()))

    // gentype hadd(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_hadd<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_hadd<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_hadd<cl_long, cl_long, cl_long>()))
    TEST_BINARY_FUNC_MACRO((int_func_hadd<cl_ulong, cl_ulong, cl_ulong>()))

    // gentype rhadd(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_rhadd<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_rhadd<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_rhadd<cl_long, cl_long, cl_long>()))
    TEST_BINARY_FUNC_MACRO((int_func_rhadd<cl_ulong, cl_ulong, cl_ulong>()))

    // gentype clamp(gentype x, gentype minval, gentype maxval);
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_int, cl_int, cl_int, cl_int>()))
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_uint, cl_uint, cl_uint, cl_uint>()))
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_long, cl_long, cl_long, cl_long>()))
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_ulong, cl_ulong, cl_ulong, cl_ulong>()))

    // gentype clamp(gentype x, scalar minval, scalar maxval);
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_int2, cl_int, cl_int, cl_int2>()))
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_uint4, cl_uint, cl_uint, cl_uint4>()))
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_long8, cl_long, cl_long, cl_long8>()))
    TEST_TERNARY_FUNC_MACRO((int_func_clamp<cl_ulong16, cl_ulong, cl_ulong, cl_ulong16>()))

    // gentype mad_hi(gentype a, gentype b, gentype c);
    TEST_TERNARY_FUNC_MACRO((int_func_mad_hi<cl_short, cl_short, cl_short, cl_short>()))
    TEST_TERNARY_FUNC_MACRO((int_func_mad_hi<cl_ushort, cl_ushort, cl_ushort, cl_ushort>()))
    TEST_TERNARY_FUNC_MACRO((int_func_mad_hi<cl_int, cl_int, cl_int, cl_int>()))
    TEST_TERNARY_FUNC_MACRO((int_func_mad_hi<cl_uint, cl_uint, cl_uint, cl_uint>()))

    // gentype mad_sat(gentype a, gentype b, gentype c);
    TEST_TERNARY_FUNC_MACRO((int_func_mad_sat<cl_ushort, cl_ushort, cl_ushort, cl_ushort>()))
    TEST_TERNARY_FUNC_MACRO((int_func_mad_sat<cl_uint, cl_uint, cl_uint, cl_uint>()))
    TEST_TERNARY_FUNC_MACRO((int_func_mad_sat<cl_ulong, cl_ulong, cl_ulong, cl_ulong>()))

    // gentype max(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_long, cl_long, cl_long>()))
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_ulong, cl_ulong, cl_ulong>()))

    // gentype max(gentype x, scalar y);
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_int2, cl_int, cl_int2>()))
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_uint4, cl_uint, cl_uint4>()))
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_long8, cl_long, cl_long8>()))
    TEST_BINARY_FUNC_MACRO((int_func_max<cl_ulong16, cl_ulong, cl_ulong16>()))

    // gentype min(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_long, cl_long, cl_long>()))
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_ulong, cl_ulong, cl_ulong>()))

    // gentype min(gentype x, scalar y);
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_int2, cl_int, cl_int2>()))
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_uint4, cl_uint, cl_uint4>()))
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_long8, cl_long, cl_long8>()))
    TEST_BINARY_FUNC_MACRO((int_func_min<cl_ulong16, cl_ulong, cl_ulong16>()))

    // gentype mul_hi(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_mul_hi<cl_short, cl_short, cl_short>()))
    TEST_BINARY_FUNC_MACRO((int_func_mul_hi<cl_ushort, cl_ushort, cl_ushort>())) 
    TEST_BINARY_FUNC_MACRO((int_func_mul_hi<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_mul_hi<cl_uint, cl_uint, cl_uint>()))

    // gentype sub_sat(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((int_func_sub_sat<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_sub_sat<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_sub_sat<cl_long, cl_long, cl_long>()))
    TEST_BINARY_FUNC_MACRO((int_func_sub_sat<cl_ulong, cl_ulong, cl_ulong>()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_NUMERIC_HPP
