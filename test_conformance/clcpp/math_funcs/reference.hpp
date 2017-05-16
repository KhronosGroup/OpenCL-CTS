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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_REFERENCE_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_REFERENCE_HPP

#include <type_traits>
#include <cmath>
#include <limits>

#include "../common.hpp"

namespace reference
{
    // Reference functions for OpenCL comparison functions that
    // are not already defined in STL.
    cl_float maxmag(const cl_float& x, const cl_float& y)
    {
        if((std::abs)(x) > (std::abs)(y))
        {
            return x;
        }
        else if((std::abs)(y) > (std::abs)(x))
        {
            return y;
        }
        return (std::fmax)(x, y);
    }

    cl_float minmag(const cl_float& x, const cl_float& y)
    {
        if((std::abs)(x) < (std::abs)(y))
        {
            return x;
        }
        else if((std::abs)(y) < (std::abs)(x))
        {
            return y;
        }
        return (std::fmin)(x, y);
    }

    // Reference functions for OpenCL exp functions that
    // are not already defined in STL.
    cl_double exp10(const cl_double& x)
    {   
        // 10^x = exp2( x * log2(10) )
        auto log2_10 = (std::log2)(static_cast<long double>(10.0));
        cl_double x_log2_10 = static_cast<cl_double>(x * log2_10);
        return (std::exp2)(x_log2_10);
    }

    // Reference functions for OpenCL floating point functions that
    // are not already defined in STL.
    cl_double2 fract(cl_double x)
    {
        // Copied from math_brute_force/reference_math.c
        cl_double2 r;
        if((std::isnan)(x))
        {
            r.s[0] = std::numeric_limits<cl_double>::quiet_NaN();
            r.s[1] = std::numeric_limits<cl_double>::quiet_NaN();
            return r;
        }

        r.s[0] = (std::modf)(x, &(r.s[1]));
        if(r.s[0] < 0.0 )
        {
            r.s[0] = 1.0f + r.s[0];
            r.s[1] -= 1.0f;
            if( r.s[0] == 1.0f )
                r.s[0] = HEX_FLT(+, 1, fffffe, -, 1);
        }
        return r;
    }

    cl_double2 remquo(cl_double x, cl_double y)
    {
        cl_double2 r;
        // remquo return the same value that is returned by the
        // remainder function
        r.s[0] = (std::remainder)(x,y);
        // calulcate quo
        cl_double x_y = (x - r.s[0]) / y;
        cl_uint quo = (std::abs)(x_y);
        r.s[1] = quo & 0x0000007fU;
        if(x_y < 0.0)
            r.s[1] = -r.s[1];

        // fix edge cases
        if(!(std::isnan)(x) && y == 0.0)
        {
            r.s[1] = 0;
        }
        else if((std::isnan)(x) && (std::isnan)(y))
        {
            r.s[1] = 0;
        }
        return r;
    }

    // Reference functions for OpenCL half_math:: functions that
    // are not already defined in STL.
    cl_double divide(cl_double x, cl_double y)
    {
        return x / y;
    }

    cl_double recip(cl_double x)
    {
        return 1.0 / x;
    }

    // Reference functions for OpenCL other functions that
    // are not already defined in STL.
    cl_double mad(cl_double x, cl_double y, cl_double z)
    {
        return (x * y) + z;
    }

    // Reference functions for OpenCL power functions that
    // are not already defined in STL.
    cl_double rsqrt(const cl_double& x)
    {
        return cl_double(1.0) / ((std::sqrt)(x));
    }

    cl_double powr(const cl_double& x, const cl_double& y)
    {
        //powr(x, y) returns NaN for x < 0.
        if( x < 0.0 )
            return std::numeric_limits<cl_double>::quiet_NaN();

        //powr ( x, NaN ) returns the NaN for x >= 0.
        //powr ( NaN, y ) returns the NaN.
        if((std::isnan)(x) || (std::isnan)(y) )
            return std::numeric_limits<cl_double>::quiet_NaN();

        if( x == 1.0 )
        {
            //powr ( +1, +-inf ) returns NaN.
            if((std::abs)(y) == INFINITY )
                return std::numeric_limits<cl_double>::quiet_NaN();

            //powr ( +1, y ) is 1 for finite y. (NaN handled above)
            return 1.0;
        }

        if( y == 0.0 )
        {
            //powr ( +inf, +-0 ) returns NaN.
            //powr ( +-0, +-0 ) returns NaN.
            if( x == 0.0 || x == std::numeric_limits<cl_double>::infinity())
                return std::numeric_limits<cl_double>::quiet_NaN();

            //powr ( x, +-0 ) is 1 for finite x > 0.  (x <= 0, NaN, INF already handled above)
            return 1.0;
        }

        if( x == 0.0 )
        {
            //powr ( +-0, -inf) is +inf.
            //powr ( +-0, y ) is +inf for finite y < 0.
            if( y < 0.0 )
                return std::numeric_limits<cl_double>::infinity();

            //powr ( +-0, y ) is +0 for y > 0.    (NaN, y==0 handled above)
            return 0.0;
        }

        // x = +inf
        if( (std::isinf)(x) )
        {
            if( y < 0 )
                return 0;
            return std::numeric_limits<cl_double>::infinity();
        }

        double fabsx = (std::abs)(x);
        double fabsy = (std::abs)(y);

        //y = +-inf cases
        if( (std::isinf)(fabsy) )
        {
            if( y < 0.0 )
            {
                if( fabsx < 1.0 )
                    return std::numeric_limits<cl_double>::infinity();
                return 0;
            }
            if( fabsx < 1.0 )
                return 0.0;
            return std::numeric_limits<cl_double>::infinity();
        }        
        return (std::pow)(x, y);
    }

    cl_double rootn(const cl_double& x, const cl_int n)
    {
        //rootn (x, 0) returns a NaN.
        if(n == 0)
            return std::numeric_limits<cl_double>::quiet_NaN();

        //rootn ( x, n )  returns a NaN for x < 0 and n is even.
        if(x < 0 && 0 == (n & 1))
            return std::numeric_limits<cl_double>::quiet_NaN();

        if(x == 0.0)
        {
            if(n > 0)
            {
                //rootn ( +-0,  n ) is +0 for even n > 0.
                if(0 == (n & 1))
                {
                    return cl_double(0.0);
                }
                //rootn ( +-0,  n ) is +-0 for odd n > 0.
                else
                {
                    return x;
                }
            }
            else
            {
                //rootn ( +-0,  n ) is +inf for even n < 0.
                if(0 == ((-n) & 1))
                {
                    return std::numeric_limits<cl_double>::infinity();
                }
                //rootn ( +-0,  n ) is +-inf for odd n < 0.
                else
                {
                    return (std::copysign)(
                        std::numeric_limits<cl_double>::infinity(), x
                    );
                }   
            }
        }

        cl_double r = (std::abs)(x);
        r = (std::exp2)((std::log2)(r) / static_cast<cl_double>(n));
        return (std::copysign)(r, x);
    }

    // Reference functions for OpenCL trigonometric functions that
    // are not already defined in STL.
    cl_double acospi(cl_double x)
    {
        return (std::acos)(x) / CL_M_PI;
    }

    cl_double asinpi(cl_double x)
    {
        return (std::asin)(x) / CL_M_PI;
    }

    cl_double atanpi(cl_double x)
    {
        return (std::atan)(x) / CL_M_PI;
    }

    cl_double cospi(cl_double x)
    {
        return (std::cos)(x * CL_M_PI);
    }

    cl_double sinpi(cl_double x)
    {
        return (std::sin)(x * CL_M_PI);
    }

    cl_double tanpi(cl_double x)
    {
        return (std::tan)(x * CL_M_PI);
    }

    cl_double atan2(cl_double x, cl_double y)
    {
    #if defined(WIN32) || defined(_WIN32) 
        // Fix edge cases for Windows
        if ((std::isinf)(x) && (std::isinf)(y)) {
            cl_double retval = (y > 0) ? CL_M_PI_4 : 3.f * CL_M_PI_4;
            return (x > 0) ? retval : -retval;
        }
    #endif // defined(WIN32) || defined(_WIN32) 
        return (std::atan2)(x, y);
    }

    cl_double atan2pi(cl_double x, cl_double y)
    {
        return ::reference::atan2(x, y) / CL_M_PI;
    }

    cl_double2 sincos(cl_double x)
    {
        cl_double2 r;
        r.s[0] = (std::sin)(x);
        r.s[1] = (std::cos)(x);
        return r;
    }
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_REFERENCE_HPP
