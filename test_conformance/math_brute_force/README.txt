Copyright:	(c) 2009-2013 by Apple Inc. All Rights Reserved.

math_brute_force test                                                   Feb 24, 2009
=====================

Usage:

        Please run the executable with --help for usage information.
	


System Requirements:

        This test requires support for correctly rounded single and double precision arithmetic.
The current version also requires a reasonably accurate operating system math library to 
be present. The OpenCL implementation must be able to compile kernels online. The test assumes
that the host system stores its floating point data according to the IEEE-754 binary single and 
double precision floating point formats. 


Test Completion Time:

        This test takes a while. Modern desktop systems can usually finish it in 1-3
days. Engineers doing OpenCL math library software development may find wimpy mode (-w)
a useful screen to quickly look for problems in a new implementation, before committing
to a lengthy test run. Likewise, it is possible to run just a range of tests, or specific
tests. See Usage above.


Test Design:

        This test is designed to do a somewhat exhaustive examination of the single
and double precision math library functions in OpenCL, for all vector lengths. Math 
library functions are compared against results from a higher precision reference 
function to determine correctness. All possible inputs are  examined for unary 
single precision functions.  Other functions are tested against a table of difficult 
values, followed by a few billion random values. If an error is found in a function,
the test for that function terminates early, reports an error, and moves on to the 
next test, if any.

This test doesn't test the native_<funcname> functions, for which any result is conformant.

For the OpenCL 1.0 time frame, the reference library shall be the operating system 
math library, as modified by the test itself to conform to the OpenCL specification. 
That will help ensure that all devices on a particular operating system are returning 
similar results.  Going forward to future OpenCL releases, it is planned to gradually 
introduce a reference math library directly into the test, so as to reduce inter-
platform variance between OpenCL implementations. 

Generally speaking, this test will consider a result correct if it is one of the following:

        1) bitwise identical to the output of the reference function, 
                rounded to the appropriate precision

        2) within the allowed ulp error tolerance of the infinitely precise
                result (as estimated by the reference function)

        3) If the reference result is a NaN, then any NaN is deemed correct.

        4) if the devices is running in FTZ mode, then the result is also correct
                if the infinitely precise result (as estimated by the reference
                function) is subnormal, and the returned result is a zero
        
        5) if the devices is running in FTZ mode, then we also calculate the 
                estimate of the infinitely precise result with the reference function 
                with subnormal inputs flushed to +- zero.  If any of those results 
                are within the error tolerance of the returned result, then it is 
                deemed correct

        6) half_func functions may flush per 4&5 above, even if the device is not
                in FTZ mode.

        7) Functions are allowed to prematurely overflow to infinity, so long as 
                the estimated infinitely precise result is within the stated ulp 
                error limit of the maximum finite representable value of appropriate 
                sign

        8) Functions are allowed to prematurely underflow (and if in FTZ mode, 
                have behavior covered by 4&5 above), so long as the estimated
                infinitely precise result is within the stated ulp error limit
                of the minimum normal representable value of appropriate sign

        9) Some functions have limited range. Results of inputs outside that range
                are considered correct, so long as a result is returned.

        10) Some functions have infinite error bounds. Results of these function
                are considered correct, so long as a result is returned.

        11) The test currently does not discriminate based on the sign of zero
                We anticipate a later test will.

        12) The test currently does not check to make sure that edge cases called 
                out in the standard (e.g. pow(1.0, any) = 1.0) are exactly correct.
                We anticipate a later test will.

        13) The test doesn't check IEEE flags or exceptions. See section 7.3 of the 
                OpenCL standard.



Performance Measurement:

        There is also some optional timing code available, currently turned off by default. 
These may be useful for tracking internal performance regressions, but is not required to 
be part of the conformance submission.


If the test is believed to be in error:

The above correctness heuristics shall not be construed to be an alternative to the correctness 
criteria established by the OpenCL standard. An implementation shall be judged correct
or not on appeal based on whether it is within prescribed error bounds of the infinitely 
precise result. (The ulp is defined in section 7.4 of the spec.) If the input value corresponds
to an edge case listed in OpenCL specification sections covering edge case behavior, or 
similar sections in the C99 TC2 standard (section F.9 and G.6), the the function shall return
exactly that result, and the sign of a zero result shall be correct. In the event that the test 
is found to be faulty, resulting in a spurious failure result, the committee shall make a reasonable 
attempt to fix the test. If no practical and timely remedy can be found, then the implementation 
shall be granted a waiver.


Guidelines for reference function error tolerances:

        Errors are measured in ulps, and stored in a single precision representation. So as
to avoid introducing error into the error measurement due to error in the reference function
itself, the reference function should attempt to deliver 24 bits more precision than the test 
function return type. (All functions are currently either required to be correctly rounded or 
may have >= 1 ulp of error. This places the 1's bit at the LSB of the result, with 23 bits of 
sub-ulp accuracy. One more bit is required to avoid accrual of extra error due to round-to-
nearest behavior. If we start to require sub-ulp precision, then the accuracy requirements 
for reference functions increase.) Therefore reference functions for single precision should 
have 24+24=48 bits of accuracy, and reference functions for double precision should ideally 
have 53+24 = 77 bits of accuracy. 

A double precision system math library function should be sufficient to safely verify a single 
precision OpenCL math library function.  A long double precision math library function may or 
may not be sufficient to verify a double precision OpenCL math library function, depending on 
the precision of the long double type. A later version of these tests is expected to replace 
long double with a head+tail double double representation that can represent sufficient precision,
on all platforms that support double. 


Revision history:

 Feb 24, 2009                IRO        Created README
                                        Added some reference functions so the test will run on Windows.

