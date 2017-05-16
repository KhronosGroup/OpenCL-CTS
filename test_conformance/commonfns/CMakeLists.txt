set(MODULE_NAME COMMONFNS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_clamp.c
    test_degrees.c
    test_max.c
    test_maxf.c
    test_min.c
    test_minf.c
    test_mix.c
    test_radians.c
    test_step.c
    test_stepf.c
    test_smoothstep.c
    test_smoothstepf.c
    test_sign.c
    test_fmax.c
    test_fmin.c
    test_fmaxf.c
    test_fminf.c
    test_binary_fn.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
)

include(../CMakeCommon.txt)
