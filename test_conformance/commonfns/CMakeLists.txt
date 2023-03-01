set(MODULE_NAME COMMONFNS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_clamp.cpp
    test_degrees.cpp
    test_max.cpp
    test_maxf.cpp
    test_min.cpp
    test_minf.cpp
    test_mix.cpp
    test_radians.cpp
    test_step.cpp
    test_stepf.cpp
    test_smoothstep.cpp
    test_smoothstepf.cpp
    test_sign.cpp
    test_fmax.cpp
    test_fmin.cpp
    test_fmaxf.cpp
    test_fminf.cpp
    test_binary_fn.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
clamp
degrees
fmax
fmaxf
fmin
fminf
max
maxf
min
minf
mix
radians
step
stepf
smoothstep
smoothstepf
sign
)

include(../CMakeCommon.cmake)
