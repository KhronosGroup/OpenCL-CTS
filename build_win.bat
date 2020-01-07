@ECHO off
setlocal ENABLEDELAYEDEXPANSION

IF DEFINED ProgramFiles(x86) SET ProgFilesDir=%ProgramFiles(x86)%
IF NOT DEFINED ProgFilesDir SET ProgFilesDir=%ProgramFiles%

rem -------------------------------- Update these to match what's on your PC ------------------------------------------------

SET VCPATH="%ProgFilesDir%\Microsoft Visual Studio 14.0\Common7\IDE\devenv.com"

SET PATH=%CMAKEPATH%;%PATH%

rem -------------------------------------------------------------------------------------------------------------------------

setlocal ENABLEDELAYEDEXPANSION

call "%VS140COMNTOOLS%\vsvars32.bat"

mkdir build_win
pushd build_win
IF NOT EXIST CLConform.sln (
   echo "Solution file not found, running Cmake"
   cmake -G "Visual Studio 14 2015 Win64" ..\.  -DKHRONOS_OFFLINE_COMPILER=<TO_SET> -DCL_LIBCLCXX_DIR=<TO_SET> -DCL_INCLUDE_DIR=<TO_SET> -DCL_LIB_DIR=<TO_SET> -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=. -DOPENCL_LIBRARIES=OpenCL
) else (
   echo "Solution file found CLConform.sln "
)

echo Building CLConform.sln...
%VCPATH% CLConform.sln /build


GOTO:EOF
