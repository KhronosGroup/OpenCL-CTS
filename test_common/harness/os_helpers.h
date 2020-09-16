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
#ifndef __os_helpers_h__
#define __os_helpers_h__

#include "compat.h"

// -------------------------------------------------------------------------------------------------
// C++ interface.
// -------------------------------------------------------------------------------------------------

#ifdef __cplusplus

    #include <string>

    std::string err_msg( int err );
    std::string dir_sep();
    std::string exe_path();
    std::string exe_dir();

#endif // __cplusplus

// -------------------------------------------------------------------------------------------------
// C interface.
// -------------------------------------------------------------------------------------------------

char * get_err_msg( int err );  // Returns system error message. Subject to free.
char * get_dir_sep();           // Returns dir separator. Subject to free.
char * get_exe_path();          // Returns path of current executable. Subject to free.
char * get_exe_dir();           // Returns dir of current executable. Subject to free.

#endif // __os_helpers_h__
