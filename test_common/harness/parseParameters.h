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
#ifndef _parseParameters_h
#define _parseParameters_h

#include "compat.h"
#include <string>
#include <vector>

enum CompilationMode
{
    kOnline = 0,
    kBinary,
    kSpir_v
};

enum CompilationCacheMode
{
    kCacheModeCompileIfAbsent = 0,
    kCacheModeForceRead,
    kCacheModeOverwrite,
    kCacheModeDumpCl
};

extern CompilationMode gCompilationMode;
extern CompilationCacheMode gCompilationCacheMode;
extern std::string gCompilationCachePath;
extern std::string gCompilationProgram;
extern bool gDisableSPIRVValidation;
extern std::string gSPIRVValidator;
extern bool gListTests;
extern bool gWimpyMode;

extern int
parseCommonParamAndGetRemovedArgs(int argc, const char *argv[],
                                  std::vector<std::string> &removed_args,
                                  bool &help);
extern int parseCommonParam(int argc, const char *argv[]);

extern void parseWimpyReductionFactor(const char *&arg,
                                      int &wimpyReductionFactor);

#endif // _parseParameters_h
