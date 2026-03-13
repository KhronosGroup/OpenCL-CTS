// Copyright (c) 2026 The Khronos Group Inc.
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
#include "harness/errorHelpers.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
using PluginHandle = HMODULE;
#define LoadPlugin() ::LoadLibraryA("LoadUnloadPlugin.dll")
#define ClosePlugin(_handle) ::FreeLibrary(_handle)
#define GetFunctionAddress(_handle, _name) ::GetProcAddress(_handle, _name)
#else
using PluginHandle = void *;
#define LoadPlugin() ::dlopen("./libLoadUnloadPlugin.so", RTLD_NOW)
#define ClosePlugin(_handle) ::dlclose(_handle)
#define GetFunctionAddress(_handle, _name) ::dlsym(_handle, _name)
#endif

typedef int TestFunction_t(int argc, const char *argv[]);
typedef TestFunction_t *TestFunction_ptr;

int main(int argc, const char *argv[])
{
    constexpr int iterations = 5;
    int result = EXIT_SUCCESS;
    for (int i = 0; i < iterations && result == EXIT_SUCCESS; i++)
    {
        log_info("Iteration %d of %d...\n", i + 1, iterations);

        log_info("Loading plugin...\n");
        PluginHandle plugin = LoadPlugin();
        if (!plugin)
        {
            log_error("Failed to load plugin!\n");
            return EXIT_FAILURE;
        }

        log_info("Getting test pointer...\n");
        TestFunction_ptr testFunction = reinterpret_cast<TestFunction_ptr>(
            GetFunctionAddress(plugin, "do_test"));
        if (!testFunction)
        {
            log_error("Failed to get test function address!\n");
            ClosePlugin(plugin);
            return EXIT_FAILURE;
        }

        log_info("Running test...\n");
        result = testFunction(argc, argv);
        if (result != EXIT_SUCCESS)
        {
            log_error("Test function failed!\n");
        }

        log_info("Closing plugin...\n");
        ClosePlugin(plugin);
    }

    return result;
}
