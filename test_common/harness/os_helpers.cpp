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
#include "os_helpers.h"
#include "errorHelpers.h"

// =================================================================================================
// C++ interface.
// =================================================================================================

#include <cerrno> // errno, error constants
#include <climits> // PATH_MAX
#include <cstdlib> // abort, _splitpath, _makepath
#include <cstring> // strdup, strerror_r
#include <sstream>

#include <vector>

#if defined(__ANDROID__)
#include <android/api-level.h>
#include "harness/mt19937.h"
#endif

#if !defined(_WIN32)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#endif


#define CHECK_PTR(ptr)                                                         \
    if ((ptr) == NULL)                                                         \
    {                                                                          \
        abort();                                                               \
    }

typedef std::vector<char> buffer_t;

#if !defined(PATH_MAX)
#define PATH_MAX 1000
#endif

int const _size = PATH_MAX + 1; // Initial buffer size for path.
int const _count = 8; // How many times we will try to double buffer size.

// -------------------------------------------------------------------------------------------------
// MacOS X
// -------------------------------------------------------------------------------------------------

#if defined(__APPLE__)


#include <mach-o/dyld.h> // _NSGetExecutablePath
#include <libgen.h> // dirname


static std::string
_err_msg(int err, // Error number (e. g. errno).
         int level // Nesting level, for avoiding infinite recursion.
)
{

    /*
        There are 3 incompatible versions of strerror_r:

            char * strerror_r( int, char *, size_t );  // GNU version
            int    strerror_r( int, char *, size_t );  // BSD version
            int    strerror_r( int, char *, size_t );  // XSI version

        BSD version returns error code, while XSI version returns 0 or -1 and
       sets errno.

    */

    // BSD version of strerror_r.
    buffer_t buffer(100);
    int count = _count;
    for (;;)
    {
        int rc = strerror_r(err, &buffer.front(), buffer.size());
        if (rc == EINVAL)
        {
            // Error code is not recognized, but anyway we got the message.
            return &buffer.front();
        }
        else if (rc == ERANGE)
        {
            // Buffer is not enough.
            if (count > 0)
            {
                // Enlarge the buffer.
                --count;
                buffer.resize(buffer.size() * 2);
            }
            else
            {
                std::stringstream ostr;
                ostr << "Error " << err << " "
                     << "(Getting error message failed: "
                     << "Buffer of " << buffer.size()
                     << " bytes is still too small"
                     << ")";
                return ostr.str();
            }; // if
        }
        else if (rc == 0)
        {
            // We got the message.
            return &buffer.front();
        }
        else
        {
            std::stringstream ostr;
            ostr << "Error " << err << " "
                 << "(Getting error message failed: "
                 << (level < 2 ? _err_msg(rc, level + 1) : "Oops") << ")";
            return ostr.str();
        }; // if
    }; // forever

} // _err_msg


std::string dir_sep() { return "/"; } // dir_sep


std::string exe_path()
{
    buffer_t path(_size);
    int count = _count;
    for (;;)
    {
        uint32_t size = path.size();
        int rc = _NSGetExecutablePath(&path.front(), &size);
        if (rc == 0)
        {
            break;
        }; // if
        if (count > 0)
        {
            --count;
            path.resize(size);
        }
        else
        {
            log_error("ERROR: Getting executable path failed: "
                      "_NSGetExecutablePath failed: Buffer of %lu bytes is "
                      "still too small\n",
                      (unsigned long)path.size());
            exit(2);
        }; // if
    }; // forever
    return &path.front();
} // exe_path


std::string exe_dir()
{
    std::string path = exe_path();
    // We cannot pass path.c_str() to `dirname' bacause `dirname' modifies its
    // argument.
    buffer_t buffer(path.c_str(),
                    path.c_str() + path.size() + 1); // Copy with trailing zero.
    return dirname(&buffer.front());
} // exe_dir


#endif // __APPLE__

// -------------------------------------------------------------------------------------------------
// Linux
// -------------------------------------------------------------------------------------------------

#if defined(__linux__)


#include <cerrno> // errno
#include <libgen.h> // dirname
#include <unistd.h> // readlink


static std::string _err_msg(int err, int level)
{

    /*
        There are 3 incompatible versions of strerror_r:

            char * strerror_r( int, char *, size_t );  // GNU version
            int    strerror_r( int, char *, size_t );  // BSD version
            int    strerror_r( int, char *, size_t );  // XSI version

        BSD version returns error code, while XSI version returns 0 or -1 and
       sets errno.

    */

#if (defined(__ANDROID__) && __ANDROID_API__ < 23)                             \
    || ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !_GNU_SOURCE)  \
    || (defined(_GNU_SOURCE) && !defined(__GLIBC__) && !defined(__USE_GNU))

// XSI version of strerror_r.
#warning Not tested!
    buffer_t buffer(200);
    int count = _count;
    for (;;)
    {
        int rc = strerror_r(err, &buffer.front(), buffer.size());
        if (rc == -1)
        {
            int _err = errno;
            if (_err == ERANGE)
            {
                if (count > 0)
                {
                    // Enlarge the buffer.
                    --count;
                    buffer.resize(buffer.size() * 2);
                }
                else
                {
                    std::stringstream ostr;
                    ostr << "Error " << err << " "
                         << "(Getting error message failed: "
                         << "Buffer of " << buffer.size()
                         << " bytes is still too small"
                         << ")";
                    return ostr.str();
                }; // if
            }
            else
            {
                std::stringstream ostr;
                ostr << "Error " << err << " "
                     << "(Getting error message failed: "
                     << (level < 2 ? _err_msg(_err, level + 1) : "Oops") << ")";
                return ostr.str();
            }; // if
        }
        else
        {
            // We got the message.
            return &buffer.front();
        }; // if
    }; // forever

#else

    // GNU version of strerror_r.
    char buffer[2000];
    return strerror_r(err, buffer, sizeof(buffer));

#endif

} // _err_msg


std::string dir_sep() { return "/"; } // dir_sep


std::string exe_path()
{

    static std::string const exe = "/proc/self/exe";

    buffer_t path(_size);
    int count = _count; // Max number of iterations.

    for (;;)
    {

        ssize_t len = readlink(exe.c_str(), &path.front(), path.size());

        if (len < 0)
        {
            // Oops.
            int err = errno;
            log_error("ERROR: Getting executable path failed: "
                      "Reading symlink `%s' failed: %s\n",
                      exe.c_str(), err_msg(err).c_str());
            exit(2);
        }; // if

        if (static_cast<size_t>(len) < path.size())
        {
            // We got the path.
            path.resize(len);
            break;
        }; // if

        // Oops, buffer is too small.
        if (count > 0)
        {
            --count;
            // Enlarge the buffer.
            path.resize(path.size() * 2);
        }
        else
        {
            log_error("ERROR: Getting executable path failed: "
                      "Reading symlink `%s' failed: Buffer of %lu bytes is "
                      "still too small\n",
                      exe.c_str(), (unsigned long)path.size());
            exit(2);
        }; // if

    }; // forever

    return std::string(&path.front(), path.size());

} // exe_path


std::string exe_dir()
{
    std::string path = exe_path();
    // We cannot pass path.c_str() to `dirname' bacause `dirname' modifies its
    // argument.
    buffer_t buffer(path.c_str(),
                    path.c_str() + path.size() + 1); // Copy with trailing zero.
    return dirname(&buffer.front());
} // exe_dir

#endif // __linux__

// -------------------------------------------------------------------------------------------------
// MS Windows
// -------------------------------------------------------------------------------------------------

#if defined(_WIN32)


#include <windows.h>

#include <cctype>
#include <algorithm>


static std::string _err_msg(int err, int level)
{

    std::string msg;

    LPSTR buffer = NULL;
    DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
        | FORMAT_MESSAGE_IGNORE_INSERTS;

    DWORD len = FormatMessageA(flags, NULL, err, LANG_USER_DEFAULT,
                               reinterpret_cast<LPSTR>(&buffer), 0, NULL);

    if (buffer == NULL || len == 0)
    {

        int _err = GetLastError();
        char str[1024] = { 0 };
        snprintf(str, sizeof(str),
                 "Error 0x%08x (Getting error message failed: %s )", err,
                 (level < 2 ? _err_msg(_err, level + 1).c_str() : "Oops"));
        msg = std::string(str);
    }
    else
    {

        // Trim trailing whitespace (including `\r' and `\n').
        while (len > 0 && isspace(buffer[len - 1]))
        {
            --len;
        }; // while

        // Drop trailing full stop.
        if (len > 0 && buffer[len - 1] == '.')
        {
            --len;
        }; // if

        msg.assign(buffer, len);

    }; // if

    if (buffer != NULL)
    {
        LocalFree(buffer);
    }; // if

    return msg;

} // _get_err_msg


std::string dir_sep() { return "\\"; } // dir_sep


std::string exe_path()
{

    buffer_t path(_size);
    int count = _count;

    for (;;)
    {

        DWORD len = GetModuleFileNameA(NULL, &path.front(),
                                       static_cast<DWORD>(path.size()));

        if (len == 0)
        {
            int err = GetLastError();
            log_error("ERROR: Getting executable path failed: %s\n",
                      err_msg(err).c_str());
            exit(2);
        }; // if

        if (len < path.size())
        {
            path.resize(len);
            break;
        }; // if

        // Buffer too small.
        if (count > 0)
        {
            --count;
            path.resize(path.size() * 2);
        }
        else
        {
            log_error("ERROR: Getting executable path failed: "
                      "Buffer of %lu bytes is still too small\n",
                      (unsigned long)path.size());
            exit(2);
        }; // if

    }; // forever

    return std::string(&path.front(), path.size());

} // exe_path


std::string exe_dir()
{

    std::string exe = exe_path();
    int count = 0;

    // Splitting path into components.
    buffer_t drv(_MAX_DRIVE);
    buffer_t dir(_MAX_DIR);
    count = _count;
#if defined(_MSC_VER)
    for (;;)
    {
        int rc =
            _splitpath_s(exe.c_str(), &drv.front(), drv.size(), &dir.front(),
                         dir.size(), NULL, 0, // We need neither name
                         NULL, 0 // nor extension
            );
        if (rc == 0)
        {
            break;
        }
        else if (rc == ERANGE)
        {
            if (count > 0)
            {
                --count;
                // Buffer is too small, but it is not clear which one.
                // So we have to enlarge all.
                drv.resize(drv.size() * 2);
                dir.resize(dir.size() * 2);
            }
            else
            {
                log_error("ERROR: Getting executable path failed: "
                          "Splitting path `%s' to components failed: "
                          "Buffers of %lu and %lu bytes are still too small\n",
                          exe.c_str(), (unsigned long)drv.size(),
                          (unsigned long)dir.size());
                exit(2);
            }; // if
        }
        else
        {
            log_error("ERROR: Getting executable path failed: "
                      "Splitting path `%s' to components failed: %s\n",
                      exe.c_str(), err_msg(rc).c_str());
            exit(2);
        }; // if
    }; // forever

#else // __MINGW32__

    // MinGW does not have the "secure" _splitpath_s, use the insecure version
    // instead.
    _splitpath(exe.c_str(), &drv.front(), &dir.front(),
               NULL, // We need neither name
               NULL // nor extension
    );
#endif // __MINGW32__

    // Combining components back to path.
    // I failed with "secure" `_makepath_s'. If buffer is too small, instead of
    // returning ERANGE, `_makepath_s' pops up dialog box and offers to debug
    // the program. D'oh! So let us try to guess the size of result and go with
    // insecure `_makepath'.
    buffer_t path(std::max(drv.size() + dir.size(), size_t(_MAX_PATH)) + 10);
    _makepath(&path.front(), &drv.front(), &dir.front(), NULL, NULL);

    return &path.front();

} // exe_dir


#endif // _WIN32


std::string err_msg(int err) { return _err_msg(err, 0); } // err_msg


// =================================================================================================
// C interface.
// =================================================================================================


char* get_err_msg(int err)
{
    char* msg = strdup(err_msg(err).c_str());
    CHECK_PTR(msg);
    return msg;
} // get_err_msg


char* get_dir_sep()
{
    char* sep = strdup(dir_sep().c_str());
    CHECK_PTR(sep);
    return sep;
} // get_dir_sep


char* get_exe_path()
{
    char* path = strdup(exe_path().c_str());
    CHECK_PTR(path);
    return path;
} // get_exe_path


char* get_exe_dir()
{
    char* dir = strdup(exe_dir().c_str());
    CHECK_PTR(dir);
    return dir;
} // get_exe_dir


char* get_temp_filename()
{
    char gFileName[256] = "";
    // Create a unique temporary file to allow parallel executed tests.
#if (defined(__linux__) || defined(__APPLE__)) && (!defined(__ANDROID__))
    sprintf(gFileName, "/tmp/tmpfile.XXXXXX");
    int fd = mkstemp(gFileName);
    if (fd == -1) return strdup(gFileName);
    close(fd);
#elif defined(_WIN32)
    UINT ret = GetTempFileName(".", "tmp", 0, gFileName);
    if (ret == 0) return gFileName;
#else
    MTdata d = init_genrand((cl_uint)time(NULL));
    sprintf(gFileName, "tmpfile.%u", genrand_int32(d));
#endif

    char* fn = strdup(gFileName);
    CHECK_PTR(fn);
    return fn;
}


// end of file //
