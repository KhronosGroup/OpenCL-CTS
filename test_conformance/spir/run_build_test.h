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
#ifndef __RUN_BUILD_TEST_H__
#define __RUN_BUILD_TEST_H__

#include <string>
#include <list>
#include <vector>
#include <utility>

class OclExtensions;

struct EventHandler{
    virtual void operator()(const std::string&, const std::string&) = 0;
    virtual std::string toString()const {return std::string();}
};

/*
 * Abstract task to be executed on a cl program.
 */
class Task{
public:
    Task(cl_device_id, const char* options);

    virtual bool execute() = 0;

    virtual ~Task();

    const char* getErrorLog() const;

protected:
  void setErrorLog(cl_program);

  cl_device_id m_devid;
  std::string  m_log;
  std::string  m_options;
};

/*
 * Build task - builds a given program.
 */
class BuildTask: public Task {
public:
    BuildTask(cl_program, cl_device_id, const char* options);

    bool execute();

private:
  cl_program m_program;
};

/*
 * Spir build task - build programs from SPIR binaries.
 */
class SpirBuildTask : public BuildTask {
public:
  SpirBuildTask(cl_program, cl_device_id, const char* options);
};

/*
 * Compile task - compiles a given program.
 */
class CompileTask: public Task {
public:
    CompileTask(cl_program, cl_device_id, const char* options);

    void addHeader(const char* hname, cl_program hprog);

    bool execute();

private:
  std::vector<std::pair<const char*,cl_program> > m_headers;
  cl_program m_program;
};

/*
 * Spir compile task - compiles programs from SPIR binaries.
 */
class SpirCompileTask: public CompileTask {
public:
  SpirCompileTask(cl_program, cl_device_id, const char* options);
};

/*
 * Link task - links a given programs to an OpecnCL executable.
 */
class LinkTask: public Task{
public:
    LinkTask(cl_program* programs, int num_programs, cl_context, cl_device_id,
             const char* options=NULL);

    bool execute();

    cl_program getExecutable() const;

    ~LinkTask();
private:
    cl_program   m_executable;
    cl_program*  m_programs;
    int          m_numPrograms;
    cl_context   m_context;
};

class TestRunner{
    EventHandler*const m_successHandler, *const m_failureHandler;
    const OclExtensions *m_devExt;

public:
    TestRunner(EventHandler *success, EventHandler *failure,
               const OclExtensions& devExt);

    bool runBuildTest(cl_device_id device, const char *folder,
                      const char *test_name, cl_uint size_t_width);
};

//
//Provides means to iterate over the kernels of a given program
//
class KernelEnumerator {
    std::list<std::string> m_kernels;

    void process(cl_program prog);
public:
    typedef std::list<std::string>::iterator iterator;

    KernelEnumerator(cl_program prog);
    iterator begin();
    iterator end();
    size_t size()const;
};

#endif//__RUN_BUILD_TEST_H__
