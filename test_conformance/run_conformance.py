#! /usr/bin/python

#/******************************************************************
#//
#//  OpenCL Conformance Tests
#// 
#//  Copyright:  (c) 2008-2009 by Apple Inc. All Rights Reserved.
#//
#******************************************************************/

import os, re, sys, subprocess, time, commands, tempfile, math, string

DEBUG = 0

log_file_name = "opencl_conformance_results_" + time.strftime("%Y-%m-%d_%H-%M", time.localtime())+ ".log"
process_pid = 0

# The amount of time between printing a "." (if no output from test) or ":" (if output)
#  to the screen while the tests are running.
seconds_between_status_updates = 60*60*24*7  # effectively never

# Help info
def write_help_info() :
 print("run_conformance.py test_list [CL_DEVICE_TYPE(s) to test] [partial-test-names, ...] [log=path/to/log/file/]")
 print(" test_list - the .csv file containing the test names and commands to run the tests.")
 print(" [partial-test-names, ...] - optional partial strings to select a subset of the tests to run.")
 print(" [CL_DEVICE_TYPE(s) to test] - list of CL device types to test, default is CL_DEVICE_TYPE_DEFAULT.") 
 print(" [log=path/to/log/file/] - provide a path for the test log file, default is in the current directory.") 
 print("   (Note: spaces are not allowed in the log file path.")


# Get the time formatted nicely
def get_time() :
 return time.strftime("%d-%b %H:%M:%S", time.localtime())
 
# Write text to the screen and the log file
def write_screen_log(text) :
 global log_file
 print(text)
 log_file.write(text+"\n")

# Load the tests from a csv formated file of the form name,command
def get_tests(filename, devices_to_test):
 tests = []
 if (os.path.exists(filename) == False):
  print("FAILED: test_list \"" + filename + "\" does not exist.")
  print("")
  write_help_info()
  sys.exit(-1)
 file = open(filename, 'r')
 for line in file.readlines():
  comment = re.search("^#.*", line)
  if (comment):
   continue
  device_specific_match = re.search("^\s*(.+)\s*,\s*(.+)\s*,\s*(.+)\s*", line)
  if (device_specific_match):
   if (device_specific_match.group(1) in devices_to_test):
    test_path = string.replace(device_specific_match.group(3), '/', os.sep)
    test_name = string.replace(device_specific_match.group(2), '/', os.sep)
    tests.append((test_name, test_path))
   else:
    print("Skipping " + device_specific_match.group(2) + " because " + device_specific_match.group(1) + " is not in the list of devices to test.")
   continue
  match = re.search("^\s*(.+)\s*,\s*(.+)\s*", line)
  if (match): 
   test_path = string.replace(match.group(2), '/', os.sep)
   test_name = string.replace(match.group(1), '/', os.sep)
   tests.append((test_name, test_path))
 return tests


def run_test_checking_output(current_directory, test_dir, log_file):
  global process_pid, seconds_between_status_updates
  failures_this_run = 0
  start_time = time.time()
  # Create a temporary file for capturing the output from the test
  (output_fd, output_name) = tempfile.mkstemp()
  if ( not os.path.exists(output_name)) :
    write_screen_log("\n           ==> ERROR: could not create temporary file %s ." % output_name)
    os.close(output_fd)
    return -1
  # Execute the test
  program_to_run = test_dir_without_args = test_dir.split(None, 1)[0]
  if ( os.sep == '\\' ) : program_to_run += ".exe"
  if (os.path.exists(current_directory + os.sep + program_to_run)) :
    os.chdir(os.path.dirname(current_directory+os.sep+test_dir_without_args) )
    try:
      if (DEBUG): p = subprocess.Popen("", stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
      else :  p = subprocess.Popen(current_directory + os.sep + test_dir, stderr=output_fd, stdout=output_fd, shell=True)
    except OSError:
      write_screen_log("\n           ==> ERROR: failed to execute test. Failing test. : " + str(OSError))
      os.close(output_fd)
      return -1
  else:
    write_screen_log("\n           ==> ERROR: test file (" + current_directory + os.sep + program_to_run +") does not exist.  Failing test.")
    os.close(output_fd)
    return -1
  # Set the global pid so we can kill it if this is aborted
  process_pid = p.pid
  # Read one character at a time from the temporary output file while the process is running.
  # When we get an end-of-line, look for errors and write the results to the log file.
  # This allows us to process the file as it is being produced.
  # Keep track of the state for reading
  # Whether we are done, if we have more to read, and where in the file we last read
  done = False
  more_to_read = True
  pointer = 0
  pointer_at_last_user_update = 0
  output_this_run = False
  try: 
    read_output = open(output_name, 'r')
  except IOError:
    write_screen_log("\n           ==> ERROR: could not open output file from test.")
    os.close(output_fd)
    return -1
  line = ""
  while (not done or more_to_read):
    os.fsync(output_fd)
    # Determine if we should display some output
    elapsed_time = (time.time() - start_time)
    if (elapsed_time > seconds_between_status_updates):
      start_time = time.time()
      # If we've received output from the test since the last update, display a #
      if (pointer != pointer_at_last_user_update):
        sys.stdout.write(":")
      else:
        sys.stdout.write(".")
      pointer_at_last_user_update = pointer
      sys.stdout.flush()
    # Check if we're done
    p.poll()
    if (not done and p.returncode != None):
      if (p.returncode < 0):
        if (not output_this_run): 
         print ""
         output_this_run = True
        write_screen_log("           ==> ERROR: test killed/crashed: " + str(p.returncode)+ ".")
      done = True
    # Try reading
    try:
      read_output.seek(pointer)
      char_read = read_output.read(1)
    except IOError:
      time.sleep(1)
      continue
    # If we got a full line then process it
    if (char_read == "\n"):
      # Look for failures and report them as such
      match = re.search(".*(FAILED|ERROR).*", line)
      if (match):
        if (not output_this_run): 
         print "" 
         output_this_run = True
        print("           ==> " + line.replace('\n',''))
      match = re.search(".*FAILED.*", line)
      if (match):
        failures_this_run = failures_this_run + 1
      match = re.search(".*(PASSED).*", line)
      if (match):
       if (not output_this_run): 
        print "" 
        output_this_run = True
       print("               " + line.replace('\n',''))
      # Write it to the log
      log_file.write("     " + line +"\n")
      log_file.flush()
      line = ""
      pointer = pointer + 1
    # If we are at the end of the file, then re-open it to get new data
    elif (char_read == ""):
      more_to_read = False
      read_output.close()
      time.sleep(1)
      try: 
        os.fsync(output_fd)
        read_output = open(output_name, 'r')
        # See if there is more to read. This happens if the process ends and we have data left.
        read_output.seek(pointer)
        if (read_output.read(1) != ""):
          more_to_read = True
      except IOError:
        write_screen_log("\n           ==> ERROR: could not reopen output file from test.")
        return -1
        done = True
    else:
      line = line + char_read
      pointer = pointer + 1
  # Now we are done, so write out any remaining data in the file:
  # This should only happen if the process exited with an error.
  os.fsync(output_fd)
  while (read_output.read(1) != ""):
    log_file.write(read_output.read(1))
  # Return the total number of failures
  if (p.returncode == 0 and failures_this_run > 0):
   write_screen_log("\n           ==> ERROR: Test returned 0, but number of FAILED lines reported is " + str(failures_this_run) +".")
   return failures_this_run
  return p.returncode


def run_tests(tests) :
  global curent_directory
  global process_pid
  # Run the tests
  failures = 0
  previous_test = None
  test_number = 1
  for test in tests:
   # Print the name of the test we're running and the time
   (test_name, test_dir) = test
   if (test_dir != previous_test):
    print("==========   " + test_dir)
    log_file.write("========================================================================================\n")
    log_file.write("========================================================================================\n")
    log_file.write("(" + get_time() + ")     Running Tests: " + test_dir +"\n")
    log_file.write("========================================================================================\n")
    log_file.write("========================================================================================\n")
    previous_test = test_dir
   print("("+get_time()+")     BEGIN  " + test_name.ljust(40) +": "),
   log_file.write("     ----------------------------------------------------------------------------------------\n")
   log_file.write("     (" + get_time() + ")     Running Sub Test: " + test_name + "\n")
   log_file.write("     ----------------------------------------------------------------------------------------\n")
   log_file.flush()
   sys.stdout.flush()
  
   # Run the test
   result = 0
   start_time = time.time()
   try:
    process_pid = 0
    result = run_test_checking_output(current_directory, test_dir, log_file)
   except KeyboardInterrupt:
    # Catch an interrupt from the user
    write_screen_log("\nFAILED: Execution interrupted.  Killing test process, but not aborting full test run.")
    os.kill(process_pid, 9)
    answer = raw_input("Abort all tests? (y/n)")
    if (answer.find("y") != -1):
     write_screen_log("\nUser chose to abort all tests.")
     log_file.close()
     sys.exit(-1)     
    else:
     write_screen_log("\nUser chose to continue with other tests. Reporting this test as failed.")
     result = 1   
   run_time = (time.time() - start_time)
          
   # Move print the finish status
   if (result == 0):
    print("("+get_time()+")     PASSED " + test_name.ljust(40) +": (" + str(int(run_time)).rjust(3) + "s, test " + str(test_number).rjust(3) + os.sep + str(len(tests)) +")"),
   else:
    print("("+get_time()+")     FAILED " + test_name.ljust(40) +": (" + str(int(run_time)).rjust(3) + "s, test " + str(test_number).rjust(3) + os.sep + str(len(tests)) +")"),

   test_number = test_number + 1
   log_file.write("     ----------------------------------------------------------------------------------------\n")
   log_file.flush()
    
   print("")
   if (result != 0):
    log_file.write("  *******************************************************************************************\n")
    log_file.write("  *  ("+get_time()+")     Test " + test_name + " ==> FAILED: " + str(result)+"\n")
    log_file.write("  *******************************************************************************************\n")
    failures = failures + 1
   else:
    log_file.write("     ("+get_time()+")     Test " + test_name +" passed in " + str(run_time) + "s\n")
     
   log_file.write("     ----------------------------------------------------------------------------------------\n")
   log_file.write("\n")
  return failures





# ########################
# Begin OpenCL conformance run script
# ########################

if (len(sys.argv) < 2):
 write_help_info()
 sys.exit(-1)


current_directory = os.getcwd()
# Open the log file
for arg in sys.argv:
 match = re.search("log=(\S+)", arg)
 if (match):
  log_file_name = match.group(1).rstrip('/') + os.sep + log_file_name
try:
 log_file = open(log_file_name, "w")
except IOError:
 print "Could not open log file " + log_file_name

# Determine which devices to test
device_types = ["CL_DEVICE_TYPE_DEFAULT", "CL_DEVICE_TYPE_CPU", "CL_DEVICE_TYPE_GPU", "CL_DEVICE_TYPE_ACCELERATOR", "CL_DEVICE_TYPE_ALL"]
devices_to_test = []
for device in device_types:
 if device in sys.argv[2:]:
  devices_to_test.append(device)
if (len(devices_to_test) == 0):
 devices_to_test = ["CL_DEVICE_TYPE_DEFAULT"]
write_screen_log("Testing on: " + str(devices_to_test))

# Get the tests
tests = get_tests(sys.argv[1], devices_to_test)

# If tests are specified on the command line then run just those ones
tests_to_use = []
num_of_patterns_to_match = 0
for arg in sys.argv[2:]:
 if arg in device_types:
  continue
 if re.search("log=(\S+)", arg):
  continue
 num_of_patterns_to_match = num_of_patterns_to_match + 1
 found_it = False
 for test in tests:
  (test_name, test_dir) = test
  if (test_name.find(arg) != -1 or test_dir.find(arg) != -1):
   found_it = True
   if (test not in tests_to_use):
    tests_to_use.append(test)
 if (found_it == False):
  print("Failed to find a test matching " + arg)
if (len(tests_to_use) == 0):
 if (num_of_patterns_to_match > 0):
  print("FAILED: Failed to find any tests matching the given command-line options.")
  print("")
  write_help_info()
  sys.exit(-1)
else:
 tests = tests_to_use[:]

write_screen_log("Test execution arguments: " + str(sys.argv))
write_screen_log("Logging to file " + log_file_name +".")
write_screen_log("Loaded tests from " + sys.argv[1] + ", total of " + str(len(tests)) + " tests selected to run:")
for (test_name, test_command) in tests:
 write_screen_log(test_name.ljust(50) + " (" + test_command +")")

# Run the tests
total_failures = 0
for device_to_test in devices_to_test:
 os.environ['CL_DEVICE_TYPE'] = device_to_test
 write_screen_log("========================================================================================")
 write_screen_log("========================================================================================")
 write_screen_log(("Setting CL_DEVICE_TYPE to " + device_to_test).center(90))
 write_screen_log("========================================================================================")
 write_screen_log("========================================================================================")
 failures = run_tests(tests)
 write_screen_log("========================================================================================")
 if (failures == 0):
  write_screen_log(">> TEST on " + device_to_test + " PASSED")
 else:
  write_screen_log(">> TEST on " + device_to_test + " FAILED (" + str(failures) + " FAILURES)")
 write_screen_log("========================================================================================")
 total_failures = total_failures + failures

write_screen_log("("+get_time()+") Testing complete.  " + str(total_failures) + " failures for " + str(len(tests)) + " tests.")
log_file.close()
