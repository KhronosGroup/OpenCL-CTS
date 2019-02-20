#! /usr/bin/python

#  //  OpenCL Conformance Tests
#  // 
#  //  Copyright:	(c) 2009-2013 by Apple Inc. All Rights Reserved.
#  //

import os, re, sys, subprocess, time

# A script to run the entierty of math_brute_force, but to run each separate job in parallel.

def DEBUG(text, level=1):
 if (DEBUG_LEVEL >= level): print(text)

def write_info(text):
 print text,
 if (ATF):
  ATF_log.write("<Info>"+text+"</Info>\n")
  ATF_log.flush()
  
def write_error(text):
 print "ERROR:" + text,
 if (ATF):
  ATF_log.write("<Error>"+text+"</Error>\n")
  ATF_log.flush()
  
def start_atf():
 global ATF, ATF_log
 DEBUG("start_atf()")
 if (os.environ.get("ATF_RESULTSDIRECTORY") == None):
  ATF = False
  DEBUG("\tATF not defined",0)
  return
 ATF = True
 ATF_output_file_name = "TestLog.xml"
 output_path = os.environ.get("ATF_RESULTSDIRECTORY")
 try:
	ATF_log = open(output_path+ATF_output_file_name, "w")
 except IOError:
  DEBUG("Could not open ATF file " + ATF_output_file_name, 0)
  ATF = False
  return
 DEBUG("ATF Enabled")
 # Generate the XML header
 ATF_log.write("<Log>\n")
 ATF_log.write("<TestStart/>\n")
 DEBUG("Done start_atf()")

def stop_atf():
 DEBUG("stop_atf()")
 if (ATF):
  ATF.write("<TestFinish/>\n")
  ATF.write("</Log>\n")
  ATF.close()

def get_time() :
 return time.strftime("%A %H:%M:%S", time.localtime())

def start_test(id):
 DEBUG("start_test("+str(id) + ")")
 command = test + " " + str(id) + " " + str(id)
 try:
  write_info(get_time() + " Executing " + command + "...")
  p = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
 except OSError:
  write_error("Failed to execute " + command)
  return
 running_tests[id] = p
 DEBUG("start_test("+str(id) + ") added: " + str(running_tests[id]) + \
 ", now " + str(len(running_tests.keys())) + " tests running")




DEBUG_LEVEL = 2
test = "./bruteforce -w"
instances = 4
max_test_ID = 12
running_tests = {}
ATF_log = None
ATF = False

# Start the ATF log
start_atf()
next_test = 0
next_test_to_finish = 0

while ( (next_test <= max_test_ID) | (next_test_to_finish <= max_test_ID)):
 # If we want to run more tests, start them
 while ((len(running_tests.keys()) < instances) & (next_test <= max_test_ID)):
  start_test(next_test)
  next_test = next_test + 1
  time.sleep(1)
 # Check if the oldest test has finished
 p = running_tests[next_test_to_finish]
 if (p.poll() != None):
  write_info(get_time() + " Test " + str(next_test_to_finish) +" finished.")
  del running_tests[next_test_to_finish]
  next_test_to_finish = next_test_to_finish + 1
  # Write the results from the test out
  for line in p.stdout.readlines():
   write_info(line)
  for line in p.stderr.readlines():
   write_error(line)
   
 time.sleep(1)
 

# Stop the ATF log
stop_atf()
