#!/usr/bin/python

import sys, os, re
from subprocess import Popen, PIPE
from optparse import OptionParser

# trail_spaces: This method removes the trailing whitespaces and trailing tabs
def trail_spaces(line):
    newline=line
    carreturn = 0
    if re.search("\r\n",line):
        carreturn = 1
    status = re.search("\s+$",line)
    if status:
        if carreturn:
            newline = re.sub("\s+$","\r\n",line)
        else:
            newline = re.sub("\s+$","\n",line)

    status = re.search("\t+$",newline)
    if status:
        newline = re.sub("\t+$","",newline)
    return newline

#convert_tabs: This methos converts tabs to 4 spaces
def convert_tabs(line):
    newline=line
    status = re.search("\t",line)
    if status:
        newline = re.sub("\t","    ",line)
    return newline

#convert_lineends: This method converts lineendings from DOS to Unix
def convert_lineends(line):
    newline=line
    status = re.search("\r\n",line)
    if status:
        newline = re.sub("\r\n","\n",line)
    return newline

#processfile: This method processes each file passed to it depending
#             on the flags passed

def processfile(file,tabs, lineends,trails,verbose):
    processed_data = []
    if verbose:
        print "processing file: "+file

    with open(file,'r') as fr:
        data = fr.readlines()
    for line in data:
        if tabs:
            line = convert_tabs(line)
        if lineends:
            line = convert_lineends(line)
        if trails:
            line = trail_spaces(line)
        processed_data.append(line)

    with open(file,'w') as fw:
        fw.writelines(processed_data)

#findfiles: This method finds all the code files present in current
#            directory and subdirectories.

def findfiles(tabs,lineends,trails,verbose):
    testfiles = []
    for root, dirs, files in os.walk("./"):
        for file in files:
            for extn in ('.c','.cpp','.h','.hpp'):
                if file.endswith(extn):
                    testfiles.append(os.path.join(root, file))
    for file in testfiles:
        processfile(file,tabs,lineends,trails,verbose)

# Main function

def main():

    parser = OptionParser()
    parser.add_option("--notabs", dest="tabs", action="store_false", default=True, help="Disable converting tabs to 4 spaces.")
    parser.add_option("--notrails", dest="trails", action="store_false", default=True, help="Disable removing trailing whitespaces and trailing tabs.")
    parser.add_option("--nolineends", dest="lineends", action="store_false", default=True, help=" Disable converting line endings to Unix from DOS.")
    parser.add_option("--verbose", dest="verbose", action="store_true", default=False, help="Prints out the files being processed.")
    parser.add_option("--git", dest="SHA1", default="", help="Processes only the files present in the particular <SHA1> commit.")
    parser.add_option('-o', action="store", default=True, help="Default: All the code files (.c,.cpp,.h,.hpp) in the current directory and subdirectories will be processed")

    (options, args) = parser.parse_args()

    if options.SHA1:
        pl = Popen(["git","show", "--pretty=format:", "--name-only",options.SHA1], stdout=PIPE)
        cmdout = pl.communicate()[0]
        gitout=cmdout.split("\n")
        for file in gitout:
            print file
            if file:
                processfile(file,options.tabs,options.lineends,options.trails,options.verbose)


    if not options.SHA1:
        findfiles(options.tabs,options.lineends,options.trails,options.verbose)

# start the process by calling main
main()
