ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCFILES = cl_utils.c Test_vLoadHalf.c Test_roundTrip.c \
           Test_vStoreHalf.c main.c

CC = c++
CFLAGS = -g -Wall -Wshorten-64-to-32 $(COMPILERFLAGS) ${RC_CFLAGS} \
         ${USE_ATF}

LIBRARIES = -framework OpenCL ${RC_CFLAGS} ${ATF}


all: release

OBJECTS := ${SRCFILES:.c=.o}

release:
	echo "Build Release"
	$(CC) $(SRCFILES) -Os $(CFLAGS) -o Test_half $(LIBRARIES)

debug: $(OBJECTS)
	echo "Build Debug"
	$(CC) $(OBJECTS) -O0 $(CFLAGS) -o Test_half_debug -D_DEBUG=1 $(LIBRARIES)

test:	release
	arch -i386 ./Test_half -c > cpu.log	&
	arch -i386 ./Test_half -g > gpu.log	&
	echo "Testing 32-bit mode in progress. See cpu.log and gpu.log for results."

test64:	release
	arch -x86_64 ./Test_half -c > cpu64.log	&
	arch -x86_64 ./Test_half -g > gpu64.log	&
	echo "Testing 64-bit mode in progress. See cpu64.log and gpu64.log for results."


clean:
	rm -f ./Test_half_debug
	rm -f ./Test_half


.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
