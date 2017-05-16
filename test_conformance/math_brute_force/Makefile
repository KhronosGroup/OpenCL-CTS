ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

CC = cc
CFLAGS = -g -Wall -Wshorten-64-to-32 $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF}
LIBRARIES = -framework OpenCL -framework ApplicationServices -framework IOKit -I/System/Library/Frameworks/OpenCL.framework/Headers ${RC_CFLAGS} ${ATF}

release:
	echo "Build Release"
	$(CC) *.c ../../test_common/harness/mt19937.c ../../test_common/harness/rounding_mode.c ../../test_common/harness/ThreadPool.c -Os $(CFLAGS) -o bruteforce $(LIBRARIES)

debug:
	echo "Build Debug"
	$(CC) *.c ../../test_common/harness/mt19937.c ../../test_common/harness/rounding_mode.c ../../test_common/harness/ThreadPool.c -O0 $(CFLAGS) -D_DEBUG=1 -o bruteforce_debug  $(LIBRARIES)

test:	release
	arch -i386 ./bruteforce -c > cpu.log	&
	arch -i386 ./bruteforce -g > gpu.log	&
	echo "Testing 32-bit mode in progress. This may take up to 1 day to complete. See cpu.log and gpu.log for results."

test64:	release
	arch -x86_64 ./bruteforce -c > cpu64.log	&
	arch -x86_64 ./bruteforce -g > gpu64.log	&
	echo "Testing 64-bit mode in progress. This may take up to 1 day to complete. See cpu64.log and gpu64.log for results."


clean:
	rm -f ./bruteforce_debug
	rm -f ./bruteforce

all:	release 
