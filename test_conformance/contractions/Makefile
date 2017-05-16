ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

CC = c++
CFLAGS = -g -arch i386 -arch x86_64 -Wall $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} -I ../../test_common/harness
LIBRARIES = -framework OpenCL -framework ApplicationServices -framework IOKit -I/System/Library/Frameworks/OpenCL.framework/Headers ${RC_CFLAGS} ${ATF}

release:
	echo "Build Release"
	$(CC) *.c ../../test_common/harness/rounding_mode.c ../../test_common/harness/kernelHelpers.c ../../test_common/harness/errorHelpers.c ../../test_common/harness/mt19937.c -Os $(CFLAGS) -o contractions $(LIBRARIES)

debug:
	echo "Build Debug"
	$(CC) *.c ../../test_common/harness/rounding_mode.c ../../test_common/harness/kernelHelpers.c ../../test_common/harness/errorHelpers.c ../../test_common/harness/mt19937.c -O0 $(CFLAGS) -D_DEBUG=1 -o contractions_debug  $(LIBRARIES)

test:	release
	arch -i386 ./contractions -c > cpu.log;	
	arch -i386 ./contractions -g > gpu.log;	

test64:	release
	arch -x86_64 ./contractions_debug -c > cpu64.log;	
	arch -x86_64 ./contractions_debug -g > gpu64.log;	
	echo "Testing 64-bit mode in progress. This may take up to 1 day to complete. See cpu64.log and gpu64.log for results."


clean:
	rm -f ./contractions
	rm -f ./contractions_debug

all: release
