
PRODUCTS = \
	allocations/ \
	api/ \
	atomics/ \
	basic/ \
	buffers/ \
	commonfns/ \
  compatibility/test_conformance/ \
	compiler/ \
	computeinfo/ \
	contractions/ \
	conversions/ \
	device_partition/ \
	events/ \
	geometrics/ \
	gl/ \
	half/ \
	headers/ \
	images/ \
	integer_ops/ \
	math_brute_force/ \
	mem_host_flags/ \
	multiple_device_context/ \
	printf/ \
	profiling/ \
	relationals/ \
	select/ \
	thread_dimensions/ \
	vec_align/ \
	vec_step/ \
	workgroups/

 
TOP=$(shell pwd)

all: $(PRODUCTS)

clean:
	@for testdir in $(dir $(PRODUCTS))  ; \
		do ( \
			echo "==================================================================================" ; \
			echo "Cleaning $$testdir" ; \
			echo "==================================================================================" ; \
			if test -d $$testdir; \
				then cd $$testdir && make clean; \
				else echo "Warning: Directory '$$testdir' Does Not Exist"; \
			fi; \
			); \
		done \

$(PRODUCTS): 
	@echo "==================================================================================" ;
	@echo "(`date "+%H:%M:%S"`) Make $@" ;
	@echo "==================================================================================" ;
	@if test -d $@; \
		then cd $(dir $@) && make -i; \
		else echo "Warning: Directory '$@' Does Not Exist"; \
		fi; \

.PHONY: clean $(PRODUCTS)  all
