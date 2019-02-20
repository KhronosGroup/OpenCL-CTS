
PRODUCTS = \
	allocations/ \
	api/ \
	atomics/ \
	basic/ \
	buffers/ \
	commonfns/ \
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
	vec_step/

 
TOP=$(shell pwd)

all: $(PRODUCTS)

clean:
	@for testdir in $(dir $(PRODUCTS))  ; \
		do ( \
			echo "==================================================================================" ; \
			echo "Cleaning $$testdir" ; \
			echo "==================================================================================" ; \
			cd $$testdir && make clean \
			); \
		done \

$(PRODUCTS): 
	@echo "==================================================================================" ;
	@echo "(`date "+%H:%M:%S"`) Make $@" ;
	@echo "==================================================================================" ;
	cd $(dir $@) && make -i 

.PHONY: clean $(PRODUCTS)  all
