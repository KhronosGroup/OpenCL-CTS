
PRODUCTS = harness/\

# utils/
 
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
	cd $(dir $@) && make

.PHONY: clean $(PRODUCTS)  all
