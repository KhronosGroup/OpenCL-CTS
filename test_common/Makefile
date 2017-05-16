
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
		then cd $(dir $@) && make; \
		else echo "Warning: Directory '$@' Does Not Exist"; \
		fi; \

.PHONY: clean $(PRODUCTS)  all
