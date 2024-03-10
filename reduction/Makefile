all: reduce_sum

reduce_sum : reduce_sum.cu
	nvcc -O3 -I../include reduce_sum.cu -o ../bin/reduce_sum

clean:
	rm -rf ../bin/reduce_sum