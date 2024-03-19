all: elementwise_add relu gelu

relu : relu.cu
	nvcc -O3 -I../include relu.cu -o ../bin/relu

gelu : gelu.cu
	nvcc -O3 -I../include gelu.cu -o ../bin/gelu

elementwise_add : elementwise_add.cu
	nvcc -O3 -I../include elementwise_add.cu -o ../bin/elementwise_add

clean:
	rm -rf ../bin/elementwise_add
	rm -rf ../bin/relu
	rm -rf ../bin/gelu