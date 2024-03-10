all: elementwise_add relu

relu : relu.cu
	nvcc -O3 -I../include relu.cu -o ../bin/relu

elementwise_add : elementwise_add.cu
	nvcc -O3 -I../include elementwise_add.cu -o ../bin/elementwise_add

clean:
	rm -rf ../bin/elementwise_add
	rm -rf ../bin/relu