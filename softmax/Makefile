all: softmax

softmax : softmax.cu
	nvcc -O3 -I../include softmax.cu -o ../bin/softmax

clean:
	rm -rf ../bin/softmax