all: transpose

transpose : transpose.cu
	nvcc -O3 -I../include transpose.cu -o ../bin/transpose

clean:
	rm -rf ../bin/transpose