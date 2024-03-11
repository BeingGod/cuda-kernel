all: rms_norm

rms_norm : rms_norm.cu
	nvcc -O3 -I../include rms_norm.cu -o ../bin/rms_norm

clean:
	rm -rf ../bin/rms_norm