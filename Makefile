all:
	module load openmpi/2.1.2/2018
	module load pgi/18.4
	nvcc -O3 -std=c++11 -arch=sm_60 -ccbin=mpic++ -o sol3 solution3.cu

clean:
	rm -f sol3