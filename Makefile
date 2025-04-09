all: cuda-dev-graphs.exe

profile: 

cuda-dev-graphs.exe: cuda-dev-graphs.cu
	nvcc cuda-dev-graphs.cu -L /usr/local/cuda/lib -lcudart -o cuda-dev-graphs.exe -run

clean:
	rm cuda-dev-graphs.exe
