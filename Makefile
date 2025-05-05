all: train_mnist.exe

profile: 

train_mnist.exe: src/train_mnist.cu
	nvcc src/train_mnist.cu -L /usr/local/cuda/lib -lcudart -o train_mnist.exe -run

clean:
	rm -f train_mnist.exe
