all: train_mnist.exe

profile: 

train_mnist.exe: src/train_mnist.cu
	nvcc src/train_mnist.cu src/ops.cu src/train.cu -L /usr/local/cuda/lib -lcudart -lcublas -o train_mnist.exe -run

clean:
	rm -f train_mnist.exe

install_python_deps:
	source .venv/bin/activate
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	pip3 install numpy
