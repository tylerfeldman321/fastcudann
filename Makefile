VENV_PYTHON = .venv/bin/python3
VENV_PIP = .venv/bin/pip3

all: train_mnist.exe profile_python

train_mnist.exe: src/train_mnist.cu
	nvcc src/train_mnist.cu src/ops.cu src/train.cu -L /usr/local/cuda/lib -lcudart -lcublas -lcudnn -o train_mnist.exe -run

clean:
	rm -f train_mnist.exe

setup_python_deps:
	echo "Setting up python dependencies..."
	python3 -m venv .venv
	$(VENV_PIP) install numpy
	$(VENV_PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu128

profile_python: $(VENV_PYTHON)
	echo "========== Running mnist_pytorch_optimized.py =========="
	$(VENV_PYTHON) python/mnist_pytorch_optimized.py

	echo "========== Running mnist_pytorch.py =========="
	$(VENV_PYTHON) python/mnist_pytorch.py

	echo "========== Running mnist_pytorch_optimized =========="
	$(VENV_PYTHON) python/mnist_numpy.py
