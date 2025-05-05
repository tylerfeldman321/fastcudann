#include <iostream>
#include "../include/mnist_reader.hpp"

int main(int argc, char* argv[]) {
    std::string MNIST_DATA_LOCATION = "/home/ubuntu/fastcudann/data";
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    return 0;
}