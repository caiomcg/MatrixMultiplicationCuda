#include "Matrix.h"

Matrix::Matrix() : p_width(0), p_height(0), p_elements(nullptr) {
}

Matrix::Matrix(const unsigned int width, const unsigned int height) : p_width(width), p_height(height), p_elements(nullptr) {
	p_elements = new int[p_width * p_height * sizeof(int)];
}

Matrix::~Matrix() {
	if (p_elements == nullptr) {
		delete p_elements;
	}
}

Matrix::Matrix(const Matrix& m) {
	p_width = m.p_width;
	p_height = m.p_height;

	p_elements = new int[p_width * p_height * sizeof(int)];

	std::memcpy(p_elements, m.p_elements, p_width * p_height * sizeof(int));
}

unsigned int Matrix::getWidth() const {
	return p_width;
}

unsigned int Matrix::getHeight() const {
	return p_height;
}

int* Matrix::getElements() const {
	return p_elements;
}

cudaError Matrix::allocWithCuda(const unsigned int width, const unsigned int height) {
	p_width = width;
	p_height = height;

	return cudaMalloc(&p_elements, p_height * p_width * sizeof(int));
}

void Matrix::print() const {
	for (int i = 0; i < p_width; i++) {
		for (int j = 0; j < p_height; j++) {
			std::cout << p_elements[i*p_width + j] << " ";
		}
		std::cout << std::endl;
	}
}