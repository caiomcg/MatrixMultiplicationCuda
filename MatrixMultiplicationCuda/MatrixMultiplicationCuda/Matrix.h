#pragma once

#include "cuda_runtime.h"

#include <iostream>
#include <cstdlib>

class Matrix {
private:
	unsigned int p_width;
	unsigned int p_height;
	int* p_elements;

public:
	Matrix();
	Matrix(const unsigned int width, const unsigned int height);
	~Matrix();
	Matrix(const Matrix& m);

	unsigned int getWidth() const;
	unsigned int getHeight() const;
	int* getElements() const;

	cudaError allocWithCuda(const unsigned int width, const unsigned int height);

	void print() const;
 };