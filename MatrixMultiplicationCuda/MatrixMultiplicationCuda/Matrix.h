/**
 * @class Matrix
 *
 * @brief Matrix Class
 *
 * This class is responsible to store all information related to a matrix
 * alongside with its data.
 *
 * @author Caio Marcelo Campoy Guedes <caiomcg@gmail.com>
 */

#pragma once

#include "cuda_runtime.h" //CUDA.

#include <iostream> //C++ standard I/O.
#include <cstdlib> //C stdlib.

class Matrix {
private:
	unsigned int p_width; //Width of the matrix.
	unsigned int p_height; //Height of the matrix.
	int* p_elements; //Pointer to the matrix elements.

public:

	/**
	 * @brief Matrix default constructor.
	 */
	Matrix();

	/**
	 * @brief Matrix Custom constructor.
	 * 
	 * @param int The width of the matrix.
	 * @param int The height of the matrix.
	 */
	Matrix(const unsigned int width, const unsigned int height);

	/**
	 * @brief Matrix destructor
	 */
	~Matrix();

	/**
	 * @brief Matrix copy constructor.
	 * 
	 * @param m The object to be copied.
	 */
	Matrix(const Matrix& m);

	/**
	 * @brief Returns the width of the matrix.
	 * 
	 * @return The matrix width.
	 */
	unsigned int getWidth() const;

	/**
	 * @brief Returns the height of the matrix.
	 * 
	 * @return The matrix height.
	 */
	unsigned int getHeight() const;

	/**
	 * @brief Returns the elements of the matrix.
	 * 
	 * @return The matrix elements(don't have to be dealocated).
	 */
	int* getElements() const;

	/**
	 * @brief Alloc the matrix with cudaMalloc
	 * 
	 * @param int Matrix width
	 * @param int Matrix height
	 * 
	 * @return The eventual error that ocurred.
	 */
	cudaError allocWithCuda(const unsigned int width, const unsigned int height);

	/**
	 * @brief Print the matrix
	 */
	void print() const;
 };