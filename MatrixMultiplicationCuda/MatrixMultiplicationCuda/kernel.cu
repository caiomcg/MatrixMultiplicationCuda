#include "cuda_runtime.h" //CUDA.
#include "device_launch_parameters.h" //CUDA.

#include <iostream> //C++ standard I/O.
#include <exception> //C++ exceptions.

#include <Windows.h> //Windows header. 

#include "ArgumentParser.h" //Argument Parser.
#include "Matrix.h" //Matrix Object.

#define THREAD_BLOCK_SIZE 3 //Default block size for matrix under the GPU.

/**
 * @brief Code Usage.
 */
void usage(void) {
	std::cout << "Usage: ./MatrixMultiplicationCuda [FILE] [FILE]" << std::endl;
}

/**
 * @brief Matrix Struct used under the CUDA environment.
 * 
 * @param a First Matrix.
 * @param b Second Matrix.
 * @param c Resulting Matrix.
 */
typedef struct {
	int* elements;
	int width;
	int height;
} MatrixStruct;

/**
 * @brief The Multiplication that occurs under a CUDA thread.
 * 
 * @param a First Matrix.
 * @param b Second Matrix.
 * @param c Resulting Matrix.
 */
__global__ void multiplyMatrixesGPU(MatrixStruct a, MatrixStruct b, MatrixStruct c) {
	int calc = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row > a.height) || (col > b.width)) return;

	for (int i = 0; i < a.width; i++) {
		calc += (a.elements[row * a.width + i]) * (b.elements[i * b.width + col]);
	}
	c.elements[row * c.width + col] = calc;
}
/**
 * @brief Print 3x3 Matrixes.
 * @details Print the matrixes[formatted].
 * 
 * @param a First Matrix.
 * @param b Second Matrix.
 * @param c Resulting Matrix.
 */
void print3x3(const Matrix* a, const Matrix* b, const Matrix* c) {
	for (int i = 0; i < c->getWidth(); i++) {

		std::cout << "|";
		for (int j = 0; j < c->getHeight(); j++) {
			std::cout << a->getElements()[i*c->getWidth() + j] << (j == c->getHeight() - 1 ? "" : " ");
		}
		std::cout << "|";


		std::cout << (i == c->getWidth()/2 ? " * " : "   ");

		std::cout << "|";
		for (int j = 0; j < c->getHeight(); j++) {
			std::cout << b->getElements()[i*c->getWidth() + j] << (j == c->getHeight() - 1 ? "" : " ");
		}
		std::cout << "|";

		std::cout << (i == c->getWidth() / 2 ? " = " : "   ");
		
		std::cout << "|";
		for (int j = 0; j < c->getHeight(); j++) {
			std::cout << c->getElements()[i*c->getWidth() + j] << (j == c->getHeight() - 1 ? "" : " ");
		}
		std::cout << "|";

		std::cout << std::endl;
	}
}

/**
 * @brief Prepare the matrix to be used under CUDA and calls the cuda kernel.
 * 
 * @param a First matrix.
 * @param b Second Matrix.
 * @param c Resulting Matrix.
 */
void multiplyMatrixes(const Matrix* a, const Matrix* b, Matrix* c) {
	MatrixStruct gpu_a = { nullptr, a->getWidth(), a->getHeight() };
	MatrixStruct gpu_b = { nullptr, b->getWidth(), b->getHeight() };
	MatrixStruct gpu_c = { nullptr, c->getWidth(), c->getHeight() };

	std::cout << "CUDA PREPARATION" << std::endl;
	
	std::cout << "-------------------------------------------------------------" << std::endl;

	cudaError error = cudaMalloc(&gpu_a.elements, gpu_a.height * gpu_a.width * sizeof(int));
	std::cerr << "CUDA MALLOC A: " << cudaGetErrorString(error) << std::endl;
	error = cudaMemcpy(gpu_a.elements, a->getElements(), a->getWidth() * a->getHeight() * sizeof(int), cudaMemcpyHostToDevice);
	std::cerr << "CUDA MEMCPY A: " << cudaGetErrorString(error) << std::endl;

	error = cudaMalloc(&gpu_b.elements, gpu_b.height * gpu_b.width * sizeof(int));
	std::cerr << "CUDA MALLOC B: " << cudaGetErrorString(error) << std::endl;
	error = cudaMemcpy(gpu_b.elements, b->getElements(), b->getWidth() * b->getHeight() * sizeof(int), cudaMemcpyHostToDevice);
	std::cerr << "CUDA MEMCPY B: " << cudaGetErrorString(error) << std::endl;

	error = cudaMalloc(&gpu_c.elements, gpu_c.height * gpu_c.width * sizeof(int));
	std::cerr << "CUDA MALLOC C: " << cudaGetErrorString(error) << std::endl;

	std::cout << "-------------------------------------------------------------\n" << std::endl;

	dim3 dimBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	dim3 dimGrid((b->getWidth() + dimBlock.x - 1) / dimBlock.x, (a->getHeight() + dimBlock.y - 1) / dimBlock.y);

	multiplyMatrixesGPU <<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);

	std::cout << "KERNEL RUN" << std::endl;
	std::cout << "-------------------------------------------------------------" << std::endl;

	error = cudaThreadSynchronize();	
	std::cerr << "RUN KERNEL: " << cudaGetErrorString(error) << std::endl;

	error = cudaMemcpy(c->getElements(), gpu_c.elements, c->getWidth() * c->getHeight() * sizeof(int), cudaMemcpyDeviceToHost);
	std::cerr << "FROM DEVICE TO HOST: " << cudaGetErrorString(error) << std::endl;

	std::cout << "-------------------------------------------------------------\n" << std::endl;
	// Free device memory
	cudaFree(gpu_a.elements);
	cudaFree(gpu_b.elements);
	cudaFree(gpu_c.elements);
}

/**
 * @brief The main function;
 * 
 * @param argc Amount of arguments passed by the command line.
 * @param argv The arguments passed by the command line.
 * 
 * @return 0 if no problem occured.
 */
int main(int argc, char** argv) {
	ArgumentParser* argParser = nullptr;
	Matrix* firstMatrix		  = nullptr;
	Matrix* secondMatrix	  = nullptr;
	Matrix* resultMatrix	  = nullptr;

	try {
		argParser = new ArgumentParser(argc, argv);
	} catch (std::exception& e) {
		std::cerr << "What(): " << e.what() << std::endl;
	}

	firstMatrix = argParser->prepareMatrix(MATRIX::FIRST);
	secondMatrix = argParser->prepareMatrix(MATRIX::SECOND);
	resultMatrix = new Matrix(firstMatrix->getHeight(), secondMatrix->getWidth());

	multiplyMatrixes(firstMatrix, secondMatrix, resultMatrix);

	print3x3(firstMatrix, secondMatrix, resultMatrix);

	system("pause");

	if (argParser != nullptr) {
		delete argParser;
		argParser = nullptr;
	}

	if (firstMatrix != nullptr) {
		delete firstMatrix;
		firstMatrix = nullptr;
	}

	if (secondMatrix != nullptr) {
		delete secondMatrix;
		secondMatrix = nullptr;
	}

	if (resultMatrix != nullptr) {
		delete resultMatrix;
		resultMatrix = nullptr;
	}
	return 0;
}