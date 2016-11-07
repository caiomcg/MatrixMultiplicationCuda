/**
 * @class ArgumentParser
 *
 * @brief Argument Parser Class
 *
 * This class is responsible for reading the argv input
 * and prepare the matrix according to that information.
 *
 * @author Caio Marcelo Campoy Guedes <caiomcg@gmail.com>
 */

#pragma once

#include <fstream> //File read.
#include <string> //std::string.
#include <exception> //Exceptions.
#include <tuple> //std::tuple.

#include "Matrix.h" //Matrix.


enum MATRIX{
	FIRST,
	SECOND
};

class ArgumentParser {
private:
	std::string p_firstFilePath; //First matrix file path.
	std::string p_secondFilePath; //Second matrix file path.
	std::string p_outputFilePath; //Result matrix file path(not in use).

	/**
	 * @brief Iterate through the file and find the array dimensions.
	 * 
	 * @param matrixFile The file where the matrix is located.
	 * @return The width and height of the matrix.
	 */
	std::tuple<int, int> prepareArray(std::ifstream& matrixFile);

	/**
	 * @brief Fill the arrays.
	 * @details Arrays are filled with the data that is located under.
	 * the files passed as arguments.
	 * 
	 * @param matrix Matrix to be filled.
	 * @param matrixFile File to take the data from.
	 */
	void fillArray(Matrix* matrix, std::ifstream& matrixFile);

public:

	/**
	 * @brief Class constructor.
	 * 
	 * @param argc Amount of arguments received at the command line.
	 * @param argv The data received at the command line.
	 */
	ArgumentParser(int argc, char** argv);

	/**
	 * @brief Class destructor.
	 */
	~ArgumentParser();

	/**
	 * @brief Prepare an object of type matrix.
	 * @details Instantiate the object and initialize it with.
	 * the information located at the input file.
	 * 
	 * @param type The type of matrix.
	 * @return A pointer to a Matrix type allocated in memory.
	 */
	Matrix* prepareMatrix(const MATRIX type);
};

