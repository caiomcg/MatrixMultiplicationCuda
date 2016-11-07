#pragma once

#include <fstream>
#include <string>
#include <exception>
#include <tuple>

#include "Matrix.h"


enum MATRIX{
	FIRST,
	SECOND
};

class ArgumentParser {
private:
	std::string p_firstFilePath;
	std::string p_secondFilePath;
	std::string p_outputFilePath;

	std::tuple<int, int> prepareArray(std::ifstream& matrixFile);
	void fillArray(Matrix* matrix, std::ifstream& matrixFile);

public:
	ArgumentParser(int argc, char** argv);
	~ArgumentParser();

	Matrix* prepareMatrix(const MATRIX type);
};

