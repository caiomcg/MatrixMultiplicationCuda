#include "ArgumentParser.h"

ArgumentParser::ArgumentParser(int argc, char** argv) {
	if (argc == 1) {
		throw std::invalid_argument("Argc must contain more than one argument. Please check the usage.");
	}

	try {
		p_firstFilePath = argv[1];
		p_secondFilePath = argv[2];
	} catch (...) {
		throw std::range_error("Arguments not found in argv.");
	}

	p_outputFilePath = "out.txt";
}


ArgumentParser::~ArgumentParser() {
}

std::tuple<int, int> ArgumentParser::prepareArray(std::ifstream& matrixFile) {
	int line = 0;
	int collumn = 0;

	std::string info;
	size_t pos = 0;

	std::getline(matrixFile, info);

	if ((pos = info.find(" ")) != std::string::npos) {
		info.erase(0, pos + 1);
		if ((pos = info.find("x")) != std::string::npos) {
			line = stoi(info.substr(0, pos));
			collumn = stoi(info.erase(0, pos + 1));
		} else {
			return std::make_tuple(line, collumn);
		}
	} else {
		return std::make_tuple(line, collumn);
	}

	return std::make_tuple(line, collumn);
}

void ArgumentParser::fillArray(Matrix* matrix, std::ifstream& matrixFile) {
	for (int i = 0; i < matrix->getWidth(); i++) {
		for (int j = 0; j < matrix->getHeight(); j++) {
			matrixFile >> matrix->getElements()[i*matrix->getWidth() + j];
		}
	}
}

Matrix* ArgumentParser::prepareMatrix(const MATRIX type) {
	std::ifstream input(type == MATRIX::FIRST ? p_firstFilePath : p_secondFilePath);
	if (!input.is_open()) {
		return nullptr;
	}

	auto tuple = prepareArray(input);

	if (!std::get<0>(tuple) && !std::get<1>(tuple)) {
		return nullptr;
	}

	Matrix* matrix = new Matrix(std::get<0>(tuple), std::get<1>(tuple));
	fillArray(matrix, input);

	input.close();

	return matrix;
}
