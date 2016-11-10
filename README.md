# MatrixMultiplicationCuda

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/caiomcg/MatrixMultiplicationCuda/master/LICENSE)

Matrix Multiplication of nXn matrixes under the NVIDIA CUDA environment.

## Requirements ##

* Visual C++
* NVIDIA CUDA

## Build Environment ##

* OS: Microsoft Windows 10 Home Single Language
* IDE: Visual Studio Community 2015
* CUDA: Release 8.0, V8.0.44
* COMPILER: Microsoft Visual C++ 2015

## Build Instructions ##

* Clone the project;
* Move to the Visual Studio Solution;
* Open the solution under Visual Studio;
* Run the code.

```
$> git clone https://github.com/caiomcg/MatrixMultiplicationCuda.git
$> cd MatrixMultiplicationCuda
$> cd MatrixMultiplicationCUDA
$> open MatrixMultiplicationCUDA.sln
```

## Execution Instructions ##

* Run the project passing the Matrix to multiply through the argv parameter.

```
$> ./MatrixMultiplicationCUDA file.txt file2.txt
```

## Example of a file ##
```
Matrix 3x3

1 2 3
1 2 3
1 2 3
```
