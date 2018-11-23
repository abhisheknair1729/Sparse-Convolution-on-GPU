# Block Sparse Tensor Convolution

This project deals with block sparse tensor convolutions and its implementation on a GPU. Sparse convolutions have been implemented for 4-D tensors, with the tensor dimensions, kernel dimensions and the sparsity specifications all being user defined. The code has been written in CUDA C and implemented on a TESLA K40C NVIDIA GPU.

The repository contains

* a CUDA C implementation of block sparse tensor convolutions for 4-D tensors
* a C serial implementation of block sparse tensor convolution for 4-D tensors
* a Python implementation of block sparse tensor convolution (code written by Uber, used by us for benchmarking) 

The serial code is used to verify correctness while the Uber SBNet code is used to compare performance.

## Requirements

Code was tested and compiled for NVIDIA CUDA 9.0 architecture(Tesla K40C). To run the Uber SBNet code, please refer to their github page for details on implementation.

## Implementation

### CUDA C Code

nvcc compiler required to compile the code. If running on IITM servers, use the gsub command to run your script. Once the executable is created, run the code with the following command line inputs :

* N - number of images(batch-size of the 4th dimension)
* H - height of an image(size of the 1st dimension)
* W - width of an image(size of the 2nd dimension)
* C - number of channels in an image(size of the 3rd dimension)
* K\_H - height of the kernel
* K\_W - width of the kernel
* K\_N - number of kernels
* sparsity\_perc - sparsity percentage in the input tensor
* S\_BLOCK - width and height of a sparse block

Given below is an example of how to run the code with command line inputs.

```
nvcc -o blocksparsematmul blocksparsematmul.cu
./blocksparsematmul 4 256 256 3 3 3 1 70 16
```

Once the code runs, the output tensor is written into a file named Block\_Sparse\_Convolution\_Output.

Listed below are a few guidelines which need to be followed to get a working code :

1. The height and width(H&W) of the image has to be a multiple of the sparse block size(S\_BLOCK).
2. The sparsity percentage(sparsity\_perc) has to be less than 100.
3. The image dimensions(H&W) should not exceed 512.

### C Serial Code

gcc compiler required to compile the code. The instructions to run the serial code are same as those mentioned for the CUDA C code. Given below is an example of how to run the serial code with command line inputs :

```
gcc -o serial serial.c
./serial 4 256 256 3 3 3 1 80 16
```

The constraints for the serial code are same as that of the CUDA C code.

### Uber SBNet Code

For requirements, please refer to the uber/sbnet github page for details. If running on IITM servers, no changes are required in the code. Use the gsubtf command to run the scripts. If running on a PC, load the sbnet module from its location on your PC. The command line input definitions are listed below :

* batch - Number of images(size of the 4th dimension)
* hw - Height and Width of an image(size of the 1st and 2nd dimension)
* channels - number of channels in the image(size of the 3rd dimension)
* sparsity - Percentage sparsity in the tensor
* kw - Kernel height and width
* kn - Number of kernels
* blocksize - width and height of a sparse block

```
python sbnet.py 4 256 3 50 3 1
```

Listed below are a few constaints of the SBNet code :
* The height and the width dimensions have to be the same, hence only one command line argument for them
* The kernel height and width have to be the same, hence only once command line argument for them

### Sample Script for IITM Servers

For IITM servers, run the script below with gsubtf command so as to run the Uber SBNet code.

```
cd /$TO_DIRECTORY_WHERE_CODE_IS$/
./blocksparse 4 256 256 3 3 3 1 70 16
python sbnet.py 4 256 3 50 3 1
```

##  Authors 

1. Abhishek Nair(EE16B060)
2. Anant Shah(EE16B105)
