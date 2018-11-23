/*
Authors: Anant Shah and Abhishek Nair
Roll No: EE16B105 and EE16B060
*/

/*
Code to implement the convolution operation on images with a regular sparsity pattern.
*/

/* --------------------------- CONSTRAINTS ----------------------------------
 
Input Tensor Format            : N x H x W x C
Input Tensor Size Limit        : 512 x 512 x C
Batch Size                     : Multiple of 2

*/

/*  ---------------------------  LIBRARIES  ---------------------------------- */
#include<stdio.h>
#include<cuda.h>

/*  ---------------------------  GLOBAL PARAMETERS  -------------------------- */
#define BLOCK_SIZE 16
#define BLOCK_DIM_Z 4
#define TILE_WIDTH 16


/* Macro to check for errors in calls to the CUDA API */
#define checkError(e) {                                          \
  if(e!=cudaSuccess) {                                              \
     printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(0); \
  }                                                                 \
}


/*  --------------------------  CUDA KERNELS  ----------------------------------- */

__device__ bool checksparse(int *d_mask,int cx,int cy,int ni,int H,int W,int K_H,int K_W,int S_BLOCK){

    /* Kernel Function to check if an element in the tensor lies within a sparse block or not. We do this by comparing the realtive position of the element in the mask matrix */

    /* coordinates of this thread in the mask array */
    int s_x = cx/S_BLOCK; /* cx is the x-coordinate of the element in the image */
    int s_y = cy/S_BLOCK; /* cy is the y-coordinate of the element in the image */
    int x = 0;
    int y = 0;
    
    /* Check if all the elements surrounding the center element are zero by comparing it with the mask values. We check those surrounding elements which are valid for convolution with the kernel */
    for( int l=-(K_H-1)/2; l <= (K_H-1)/2; l++ )
        for( int p=-(K_W-1)/2; p <= (K_W-1)/2; p++ ){
            x = cx + l;
            y = cy + p;
            s_x = x/S_BLOCK; /* x-coordinate of block in the mask which the element belongs to */
            s_y = y/S_BLOCK; /* y-coordinate of block in the mask which the element belongs to */
            if( d_mask[ni*(H/S_BLOCK)*(W/S_BLOCK)+s_x*(W/S_BLOCK)+s_y] == 1 ){
                return false;
            }
    }
    return true;
}

/* ***************************************************************************************** */

__global__ void im2col(float *d_tensor,  int *d_mask, float *d_mat, unsigned int *row_address, int* d_row_map, int num_images,int N,int H,int W,int C,int K_N,int K_H,int K_W,int S_BLOCK){
    
    /* Kernel Function to convert the sparse tensor to a matrix. We create the matrix such that all the surrounding elements required for convolution are unrolled into a row. We ignore those elements which are completely surrounded by zeros. These element are found with the help of the checksparse() function. Once we store the elements in a matrix, we need a map which stores the coordinates and image id from which that row was produced. This map will be useful when we need to convert the matrix back into the tensor form after multiplication. */

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int gx = bx*blockDim.x+tx;
    int gy = by*blockDim.y+ty;
    int gz = bz*blockDim.z+tz;

    //checking for validity of thread
    if(gz<num_images){
        if( (gx < H-K_H+1) && (gy < W-K_W+1)  ){
            int centerx = gx + (K_H-1)/2; /* Corresponding x-coordinate in the acutal tensor */
            int centery = gy + (K_W-1)/2; /* Corresponding y-coordinate in the actual tensor */
            /* Check if the element needs to be added into the matrix or not */
            if( !checksparse(d_mask,centerx,centery,gz,H,W,K_H,K_W,S_BLOCK) ){
                unsigned int row_index = atomicAdd( row_address, 1 );  /* Update the number of rows in the matrix */
                int col_index = 0; /* Initialize the column index to 0 as the elements are being added along a row */
            	
                /* Load the corresponding elements required for convolution into the matrix */
                for( int l=-(K_H-1)/2; l <= (K_H-1)/2; l++ )
                    for( int p=-(K_W-1)/2; p <= (K_W-1)/2; p++ )
                        for( int q=0; q < C; q++){
                            /* mat_val = mask_val?mat_val:0 */
                            d_mat[row_index*K_H*K_W*C+col_index] = d_mask[gz*(H/S_BLOCK)*(W/S_BLOCK)+((int)((centerx+l)/S_BLOCK))*(W/S_BLOCK)+((int)((centery+p)/S_BLOCK))]?d_tensor[gz*H*W*C+(centerx+l)*W*C+(centery+p)*C+q]:0;
                            col_index += 1;
                        }
                d_row_map[row_index*3+0] = gx; /* Store the original x-coordinate corresponding to a row into a map */
                d_row_map[row_index*3+1] = gy; /* Store the original y-coordinate corresponding to a row into a map */
                d_row_map[row_index*3+2] = gz; /* Store the image corresponding to a row in a map */
            }
        }  
    }
}

/* ***************************************************************************************** */

__global__ void col2im(float *o_mat, int *d_row_map, unsigned int *row_num, float *op_img,int H,int W,int K_N,int K_H,int K_W){
    /* Kernel Function to convert the matrix multiplication output back to the tensor. We will be utilizing the map genereated earlier to reorder the elements appropiately */

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int gx = bx*blockDim.x+tx;
    int gy = by*blockDim.y+ty;
    
    int ystride = blockDim.y*gridDim.y;

    /* Iterate over the relevant rows in the output to store it into the output tensor */
    while( gy < *row_num ){
    //validity check
        if( gx < K_N ){    
            op_img[d_row_map[gy*3+2]*(H-K_H+1)*(W-K_W+1)*(K_N) + d_row_map[gy*3+0]*(W-K_W+1)*(K_N) + d_row_map[gy*3+1]*(K_N) + gx] = o_mat[gy*K_N+gx];
        }
        gy += ystride;
    }
}

/* ***************************************************************************************** */

__global__ void sparsematmul(float *image_mat,float *kernel_mat,unsigned int *width,int num_kernels,float *output_mat,int K_H,int K_W,int C){

    /* Kernel Function to perform matrix multiplication. We utilize shared memory and tiling to perform the matrix multiplication. */

    int tx = threadIdx.x;  /*Thread-ID in the x-direction */
    int ty = threadIdx.y;  /*Thread-ID in the y-direction */
    __shared__ float image_mat_s[TILE_WIDTH][TILE_WIDTH]; /* Shared memory to be used by threads in a block */    
    __shared__ float kernel_mat_s[TILE_WIDTH][TILE_WIDTH]; /* Shared memory to be used by therads in a block */

    int row = blockIdx.y*blockDim.y + ty; /* row in the output matrix */
    int col = blockIdx.x*blockDim.x + tx; /* column in the output matrix */
    float pSum = 0.0;

    for(int m=0;m<(K_W*K_H*C+TILE_WIDTH-1)/TILE_WIDTH;m++){
    /* Load Pahse : Load elements cooperatively into the shared memeory */
        if(row<*width && (m*TILE_WIDTH+tx)<(K_W*K_H*C) )     image_mat_s[ty][tx] = image_mat[row*K_W*K_H*C+m*TILE_WIDTH+tx];
        if( (ty+m*TILE_WIDTH)<(K_W*K_H*C) && col<num_kernels )      kernel_mat_s[ty][tx] = kernel_mat[(ty+m*TILE_WIDTH)*num_kernels+col]; /* This is assuming that the tile is a sqaure */
        __syncthreads();        

        for(int i=0;i<TILE_WIDTH;i++){
            pSum += image_mat_s[ty][i]*kernel_mat_s[i][tx];
        }
        __syncthreads();
        image_mat_s[ty][tx] = 0.0; /* Setting the elements in the shared memory back to 0. This takes care of the corner cases where junk values are stored */
        kernel_mat_s[ty][tx] = 0.0; 
    }
    if(row<*width && col<num_kernels)    output_mat[row*num_kernels+col] = pSum; /* Load the result into the output matrix */
}

/* -----------------------------  Utility Functions ------------------------------------------- */

void fillTensor(float *tensor,int N,int H,int W,int C){
    /* Utility function to fill the tensor */
    float (*ctensor)[H][W][C] = (float (*)[H][W][C])tensor;
    for(int i=0;i<N;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                for(int l=0;l<C;l++){
                    ctensor[i][j][k][l] = i*W*H*C+j*W*C+k*C+l;//(i*l+0.2*j+0.4*k+2*l);
                }
            }
        }
    }
}
/* ***************************************************************************************** */

void fillKernel(float *h_kernel_mat,int K_N,int K_H,int K_W,int C){
    /* Utility function to fill the kernel */
    for(int i=0;i<K_H*K_W*C;i++){
        for(int j=0;j<K_N;j++){
            if(j%2 == 0)  h_kernel_mat[i*K_N+j] = 1;//(i*j+0.2*i+0.3*j);
            else h_kernel_mat[i*K_N+j] = 1;
        }
    }
}
/* ***************************************************************************************** */

void fillMask(int *mask,int N,int H,int W,int S_BLOCK,int sparsity_perc){
    /* Utility function to define the mask. Based on the sparsity, the first x elements in each image are set to 0 */
    for(int i=0;i<N;i++){
        for(int j=0;j<(H/S_BLOCK);j++){
            for(int k=0;k<(W/S_BLOCK);k++){
                if((j+k) < ((float)sparsity_perc/100)*((H/S_BLOCK)*(W/S_BLOCK)))   mask[i*(H/S_BLOCK)*(W/S_BLOCK)+j*(W/S_BLOCK)+k] = 0;
                else                                                            mask[i*(H/S_BLOCK)*(W/S_BLOCK)+j*(W/S_BLOCK)+k] = 1;
            }
        }
    }
}
/* ***************************************************************************************** */

void print_tensor(float *h_out_tensor,int N,int H,int W,int K_N,int K_H,int K_W){
    /* Utility function to print the output tensor to a file */
    FILE *fp = fopen("Block_Sparse_Convolution_Output","w");
    for(int i=0;i<N;i++){
        for(int j=0;j<(H-K_H+1);j++){
             for(int k=0;k<(W-K_W+1);k++){
                for(int l=0;l<K_N;l++){
                    fprintf(fp,"%4.4f ",h_out_tensor[i*(H-K_H+1)*(W-K_W+1)*K_N + j*(W-K_W+1)*K_N + k*K_N + l]);
                }
             }
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}



int main(int argc,char **argv){
   
    if(argc!=10){
        printf("error : Invalid number of arguments\n");
        printf("Format : ./exec num_images img_height img_width img_channels kernel_height kernel_width num_kernels sparsity_percentage sparsity_block_size\n");
        exit(EXIT_FAILURE);
    }

    int     N = atoi(argv[1]); /* Number of images */
    int     H = atoi(argv[2]); /* Height of one image */
    int     W = atoi(argv[3]); /* Width of one image */
    int     C = atoi(argv[4]); /* Number of channels in the image */
    int     K_H = atoi(argv[5]); /* Height of the kernel */
    int     K_W = atoi(argv[6]); /* Widht of the kernel */
    int     K_N = atoi(argv[7]); /* Number of kernels */
    int     sparsity_perc = atoi(argv[8]); /* Sparsity percentage in the mask */
    int     S_BLOCK = atoi(argv[9]); /* Size of a sparse block */

    float           *tensor; /* Pointer to the tensor on the host */
    checkError(cudaHostAlloc( (void **) &tensor, sizeof(float)*N*H*W*C, 0 )); /* Allocating pinned memory on the CPU so as to facilitate streams */
    float           *kernel = (float *)malloc(sizeof(float)*K_N*K_H*K_W*C); /* Allocate memory for the kernel on the CPU  */
    float           *h_out_tensor = (float *)malloc(sizeof(float)*N*(W-K_W+1)*(H-K_H+1)*K_N) ; // A pointer to the output tensor on the host 
    int             num_images = 2; /* Number of images handled by each stream */

    if( (H%S_BLOCK != 0) && (W%S_BLOCK != 0)){
        printf("Invalid dimensions for input and sparsity blocks. Input tensor dimensions must be integral multiples of sparsity block dimensions\n");
        exit(EXIT_FAILURE);
    }
  
    if( N%num_images!=0 ){
        printf("error : Number of images is not a multiple of that handled by each stream. Please enter a mulitple of %d as number of images \n",num_images);
        exit(EXIT_FAILURE);
    }

    int     *mask; /* Define a pointer to the mask */
    checkError( cudaHostAlloc((void **)&mask, sizeof(float)*N*(H/S_BLOCK)*(W/S_BLOCK), 0)); /* Allocate pinned memory for the mask to facilitate streams */
    
    fillTensor(tensor,N,H,W,C); /* Fill the tensor from the syntehtic input */
    fillKernel(kernel,K_N,K_H,K_W,C); /* Fill the kernel from the synthetic input */
    fillMask(mask,N,H,W,S_BLOCK,sparsity_perc); /* Fill the mask array */
    
    float           *d_tensor; /* A pointer to the tensor on the device */
    float           *d_kernel; /* A pointer to the kernel on the device */
    float           *d_ker_mat;  //the matrix for the kernel generated by im2col_ker
    int             *d_mask; /* A pointer to the mask on the device */
    float           *d_mat;    // d_mat is the matrix used for matrix multiplication
    int             *d_row_map; /* A pointer to the map array; which holds the coordinate ids for each row in the matrix */
    float           *d_out_mat; // Output matrix 
    float           *d_out_tensor; // Pointer to the output tensor on the device 
    unsigned int    *row_address; /* A pointer to keep a track of how many rows have been added to the matrix per stream  */
    int             NUM_STREAMS = (N+num_images-1)/num_images; /* Number of streams defined to handle all the images */
    cudaEvent_t     start,stop; /* CUDA events to time the program */

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    /*********** Allocate memory on the device for each of the matrices ***************/

    checkError(cudaMalloc((void **)&d_tensor, sizeof(float)*N*H*W*C ));
    checkError(cudaMalloc((void **)&d_kernel, sizeof(float)*K_N*K_H*K_W*C)); 
    checkError(cudaMalloc((void **)&d_mask, sizeof(int)*N*(H/S_BLOCK)*(W/S_BLOCK) ));
    //allocating data for the entire dense matrix
    //need to find a way to know how many columns there will be in the matrix
    checkError(cudaMalloc((void **)&d_mat, sizeof(float)*K_H*K_W*C*(H-K_H+1)*(W-K_W+1)*N));
    checkError(cudaMalloc((void **)&d_row_map, sizeof(int)*(H-K_H+1)*(W-K_W+1)*N*3));
    checkError(cudaMalloc((void **)&d_out_tensor,sizeof(float)*N*(H-K_H+1)*(W-K_W+1)*K_N));
    //allocating memory for N row_addresses
    checkError(cudaMalloc((void **)&row_address, sizeof(unsigned int)*NUM_STREAMS));
    checkError(cudaMalloc((void **)&d_ker_mat, sizeof(float)*K_N*K_H*K_W*C));
    checkError(cudaMemcpy( d_kernel, kernel, sizeof(float)*K_N*K_H*K_W*C, cudaMemcpyHostToDevice ));
    checkError(cudaMemset( d_mat, 0, sizeof(float)*K_H*K_W*C*(H-K_H+1)*(W-K_W+1)*N ));
    checkError(cudaMemset( row_address, 0, sizeof(unsigned int)*NUM_STREAMS ));
    checkError(cudaMalloc( (void **)&d_out_mat, sizeof(float)*K_N*(H-K_H+1)*(W-K_W+1)*N));

    /**********************************************************************************/

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_DIM_Z); /* Block dimensions for the im2col operation */
    dim3 matthreads(BLOCK_SIZE,BLOCK_SIZE,(num_images+BLOCK_DIM_Z-1)/BLOCK_DIM_Z); /* Grid dimensions for the im2col operation  */
    dim3 blocks((H+BLOCK_SIZE-1)/BLOCK_SIZE,(W+BLOCK_SIZE-1)/BLOCK_SIZE); /*Grid definition for the im2col kernel operation */
    dim3 matblocks((K_N+BLOCK_SIZE-1)/BLOCK_SIZE,((H-K_H+1)*(W-K_W+1)*num_images+BLOCK_SIZE-1)/BLOCK_SIZE); // 2-D grid dimension to perform the matrix multiplication 
 
    cudaStream_t streams[NUM_STREAMS]; /* Declaring a set of CUDA streams */
    for( int i=0; i<NUM_STREAMS; i++ ) cudaStreamCreate(&streams[i]); /* Initializing a set of streams to work on a set of each image */

    if(N<num_images)    num_images = N; /* To take care of the case of illegal memory access if the number of images are less than that assigned to each stream */

    /**************** Initialize a set of offsets for the streams *****************/

    int offset = 0;
    int mask_offset = 0;
    int mat_offset = 0;
    int map_offset = 0;
    int o_offset = 0;
    int om_offset = 0;

    /*****************************************************************************/

    for(int j=0; j<NUM_STREAMS; j++){

        /* Initialize a set of off-sets for each stream */
        offset = j*H*W*C*num_images; 
        mask_offset = j*(H/S_BLOCK)*(W/S_BLOCK)*num_images;
        mat_offset = K_H*K_W*C*(H-K_H+1)*(W-K_W+1)*j*num_images;
        map_offset = 3*(H-K_H+1)*(W-K_W+1)*j*num_images;
        o_offset = (H-K_H+1)*(W-K_W+1)*K_N*j*num_images;
        om_offset = K_N*(H-K_H+1)*(W-K_W+1)*j*num_images;

        /* Asynchronously copy the images and kernels to the device */
        checkError(cudaMemcpyAsync( &d_tensor[offset], &tensor[offset], sizeof(float)*H*W*C*num_images, cudaMemcpyHostToDevice, streams[j] ));
        checkError(cudaMemcpyAsync( &d_mask[mask_offset], &mask[mask_offset], sizeof(int)*(H/S_BLOCK)*(W/S_BLOCK)*num_images, cudaMemcpyHostToDevice, streams[j] ));
        /* Transform the tensor into the matrix form  */
        im2col<<<blocks, threads, 0, streams[j]>>>(&d_tensor[offset],&d_mask[mask_offset], &d_mat[mat_offset], &row_address[j], &d_row_map[map_offset], num_images,N,H,W,C,K_N,K_H,K_W,S_BLOCK);
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess ) {
            printf("Error in launching kernel1 in stream %d\n",j);
        }
        checkError(error);

        /* Perform the matrix multiplication  */
        sparsematmul<<<matblocks,matthreads, 0, streams[j]>>>(&d_mat[mat_offset], d_kernel, &row_address[j], K_N, &d_out_mat[om_offset],K_H,K_W,C);
        error = cudaGetLastError();
        if( error != cudaSuccess ) {
                printf("Error in launching kernel 2 in stream %d\n",j);
        }
        checkError(error);

        /* Convert the matrix multiplication output back to the tensor */
        col2im<<<matblocks,matthreads, 0 , streams[j]>>>(&d_out_mat[om_offset], &d_row_map[map_offset], &row_address[j], &d_out_tensor[o_offset],H,W,K_N,K_H,K_W);
        error = cudaGetLastError();
        if( error != cudaSuccess ) {
            printf("Error in launching kernel 3 in stream %d\n",j);
        }
        checkError(error);
        
        /* Asynchronously copy the tensor from the device to the host */
        checkError(cudaMemcpyAsync(&h_out_tensor[o_offset], &d_out_tensor[o_offset], sizeof(float)*(H-K_H+1)*(W-K_W+1)*K_N*num_images, cudaMemcpyDeviceToHost,streams[j]));
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    print_tensor(h_out_tensor,N,H,W,K_N,K_H,K_W);

    float   run_time = 0.0;
    cudaEventElapsedTime(&run_time,start,stop);
    //printf("%d %d %d %d %d %d %d %d %.4f\n",N,H,W,C,K_N,K_H,K_W,sparsity_perc,run_time);
    
    printf("Output : Block_Sparse_Convolution_Output\nRunning Time : %f (ms) \nConfiguration Parameters ::\nBatch Size : %d\nImage Height : %d\nImage Width : %d\nNo. of Channels : %d\nKernel Height : %d\nKernel Width : %d\nNo. of Kernels : %d\nPercentage Sparsity : %d\n ",run_time,N,H,W,C,K_H,K_W,K_N,sparsity_perc);

    for( int i=0; i<NUM_STREAMS; i++ ) cudaStreamDestroy(streams[i]);

    return 0;
}
