/*
Authors: Anant Shah and Abhishek Nair 
Roll No: EE16B105   and EE16B060
*/

/*
Vanilla Code for Convolution using for loops
*/

/* ------------------------------------ LIBRARIES ----------------------------------- */
#include<stdio.h>
#include<math.h>
#include<stdlib.h>

/* ----------------------------------- UTILITY FUNCTIONS ---------------------------- */

void fillMatrix(float *matrix,int X,int Y,int D,int N,int sparsity_perc,int S_BLOCK){
	float (*m)[X][Y][D]=(float (*)[X][Y][D])matrix;
	for(int n=0;n<N;n++){
		for(int i=0;i<X;i++){
			for(int j=0;j<Y;j++){
				for(int k=0;k<D;k++){
                			//if(((n*X*Y+i*Y+j)/S_BLOCK)%2==0)                        m[n][i][j][k] = 0.0; /* For the case of 50% sparsity */
					if((i/S_BLOCK+j/S_BLOCK) < ((float)sparsity_perc/100)*((X/S_BLOCK)*(Y/S_BLOCK)))			m[n][i][j][k] = 0.0;
					else                                        			                                m[n][i][j][k]=n*X*Y*D+i*Y*D+j*D+k;
				}
			}
		}
	}
}

/* ***************************************************************************************** */

void fillKernel(float *kernel,int K_X,int K_Y,int D,int K_N){

    float (*t)[K_X][K_Y][D]=(float (*)[K_X][K_Y][D])kernel;

    for(int i=0;i<K_N;i++){
	    for(int j=0;j<K_X;j++){
		    for(int k=0;k<K_Y;k++){
			    for(int l=0;l<D;l++){
                    if(i%2==0)      t[i][j][k][l] = 1;
                    else            t[i][j][k][l] = 1;
				}
			}
		}
	}
}
/* ***************************************************************************************** */

void print_matrix_to_file(float *m,int O_X,int O_Y,int K_N,int N){

	const char *fname = "Block_Sparse_Serial_Out";
	FILE *f = fopen(fname, "w");

	float (*mat)[O_X][O_Y][K_N]=(float (*)[O_X][O_Y][K_N])m;		

	for(unsigned i=0; i<N; i++){
		for(unsigned j=0; j<O_X; j++)
			for(unsigned k=0;k<O_Y;k++)
				for(unsigned l=0;l<K_N;l++)
					fprintf(f,"%4.4f ", mat[i][j][k][l]);
		fprintf(f,"\n");
    }
	fclose(f);
}
/* ***************************************************************************************** */


int main(int argc,char **argv)
{

	/*********************** Input from the command line ********************/
	if(argc!=10){
		printf("error : Invalid number of arguments\n");
		exit(EXIT_FAILURE);
	}

	int 	N = atoi(argv[1]); /* X-dimension of the 3-D signal */
	int 	X = atoi(argv[2]); /* Y-dimension of the 3-D signal */
	int 	Y = atoi(argv[3]); /* Depth of the 3-D signal  */
	int 	D = atoi(argv[4]); /* X-dimension of the kernel */
	int 	K_X = atoi(argv[5]); /* Y-dimension of the kernel */
	int 	K_Y = atoi(argv[6]); /* Depth in the 4th dimension */
	int 	K_N = atoi(argv[7]); /* Number of kernel functions */
	int	    sparsity_perc = atoi(argv[8]); /* Sparsity percentage in the tensor */
    int     S_BLOCK = atoi(argv[9]); /* Height and width of a sparse block */

	/************************************************************************/

    if(((X%S_BLOCK)!=0) || ((Y%S_BLOCK)!=0)){
        printf("error : The image dimensions have to be a multiple of the sparse block size");
        exit(EXIT_FAILURE);
    }

	int	O_X = X-(K_X-1); /* X-dimension of the output */
	int	O_Y = Y-(K_Y-1); /* Y-dimension of the output */
	
	float *matrix=(float *)malloc(sizeof(float)*X*Y*D*N);
	float *kernel=(float*)malloc(sizeof(float)*K_X*K_Y*D*K_N);
	float *output=(float *)malloc(sizeof(float)*O_X*O_Y*N*K_N);

	fillMatrix(matrix,X,Y,D,N,sparsity_perc,S_BLOCK);
	fillKernel(kernel,K_X,K_Y,D,K_N);
	
    for(int i=0;i<N;i++){
		for(int j=(K_X-1)/2;j<X-(K_X-1)/2;j++){
			for(int k=(K_Y-1)/2;k<Y-(K_Y-1)/2;k++){
				for(int l=0;l<K_N;l++){
					for(int m=0;m<K_X;m++){
						for(int n=0;n<K_Y;n++){
							for(int o=0;o<D;o++){
								output[i*O_X*O_Y*K_N+(j-(K_X-1)/2)*K_N*O_Y+(k-(K_Y-1)/2)*K_N+l] += matrix[i*X*Y*D+(j-(K_X-1)/2+m)*Y*D+(k-(K_Y-1)/2+n)*D+o] * kernel[l*K_X*K_Y*D+m*K_Y*D+n*D+o];
							}
						}
					}
				}
			}
		}
	}
	
	 
	print_matrix_to_file(output,O_X,O_Y,K_N,N);

	free(matrix);
	free(kernel);
	free(output);
}
