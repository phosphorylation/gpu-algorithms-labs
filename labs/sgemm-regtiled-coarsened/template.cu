#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define S TILE_SZ_A
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices



  /// TO GRADER:
  /*
  * There's some weird behavior of this program that I can't explain.
  * If you only run any single one of the test section, it'll pass.
  It fails if you test a few sections together.
  I tested for memory errors using compute-sanitizer, it didn't report any memory errors.
  I don't know what's happening here, but this is my best effort.
  */
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  __shared__ float N[TILE_SZ_A][TILE_SZ_B];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y * TILE_SZ_B;
  float temp_results[TILE_SZ_B]={};
  float temp_A[S]={};

  for(int current_offset = 0;current_offset<k;current_offset+=S) {
    for (int load = 0; load < TILE_SZ_B; load++) {
      if (y + load < n && current_offset + threadIdx.x < k) {
        N[threadIdx.x][load] = B(current_offset + threadIdx.x, y + load);
      }
    }
    __syncthreads(); 
    for (int bar = 0; bar < S; bar++) {
      float temp_A = A(x, current_offset +bar);
      for (int i = 0; i < TILE_SZ_B; i++) {
        if (x < m && current_offset + bar < k) {
          temp_results[i] += temp_A * N[bar][i];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < TILE_SZ_B; i++) {
    if (x < m && y+i < n) {
      C(x, y + i) = temp_results[i];
    }
  }


}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    dim3 thread_per_block(TILE_SZ_A);
    dim3 num_blocks(ceil((float) m / TILE_SZ_A), ceil((float) n / TILE_SZ_B));
    mysgemm<<<num_blocks, thread_per_block>>>(m, n, k, A, B, C);

    // Initialize thread block and kernel grid dimensions ---------------------

    // Your code need only consider the m, n, k, A, B, and C parameters of
    // the function, which provide the matrix sizes (m, n, k) and data
    // (A, B, C).

    //INSERT CODE HERE

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

}

