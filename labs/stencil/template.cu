#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 30

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {
#define A0(x, y, z) A0[(z) * (nx * ny) + (y) *nx + (x)]
#define Anext(x, y, z) Anext[(z) * (nx * ny) + (y) *nx + (x)]
  __shared__ int ds_A0[TILE_SIZE][TILE_SIZE];
  int i        = blockIdx.x * blockDim.x + threadIdx.x;
  int j        = blockIdx.y * blockDim.y + threadIdx.y;
  int tx       = threadIdx.x;
  int ty       = threadIdx.y;
  if (nz >= 3) {
    int previous = A0(i, j, 0);
    int current  = A0(i, j, 1);
    int next     = A0(i, j, 2);
    for (int k = 1; k < nz - 1; k++) {
      ds_A0[tx][ty] = current;
      __syncthreads();
      if (i > 0 && i < (nx - 1) && j > 0 && j < (ny - 1)) {
        Anext(i, j, k) = previous + next +
          (tx < TILE_SIZE - 1 ? ds_A0[tx + 1][ty] : A0(i + 1, j, k)) +
          (tx > 0 ? ds_A0[tx - 1][ty] : A0(i - 1, j, k)) +
          (ty > 0 ? ds_A0[tx][ty - 1] : A0(i, j - 1, k)) +
          (ty < TILE_SIZE - 1 ? ds_A0[tx][ty + 1] : A0(i, j + 1, k)) -
          6 * current;
      }
      previous = current;
      current  = next;
      if (k < nz - 2)
        next = A0(i, j, k + 2);
      __syncthreads();
    }
  }


// INSERT KERNEL CODE HERE
#undef A0
#undef Anext
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  dim3 thread_per_block(TILE_SIZE , TILE_SIZE);
  dim3 num_blocks(ceil(static_cast<float>(nx) / thread_per_block.x), ceil(static_cast<float> (ny) / thread_per_block.y));
  kernel<<<num_blocks, thread_per_block>>>(A0, Anext, nx, ny, nz);
  // INSERT CODE HERE
}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
