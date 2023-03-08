#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <math.h>

#include "template.hu"

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; // x is per grid
    for (int start = 0; start < numEdges; start += gridDim.x * blockDim.x) {
        uint32_t U = edgeSrc[start+x];
        uint32_t V = edgeDst[start+x];
        uint32_t start_u = rowPtr[U];
        uint32_t start_v = rowPtr[V];
        uint32_t total_set_size = (rowPtr[U+1]-start_u)+(rowPtr[V+1]-start_v)-2;
        uint64_t temporal_sum = 0;
        start_u++;
        start_v++;
        uint32_t advance_u = edgeDst[start_u];
        uint32_t advance_v = edgeDst[start_v];
        while(total_set_size>1){
            if(advance_u>advance_v){
                start_v++;
                advance_v = edgeDst[start_v];
                total_set_size--;
            }else if(advance_u<advance_v){
                start_u++;
                advance_u = edgeDst[start_u];
                total_set_size--;
            }else{
                temporal_sum++;
                start_v++;
                advance_v = edgeDst[start_v];
                start_u++;
                advance_u = edgeDst[start_u];
                total_set_size-=2;
            }
        }
        triangleCounts[start+x]=temporal_sum;
    }

  // Determine the source and destination node for the edge

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array

  // Determine how many elements of those two arrays are common
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
  uint64_t result = 0;
  uint64_t total = view.nnz();
  std::vector<uint64_t> triangles(total);
  uint64_t * cuda_mem;

  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  dim3 dimGrid (math.ceil(total/512f));
  if (mode == 1) {
    //@@ launch the linear search kernel here
      CUDA_RUNTIME(cudaMalloc((void **)&cuda_mem,
                            total * sizeof(uint64_t)));
      CUDA_RUNTIME(cudaDeviceSynchronize());
      kernel_tc<<<dimGrid, dimBlock>>>(cuda_mem,view.row_ind(),view.col_ind(),view.row_ptr(),total);
      CUDA_RUNTIME(cudaDeviceSynchronize());
      CUDA_RUNTIME(cudaMemcpy(triangles.data(), cuda_mem,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost));

  } else if (2 == mode) {

    //@@ launch the hybrid search kernel here
    // your_kernel_name_goes_here<<<dimGrid, dimBlock>>>(...)

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count

  for(auto edge:triangles){
      result+=edge;
  }
  return result;
}
