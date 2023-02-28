#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)
typedef unsigned int uint;
/******************************************************************************
 GPU kernels
*******************************************************************************/
//__device__ void mutex_lock(unsigned int *mutex) {
//    unsigned int ns = 8;
//    while (atomicCAS(mutex, 0, 1) == 1) {
//        __nanosleep(ns);
//        if (ns < 256) {
//            ns *= 2;
//        }
//    }
//}
//
//__device__ void mutex_unlock(unsigned int *mutex) {
//    atomicExch(mutex, 0);
//}

__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to global queue
  int x =blockDim.x*blockIdx.x+threadIdx.x;
  for(int start = 0;start<*numCurrLevelNodes;start+=gridDim.x*blockDim.x){
      //parallelize over nodes
      if(x+start<*numCurrLevelNodes){
          int my_node = currLevelNodes[x+start];
          int my_neighbor_start = nodePtrs[my_node];
          int my_neighbor_end = nodePtrs[my_node+1];
          for(int c=my_neighbor_start;c<my_neighbor_end;c++){
              int neighbor = nodeNeighbors[c];
              // this node hasn't been visited
              if(atomicCAS(nodeVisited+neighbor,0,1)==0){
                  int my_index = atomicAdd(numNextLevelNodes,1);
                  nextLevelNodes[my_index]=neighbor;
              }
          }
      }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue (size should be BQ_CAPACITY)

  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to block queue
  // If full, add neighbor to global queue

  // Allocate space for block queue to go into global queue

  // Store block queue in global queue
    __shared__ int block_queue[BQ_CAPACITY];
    __shared__ int bqueue_count;
    __shared__ int start_fill_shared;
    if (threadIdx.x == 0){
        bqueue_count = 0;
    }
    int x =blockDim.x*blockIdx.x+threadIdx.x;  // x is per grid
    for(int start = 0;start<*numCurrLevelNodes;start+=gridDim.x*blockDim.x){
        //parallelize over nodes
        if(x+start<*numCurrLevelNodes){
            int my_node = currLevelNodes[x+start];
            for (int c = nodePtrs[my_node]; c < nodePtrs[my_node + 1]; c++) {
                int neighbor = nodeNeighbors[c];
                // this node hasn't been visited
                if (atomicCAS(nodeVisited + neighbor, 0, 1) == 0) {
                  if (bqueue_count >= BQ_CAPACITY) {
                    int my_index             = atomicAdd(numNextLevelNodes, 1);
                    nextLevelNodes[my_index] = neighbor;
                  } else {
                    int my_index          = atomicAdd(&bqueue_count, 1);
                    block_queue[my_index] = neighbor;
                  }
                }
            }
        }
        __syncthreads();
        if((bqueue_count>2000 ||start+gridDim.x*blockDim.x>=*numCurrLevelNodes)) {
            if (threadIdx.x == 0) {
                start_fill_shared = atomicAdd(numNextLevelNodes, bqueue_count);
            }
            __syncthreads();
            for (int fill = 0; fill < bqueue_count; fill += BLOCK_SIZE) {
                nextLevelNodes[start_fill_shared + fill + threadIdx.x] = block_queue[fill + threadIdx.x];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                bqueue_count = 0;
            }
            __syncthreads();
        }
    }
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses NUM_WARP_QUEUES warp queues of capacity 
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.  

  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.

  // Initialize shared memory queues (warp and block)

  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to the queue
  // If full, add neighbor to block queue
  // If full, add neighbor to global queue

  // Allocate space for warp queue to go into block queue

  // Store warp queues in block queue (use one warp or one thread per queue)
  // Add any nodes that don't fit (remember, space was allocated above)
  //    to the global queue

  // Saturate block queue counter (too large if warp queues overflowed)
  // Allocate space for block queue to go into global queue

  // Store block queue in global queue
    __shared__ int block_queue[BQ_CAPACITY];
    __shared__ int warp_queue[NUM_WARP_QUEUES][WQ_CAPACITY];
    __shared__ int bqueue_count;
    __shared__ int start_fill_shared;
    __shared__ int warp_queue_count[NUM_WARP_QUEUES];
    if (threadIdx.x == 0) {
        bqueue_count = 0;
    }
    int my_warp = threadIdx.x % NUM_WARP_QUEUES;
    if (threadIdx.x < NUM_WARP_QUEUES) {
        warp_queue_count[my_warp] = 0;
    }
    __syncthreads();
    int x = blockDim.x * blockIdx.x + threadIdx.x; // x is per grid
    for (int start = 0; start < *numCurrLevelNodes; start += gridDim.x * blockDim.x) {
        // parallelize over nodes
        if (x + start < *numCurrLevelNodes) {
            int my_node = currLevelNodes[x + start];
            for (int c = nodePtrs[my_node]; c < nodePtrs[my_node + 1]; c++) {
                int neighbor = nodeNeighbors[c];
                // this node hasn't been visited
                if (atomicCAS(nodeVisited + neighbor, 0, 1) == 0) {
                    if (warp_queue_count[my_warp] >= WQ_CAPACITY) {
                    int my_index             = atomicAdd(numNextLevelNodes, 1);
                    nextLevelNodes[my_index] = neighbor;
                    } else {
                      int my_index                  = atomicAdd(&warp_queue_count[my_warp], 1);
                      warp_queue[my_warp][my_index] = neighbor;
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < NUM_WARP_QUEUES) {
            int start_fill_warp = atomicAdd(&bqueue_count, warp_queue_count[my_warp]);
            for (int fill = 0; fill < warp_queue_count[my_warp]; fill++) {
                block_queue[start_fill_warp + fill] = warp_queue[my_warp][fill];
            }
            warp_queue_count[my_warp] = 0;
        }
        __syncthreads();
        if ((bqueue_count > 2000 || start + gridDim.x * blockDim.x >= *numCurrLevelNodes)) {
            if (threadIdx.x == 0) {
                start_fill_shared = atomicAdd(numNextLevelNodes, bqueue_count);
            }
            __syncthreads();
            for (int fill = 0; fill < bqueue_count; fill += BLOCK_SIZE) {
                nextLevelNodes[start_fill_shared + fill + threadIdx.x] = block_queue[fill + threadIdx.x];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                bqueue_count = 0;
            }
            __syncthreads();
        }
    }
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
