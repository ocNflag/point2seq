/*
Query voxel centers for each point
Written by Jiageng Mao
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "fill_dense_gpu.h"
#include "pv_cuda_utils.h"


__global__ void fill_dense_kernel(int x_max, int y_max, int z_max, int num_voxels, const int *v_indices, int *xyz_to_vidx) {
    /*
        v_indices: [num_voxels, 3] zyx indices of voxels
        xyz_to_vidx: [x_max, y_max, z_max] voxel coordinates to voxel indices
    */

    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= num_voxels) return;
    int z_idx = v_indices[v_idx * 3 + 0]; 
    int y_idx = v_indices[v_idx * 3 + 1]; 
    int x_idx = v_indices[v_idx * 3 + 2]; 
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx[x_idx * y_max * z_max + y_idx * z_max + z_idx] = v_idx;
}


void fill_dense_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, const int *v_indices, int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    fill_dense_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, v_indices, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
