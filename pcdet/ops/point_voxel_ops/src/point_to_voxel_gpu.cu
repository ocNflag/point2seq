/*
Query points for each voxel
Written by Jiageng Mao
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "point_to_voxel_gpu.h"
#include "pv_cuda_utils.h"


__global__ void point_to_voxel_query_kernel(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                            int num_points, int num_voxels, int num_samples,
                                            const float *xyz, const int *xyz_to_vidx, int *v_map, int *v_mask) {
    /*
        xyz: [num_points, 3] xyz coordinates of points
        xyz_to_vidx: [x_max, y_max, z_max] voxel coordinates to voxel indices
        v_map: [num_voxels, num_samples] points indices for each voxel
        v_mask: [num_voxels, 1] num_points for each voxel
    */

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= num_points) return;
    float x = xyz[pt_idx * 3 + 0];
    float y = xyz[pt_idx * 3 + 1];
    float z = xyz[pt_idx * 3 + 2];                                         
    int x_idx = floor(x / x_size);                                            
    int y_idx = floor(y / y_size);                                            
    int z_idx = floor(z / z_size);                                            
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max)
        return;
    int v_idx = xyz_to_vidx[x_idx * y_max * z_max + y_idx * z_max + z_idx];
    if (v_idx < 0) return; // -1 not exist                                             
    int sample_idx = atomicAdd(v_mask + v_idx, 1); 
    if (sample_idx >= num_samples) return;
    v_map[v_idx * num_samples + sample_idx] = pt_idx;

}


void point_to_voxel_query_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                          int num_points, int num_voxels, int num_samples,
                                          const float *xyz, const int *xyz_to_vidx, int *v_map, int *v_mask){

    cudaError_t err;

    dim3 blocks(DIVUP(num_points, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    point_to_voxel_query_kernel<<<blocks, threads>>>(x_size, y_size, z_size, x_max, y_max, z_max,
                                                     num_points, num_voxels, num_samples,
                                                     xyz, xyz_to_vidx, v_map, v_mask);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
