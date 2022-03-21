/*
Query voxel centers for each point
Written by Jiageng Mao
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "voxel_to_point_gpu.h"
#include "pv_cuda_utils.h"


__global__ void voxel_to_point_query_kernel(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                            int num_points, int num_voxels, int num_samples, int step_range,
                                            const float *xyz, const int *xyz_to_vidx, int *p_map, int *p_mask) {
    /*
        xyz: [num_points, 3] xyz coordinates of points
        xyz_to_vidx: [x_max, y_max, z_max] voxel coordinates to voxel indices
        p_map: [num_points, num_samples] points indices for each voxel
        p_mask: [num_points, 1] num_points for each voxel
    */

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= num_points) return;
    float x = xyz[pt_idx * 3 + 0];
    float y = xyz[pt_idx * 3 + 1];
    float z = xyz[pt_idx * 3 + 2];
    float x_res = x / x_size - 0.5;
    float y_res = y / y_size - 0.5;
    float z_res = z / z_size - 0.5;                                   
    int x_f_idx = floor(x_res);                                            
    int y_f_idx = floor(y_res);                                            
    int z_f_idx = floor(z_res);
    int x_c_idx = ceil(x_res);                                            
    int y_c_idx = ceil(y_res);                                            
    int z_c_idx = ceil(z_res);
    for (int x_idx = x_f_idx - step_range + 1; x_idx <= x_c_idx + step_range - 1; ++x_idx){
        for (int y_idx = y_f_idx - step_range + 1; y_idx <= y_c_idx + step_range - 1; ++y_idx){
            for (int z_idx = z_f_idx - step_range + 1; z_idx <= z_c_idx + step_range - 1; ++z_idx){
                if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) continue;
                int v_idx = xyz_to_vidx[x_idx * y_max * z_max + y_idx * z_max + z_idx];
                if (v_idx < 0) continue; // -1 not exist
                int sample_idx = atomicAdd(p_mask + pt_idx, 1);
                if (sample_idx >= num_samples) return;
                p_map[pt_idx * num_samples + sample_idx] = v_idx;
            }
        }
    }

}


void voxel_to_point_query_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                          int num_points, int num_voxels, int num_samples, int step_range,
                                          const float *xyz, const int *xyz_to_vidx, int *p_map, int *p_mask){

    cudaError_t err;

    dim3 blocks(DIVUP(num_points, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    voxel_to_point_query_kernel<<<blocks, threads>>>(x_size, y_size, z_size, x_max, y_max, z_max,
                                                     num_points, num_voxels, num_samples, step_range,
                                                     xyz, xyz_to_vidx, p_map, p_mask);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
