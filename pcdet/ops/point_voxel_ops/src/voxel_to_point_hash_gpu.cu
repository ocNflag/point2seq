/*
Query voxel centers for each point
Written by Jiageng Mao
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "voxel_to_point_hash_gpu.h"
#include "pv_cuda_utils.h"

// 32 bit Murmur3 hash
// unsigned int -> int, k >= 0, max_hash_size >0, should be ok?
__device__ int hash_vp(int k, int max_hash_size) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    //return k & (max_hash_size-1);
    return k % max_hash_size;
}

__global__ void voxel_to_point_query_hash_kernel(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                                    int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                                    int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                                    const int *p_bs_cnt, const float *xyz, const int *xyz_to_vidx, int *p_map, int *p_mask) {
    /*
        xyz: [N1+N2, 4] xyz coordinates of points
        xyz_to_vidx: [B, max_hash_size, 2] voxel coordinates to voxel indices
        p_bs_cnt: [B]
        p_map: [N1+N2, num_samples] voxel indices for each point
        p_mask: [N1+N2, 1] num_voxels for each point
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_total_points) return;
    int bs_idx = (int) xyz[th_idx * 4 + 0];
    float x = xyz[th_idx * 4 + 1];
    float y = xyz[th_idx * 4 + 2];
    float z = xyz[th_idx * 4 + 3];
    float x_res = x / x_size - 0.5;
    float y_res = y / y_size - 0.5;
    float z_res = z / z_size - 0.5;                                   
    int x_f_idx = floor(x_res);                                             
    int y_f_idx = floor(y_res);                                            
    int z_f_idx = floor(z_res);
    int x_c_idx = ceil(x_res);                                            
    int y_c_idx = ceil(y_res);                                            
    int z_c_idx = ceil(z_res);

    xyz_to_vidx += bs_idx * max_hash_size * 2;
    int p_sum = 0;
    int bs_cnt = bs_idx - 1;
    while(bs_cnt >= 0){
        p_sum += p_bs_cnt[bs_cnt];
        bs_cnt--;
    }
    int pt_idx = th_idx - p_sum; // pt_idx for this sample
    p_map += p_sum * num_samples;
    p_mask += p_sum * 1;

    //int step = 2 * step_range - 1;
    //int step = step_range;
    for (int z_idx = z_f_idx - (z_step - 1); z_idx <= z_c_idx + (z_step - 1); z_idx += z_dilate) {
        if (z_idx < 0 || z_idx >= z_max) continue;
        for (int y_idx = y_f_idx - (y_step - 1); y_idx <= y_c_idx + (y_step - 1); y_idx += y_dilate) {
            if (y_idx < 0 || y_idx >= y_max) continue;
            for (int x_idx = x_f_idx - (x_step -1); x_idx <= x_c_idx + (x_step - 1); x_idx += x_dilate) {
                if (x_idx >= x_max || x_idx < 0) continue; // out of bound

                // key -> [x_max, y_max, z_max] value -> v_idx
                int key = x_idx * y_max * z_max + y_idx * z_max + z_idx;
                int hash_idx = hash_vp(key, max_hash_size);
                int v_idx = -1;
                int prob_cnt = 0;
                while (true) {
                    // found
                    if (xyz_to_vidx[hash_idx * 2 + 0] == key) {
                        v_idx = xyz_to_vidx[hash_idx * 2 + 1];
                        break;
                    }
                    // empty, not found
                    if (xyz_to_vidx[hash_idx * 2 + 0] == EMPTY_KEY) {
                        break;
                    }
                    // linear probing
                    //hash_idx = (hash_idx + 1) & (max_hash_size - 1);
                    hash_idx = (hash_idx + 1) % max_hash_size;

                    // security in case of dead loop
                    prob_cnt += 1;
                    if (prob_cnt >= max_hash_size) break;
                }
            
                if (v_idx < 0) continue; // -1 not found
                int sample_idx = atomicAdd(p_mask + pt_idx, 1);
                //if (sample_idx >= num_samples) return;
                if (sample_idx >= num_samples) continue; // check how many points in this radius
                if (sample_idx == 0) {
                    for (int s_idx = 0; s_idx < num_samples; ++s_idx) {
                        // fill with first points
                        p_map[pt_idx * num_samples + s_idx] = v_idx;
                    }
                } else {
                    p_map[pt_idx * num_samples + sample_idx] = v_idx;
                }
            }
        }
    }
    return;
}


void voxel_to_point_query_hash_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                                int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                                int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                                const int *p_bs_cnt, const float *xyz, const int *xyz_to_vidx, int *p_map, int *p_mask){

    cudaError_t err;

    dim3 blocks(DIVUP(num_total_points, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    voxel_to_point_query_hash_kernel<<<blocks, threads>>>(x_size, y_size, z_size, x_max, y_max, z_max,
                                                            x_step, y_step, z_step, x_dilate, y_dilate, z_dilate,
                                                            num_total_points, num_total_voxels, num_samples, max_hash_size,
                                                            p_bs_cnt, xyz, xyz_to_vidx, p_map, p_mask);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
