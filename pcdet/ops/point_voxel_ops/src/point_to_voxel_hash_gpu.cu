/*
Query points for each voxel
Written by Jiageng Mao
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "point_to_voxel_hash_gpu.h"
#include "pv_cuda_utils.h"

// 32 bit Murmur3 hash
// unsigned int -> int, k >= 0, max_hash_size >0, should be ok?
__device__ int hash_pv(int k, int max_hash_size) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    //return k & (max_hash_size-1);
    return k % max_hash_size;
}

__global__ void point_to_voxel_query_hash_kernel(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                                    int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                                    const int *p_bs_cnt, const int *v_bs_cnt,
                                                    const float *xyz, const int *xyz_to_vidx, int *v_map, int *v_mask) {
    /*
        xyz: [N1+N2, 4] xyz coordinates of points
        xyz_to_vidx: [B, max_hash_size, 2] voxel coordinates to voxel indices
        p_bs_cnt: [B]
        v_bs_cnt: [B]
        v_map & v_mask w/o leading bs_idx
        v_map: [M1+M2, num_samples] points indices for each voxel
        v_mask: [M1+M2, 1] num_points for each voxel
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_total_points) return;
    int bs_idx = (int) xyz[th_idx * 4 + 0];
    float x = xyz[th_idx * 4 + 1];
    float y = xyz[th_idx * 4 + 2];
    float z = xyz[th_idx * 4 + 3];                                         
    int x_idx = floor(x / x_size);                                            
    int y_idx = floor(y / y_size);                                            
    int z_idx = floor(z / z_size);                                            
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) // out of bound
        return;

    xyz_to_vidx += bs_idx * max_hash_size * 2;
    int v_sum = 0;
    int p_sum = 0;
    int bs_cnt = bs_idx - 1;
    while(bs_cnt >= 0){
        v_sum += v_bs_cnt[bs_cnt];
        p_sum += p_bs_cnt[bs_cnt];
        bs_cnt--;
    }
    int pt_idx = th_idx - p_sum; // pt_idx for this sample
    v_map += v_sum * num_samples;
    v_mask += v_sum * 1;

    // key -> [x_max, y_max, z_max] value -> v_idx
    int key = x_idx * y_max * z_max + y_idx * z_max + z_idx;
    int hash_idx = hash_pv(key, max_hash_size);
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

    if (v_idx < 0) return; // -1 not found                                            
    int sample_idx = atomicAdd(v_mask + v_idx, 1); 
    if (sample_idx >= num_samples) return;
    v_map[v_idx * num_samples + sample_idx] = pt_idx;
    return;
}


void point_to_voxel_query_hash_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                                int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                                const int *p_bs_cnt, const int *v_bs_cnt,
                                                const float *xyz, const int *xyz_to_vidx, int *v_map, int *v_mask){

    cudaError_t err;

    dim3 blocks(DIVUP(num_total_points, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    point_to_voxel_query_hash_kernel<<<blocks, threads>>>(x_size, y_size, z_size, x_max, y_max, z_max,
                                                            num_total_points, num_total_voxels, num_samples, max_hash_size,
                                                            p_bs_cnt, v_bs_cnt, xyz, xyz_to_vidx, v_map, v_mask);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
