/*
Construct voxel xyz to index hash tables
Written by Jiageng Mao
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "fill_dense_simple_gpu.h"
#include "pv_cuda_utils.h"

// simple hash only using % op
__device__ int hash_fs(int k, int max_hash_size) {
    return k % max_hash_size;
}

__global__ void fill_dense_simple_kernel(int x_max, int y_max, int z_max, int num_total_voxels, int max_hash_size, 
                                        const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx) {
    /*
        v_indices: [N1+N2, 4] bs zyx indices of voxels
        v_bs_cnt: [bs] num_voxels in each sample
        xyz_to_vidx: [B, max_hash_size, 2] hash table key-value for dim-2
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_total_voxels) return;
    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1]; 
    int y_idx = v_indices[th_idx * 4 + 2]; 
    int x_idx = v_indices[th_idx * 4 + 3];

    int v_sum = 0;
    int bs_cnt = bs_idx - 1;
    while(bs_cnt >= 0){
        v_sum += v_bs_cnt[bs_cnt];
        bs_cnt--;
    }
    int v_idx = th_idx - v_sum; // v_idx for this sample

    xyz_to_vidx += bs_idx * max_hash_size * 2;
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return; // out of bound

    // key -> [x_max, y_max, z_max] value -> v_idx
    int key = x_idx * y_max * z_max + y_idx * z_max + z_idx;
    int hash_idx = hash_fs(key, max_hash_size);
    int prob_cnt = 0;
    while(true) {
        int prev_key = atomicCAS(xyz_to_vidx + hash_idx*2 + 0, EMPTY_KEY, key); // insert key when empty
        if (prev_key == EMPTY_KEY || prev_key == key) {
            xyz_to_vidx[hash_idx*2 + 1] = v_idx; // insert value
            break;
        }
        // linear probing
        //hash_idx = (hash_idx + 1) & (max_hash_size-1);        
        hash_idx = (hash_idx + 1) % max_hash_size;
        
        // security in case of dead loop
        prob_cnt += 1;
        if (prob_cnt >= max_hash_size) break;
    }
    return;
}


void fill_dense_simple_kernel_launcher(int x_max, int y_max, int z_max, int num_total_voxels,
                                        int max_hash_size, const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_total_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    fill_dense_simple_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_total_voxels,
                                                max_hash_size, v_indices, v_bs_cnt, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
