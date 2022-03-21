/*
Fill dense 3d voxels with indices
Written by Jiageng Mao
*/

#ifndef FILL_DENSE_SIMPLE_GPU_H
#define FILL_DENSE_SIMPLE_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int fill_dense_simple_wrapper(int x_max, int y_max, int z_max, int num_total_voxels, int max_hash_size,
                            at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor);

void fill_dense_simple_kernel_launcher(int x_max, int y_max, int z_max, int num_total_voxels,
                                        int max_hash_size, const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx);

#endif
