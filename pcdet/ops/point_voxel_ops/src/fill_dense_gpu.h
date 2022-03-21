/*
Fill dense 3d voxels with indices
Written by Jiageng Mao
*/

#ifndef FILL_DENSE_GPU_H
#define FILL_DENSE_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int fill_dense_wrapper(int x_max, int y_max, int z_max, int num_voxels, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor);

void fill_dense_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, const int *v_indices, int *xyz_to_vidx);

#endif
