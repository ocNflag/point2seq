/*
Query voxel centers for each point
Written by Jiageng Mao
*/


#ifndef VOXEL_TO_POINT_SIMPLE_GPU_H
#define VOXEL_TO_POINT_SIMPLE_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int voxel_to_point_query_simple_wrapper(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                      int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                      int x_divide, int y_divide, int z_divide,
                                      int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                      at::Tensor p_bs_cnt_tensor, at::Tensor xyz_tensor, at::Tensor xyz_to_vidx_tensor,
                                      at::Tensor p_map_tensor, at::Tensor p_mask_tensor, at::Tensor p_bin_tensor);

void voxel_to_point_query_simple_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                                int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                                int x_divide, int y_divide, int z_divide,
                                                int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                                const int *p_bs_cnt, const float *xyz, const int *xyz_to_vidx, int *p_map, int *p_mask, int *p_bin);

#endif
