/*
Query points for each voxel
Written by Jiageng Mao
*/


#ifndef POINT_TO_VOXEL_BIN_GPU_H
#define POINT_TO_VOXEL_BIN_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int point_to_voxel_query_bin_wrapper(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                      int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                      int x_divide, int y_divide, int z_divide,
                                      int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                      at::Tensor p_bs_cnt_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_tensor,
                                      at::Tensor xyz_to_vidx_tensor,
                                      at::Tensor v_mask_tensor, at::Tensor v_bin_tensor);

void point_to_voxel_query_bin_kernel_launcher(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                                int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                                int x_divide, int y_divide, int z_divide,
                                                int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                                const int *p_bs_cnt, const int *v_bs_cnt, const float *xyz, const int *xyz_to_vidx,
                                                int *v_mask, int *v_bin);

int merge_pv_bin_wrapper(int x_divide, int y_divide, int z_divide, int num_total_voxels, int num_samples, int batch_size,
                                    at::Tensor v_bs_cnt_tensor, at::Tensor v_bin_tensor, at::Tensor v_map_tensor);

void merge_pv_bin_kernel_launcher(int x_divide, int y_divide, int z_divide, int num_total_voxels, int num_samples, int batch_size,
                                    const int *v_bs_cnt, const int *v_bin, int *v_map);

#endif
