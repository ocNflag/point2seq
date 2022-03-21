/*
Query voxel centers for each point
Written by Jiageng Mao
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "voxel_to_point_bin_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int voxel_to_point_query_bin_wrapper(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                      int x_step, int y_step, int z_step, int x_dilate, int y_dilate, int z_dilate,
                                      int x_divide, int y_divide, int z_divide,
                                      int num_total_points, int num_total_voxels, int num_samples, int max_hash_size,
                                      at::Tensor p_bs_cnt_tensor, at::Tensor xyz_tensor, at::Tensor xyz_to_vidx_tensor,
                                      at::Tensor p_map_tensor, at::Tensor p_mask_tensor, at::Tensor p_bin_tensor) {
    CHECK_INPUT(p_bs_cnt_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(p_map_tensor);
    CHECK_INPUT(p_mask_tensor);
    CHECK_INPUT(p_bin_tensor);

    const int *p_bs_cnt = p_bs_cnt_tensor.data<int>();
    const float *xyz = xyz_tensor.data<float>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    int *p_map = p_map_tensor.data<int>();
    int *p_mask = p_mask_tensor.data<int>();
    int *p_bin = p_bin_tensor.data<int>();

    voxel_to_point_query_bin_kernel_launcher(x_size, y_size, z_size, x_max, y_max, z_max,
                                              x_step, y_step, z_step, x_dilate, y_dilate, z_dilate,
                                              x_divide, y_divide, z_divide,
                                              num_total_points, num_total_voxels, num_samples, max_hash_size,
                                              p_bs_cnt, xyz, xyz_to_vidx, p_map, p_mask, p_bin);
    return 1;
}
