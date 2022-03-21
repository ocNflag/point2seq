/*
Query points for each voxel
Written by Jiageng Mao
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "point_to_voxel_gpu.h"

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

int point_to_voxel_query_wrapper(float x_size, float y_size, float z_size, int x_max, int y_max, int z_max,
                                 int num_points, int num_voxels, int num_samples,
                                 at::Tensor xyz_tensor, at::Tensor xyz_to_vidx_tensor,
                                 at::Tensor v_map_tensor, at::Tensor v_mask_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(v_map_tensor);
    CHECK_INPUT(v_mask_tensor);

    const float *xyz = xyz_tensor.data<float>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    int *v_map = v_map_tensor.data<int>();
    int *v_mask = v_mask_tensor.data<int>();

    point_to_voxel_query_kernel_launcher(x_size, y_size, z_size, x_max, y_max, z_max,
                                         num_points, num_voxels, num_samples,
                                         xyz, xyz_to_vidx, v_map, v_mask);
    return 1;
}
