#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

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


void draw_all_kernel_launcher(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int code_size,
                              int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                              float out_factor, float gaussian_overlap, float fg_enlarge_ratio,
                              const float *gt_boxes, const float *gt_corners,
                              float *center_map, float *corner_map, float *foreground_map,
                              int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt);


int draw_all_gpu(at::Tensor gt_boxes_tensor, at::Tensor gt_corners_tensor,
                at::Tensor center_map_tensor, at::Tensor corner_map_tensor,
                at::Tensor foreground_map_tensor, at::Tensor gt_ind_tensor,
                at::Tensor gt_mask_tensor, at::Tensor gt_cat_tensor,
                at::Tensor gt_box_encoding_tensor, at::Tensor gt_cnt_tensor,
                int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                float out_factor, float gaussian_overlap, float fg_enlarge_ratio){

    /*
    Args:
            gt_boxes: (B, max_boxes, 8 or 10) with class labels
            gt_corners: (B, max_boxes, 4)
            center_map: (B, num_cls, H, W)
            gt_ind: (B, num_cls, max_objs)
            gt_mask: (B, num_cls, max_objs)
            gt_cat: (B, num_cls, max_objs)
            gt_box_encoding: (B, num_cls, max_objs, code_size) sin/cos
            gt_cnt: (B, num_cls)
    */
    CHECK_INPUT(gt_boxes_tensor);
    CHECK_INPUT(gt_corners_tensor);
    CHECK_INPUT(center_map_tensor);
    CHECK_INPUT(corner_map_tensor);
    CHECK_INPUT(foreground_map_tensor);
    CHECK_INPUT(gt_ind_tensor);
    CHECK_INPUT(gt_mask_tensor);
    CHECK_INPUT(gt_cat_tensor);
    CHECK_INPUT(gt_box_encoding_tensor);
    CHECK_INPUT(gt_cnt_tensor);

    int batch_size = gt_boxes_tensor.size(0);
    int max_boxes = gt_boxes_tensor.size(1);
    int code_size = gt_boxes_tensor.size(2);
    int num_cls = center_map_tensor.size(1);
    int H = center_map_tensor.size(2);
    int W = center_map_tensor.size(3);
    int max_objs = gt_ind_tensor.size(2);


    float *center_map = center_map_tensor.data<float>();
    float *corner_map = corner_map_tensor.data<float>();
    float *foreground_map = foreground_map_tensor.data<float>();

    const float *gt_boxes = gt_boxes_tensor.data<float>();
    const float *gt_corners = gt_corners_tensor.data<float>();

    int *gt_ind = gt_ind_tensor.data<int>();
    int *gt_mask = gt_mask_tensor.data<int>();
    int *gt_cat = gt_cat_tensor.data<int>();
    float *gt_box_encoding = gt_box_encoding_tensor.data<float>();
    int *gt_cnt = gt_cnt_tensor.data<int>();

    draw_all_kernel_launcher(batch_size, max_boxes, max_objs, num_cls, H, W, code_size, min_radius,
                             voxel_x, voxel_y, range_x, range_y, out_factor, gaussian_overlap,
                             fg_enlarge_ratio, gt_boxes, gt_corners, center_map, corner_map, foreground_map,
                             gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt);

    return 1;
}
