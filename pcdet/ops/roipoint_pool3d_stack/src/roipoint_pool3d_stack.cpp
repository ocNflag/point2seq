#include <torch/serialize/tensor.h>
#include <torch/extension.h>

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


void ROIPoolStackLauncher(
    int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
    const int *pts_batch_id, const int *pts_batch_cnt, const float *xyz,
    const float *boxes3d, const float *pts_feature,
    float *pooled_features, int *pooled_empty_flag, int *point_empty_flag
);


int roipool3d_stack_gpu(
    at::Tensor boxes3d, at::Tensor pts_batch_id, at::Tensor pts_batch_cnt,
    at::Tensor xyz, at::Tensor pts_feature,
    at::Tensor pooled_features, at::Tensor pooled_empty_flag, at::Tensor point_empty_flag
){
    // params xyz: (N1 + N2 + ..., 3)
    // params boxes3d: (B, M, 7)
    // params pts_feature: (N1 + N2 + ..., C)
    // params pts_batch_id: (N1, N2, N3 ...)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    CHECK_INPUT(boxes3d);
    CHECK_INPUT(pts_batch_id);
    CHECK_INPUT(pts_batch_cnt);
    CHECK_INPUT(xyz);
    CHECK_INPUT(pts_feature);
    CHECK_INPUT(pooled_features);
    CHECK_INPUT(pooled_empty_flag);
    CHECK_INPUT(point_empty_flag);

    int batch_size = pts_batch_cnt.size(0);
    int pts_num = xyz.size(0);
    int boxes_num = boxes3d.size(1);
    int feature_in_len = pts_feature.size(1);
    int sampled_pts_num = pooled_features.size(2);

    const int * pts_batch_id_data = pts_batch_id.data<int>();
    const int * pts_batch_cnt_data = pts_batch_cnt.data<int>();
    const float * xyz_data = xyz.data<float>();
    const float * boxes3d_data = boxes3d.data<float>();
    const float * pts_feature_data = pts_feature.data<float>();
    float * pooled_features_data = pooled_features.data<float>();
    int * pooled_empty_flag_data = pooled_empty_flag.data<int>();
    int * point_empty_flag_data = point_empty_flag.data<int>();

    ROIPoolStackLauncher(
        batch_size, pts_num, boxes_num, feature_in_len,
        sampled_pts_num, pts_batch_id_data, pts_batch_cnt_data,
        xyz_data, boxes3d_data, pts_feature_data,
        pooled_features_data, pooled_empty_flag_data, point_empty_flag_data
    );

    return 1;
}



void ROIPointQueryStackLauncher(
    int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
    const int *pts_batch_id, const int *pts_batch_cnt, const float *xyz,
    const float *boxes3d, int *point_idx
);


int roi_point_query_stack_gpu(
    at::Tensor boxes3d, at::Tensor pts_batch_id, at::Tensor pts_batch_cnt,
    at::Tensor xyz, at::Tensor point_idx
){
    // params xyz: (N1 + N2 + ..., 3)
    // params boxes3d: (B, M, 7)
    // params pts_batch_id: (N1, N2, N3 ...)
    // params point_idx_data: (B, M, 512)

    CHECK_INPUT(boxes3d);
    CHECK_INPUT(pts_batch_id);
    CHECK_INPUT(pts_batch_cnt);
    CHECK_INPUT(xyz);
    CHECK_INPUT(point_idx);

    int batch_size = pts_batch_cnt.size(0);
    int pts_num = xyz.size(0);
    int boxes_num = boxes3d.size(1);
    int sampled_pts_num = point_idx.size(2);

    const int * pts_batch_id_data = pts_batch_id.data<int>();
    const int * pts_batch_cnt_data = pts_batch_cnt.data<int>();
    const float * xyz_data = xyz.data<float>();
    const float * boxes3d_data = boxes3d.data<float>();
    int * point_idx_data = point_idx.data<int>();

    ROIPointQueryStackLauncher(
        batch_size, pts_num, boxes_num,
        sampled_pts_num, pts_batch_id_data, pts_batch_cnt_data,
        xyz_data, boxes3d_data, point_idx_data
    );

    return 1;
}


void group_features_kernel_launcher(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out);

void group_features_grad_kernel_launcher(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features);


int group_features_grad_wrapper(int B, int M, int C, int N, int nsample,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor,
    at::Tensor features_batch_cnt_tensor, at::Tensor grad_features_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(idx_batch_cnt_tensor);
    CHECK_INPUT(features_batch_cnt_tensor);
    CHECK_INPUT(grad_features_tensor);

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
    const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
    float *grad_features = grad_features_tensor.data<float>();

    group_features_grad_kernel_launcher(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);
    return 1;
}


int group_features_wrapper(int B, int M, int C, int nsample,
    at::Tensor features_tensor, at::Tensor features_batch_cnt_tensor,
    at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor, at::Tensor out_tensor) {

    CHECK_INPUT(features_tensor);
    CHECK_INPUT(features_batch_cnt_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(idx_batch_cnt_tensor);
    CHECK_INPUT(out_tensor);

    const float *features = features_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
    const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
    float *out = out_tensor.data<float>();

    group_features_kernel_launcher(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &roipool3d_stack_gpu, "roipool3d_stack forward (CUDA)");
    m.def("point_query", &roi_point_query_stack_gpu, "roipool3d_stack forward (CUDA)");
    m.def("group_features_grad_wrapper", &group_features_grad_wrapper, "group_features_grad_wrapper");
    m.def("group_features_wrapper", &group_features_wrapper, "group_features_wrapper");
}

