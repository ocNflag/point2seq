from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate
from .pvrcnn_head_sample import PVRCNNHeadSample
from .pyramid_head import PyramidHead
from .center_rcnn_head import CenterRCNNHead
from .center_rcnn_head_zero import CenterRCNNHeadZero
from .center_rcnn_tasks import CenterRCNNTasks
from .e2e_roi_head import E2EROIHead
from .e2e_roi_headv2 import E2EROIHeadV2
from .e2e_roi_headv3 import E2EROIHeadV3


from .e2e_roi_fusion_head import E2EROIFusionHead
from .roi_fusion_head import ROIFusionHead
from .e2e_roi_fusion_headv2 import E2EROIFusionHeadV2
from .e2e_roi_fusion_headv3 import E2EROIFusionHeadV3
from .e2e_pvrcnn_head import E2EPVRCNNHead
from .e2e_pvrcnn_headv2 import E2EPVRCNNHeadV2
from .e2e_pvrcnn_headv3 import E2EPVRCNNHeadV3



__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'PVRCNNHeadSample': PVRCNNHeadSample,
    'PyramidHead': PyramidHead,
    'CenterRCNNHead': CenterRCNNHead,
    'CenterRCNNHeadZero': CenterRCNNHeadZero,
    'CenterRCNNTasks': CenterRCNNTasks,
    'E2EROIHead': E2EROIHead,
    'E2EROIFusionHead': E2EROIFusionHead,
    'ROIFusionHead': ROIFusionHead,
    'E2EROIHeadV2': E2EROIHeadV2,
    'E2EROIFusionHeadV2': E2EROIFusionHeadV2,
    'E2EROIFusionHeadV3': E2EROIFusionHeadV3,
    'E2EPVRCNNHead': E2EPVRCNNHead,
    'E2EROIHeadV3': E2EROIHeadV3,
    'E2EPVRCNNHeadV2': E2EPVRCNNHeadV2,
    'E2EPVRCNNHeadV3': E2EPVRCNNHeadV3
}
