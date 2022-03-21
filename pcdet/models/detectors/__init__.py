from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .pv_rcnn_nus import PVRCNN_NUS
from .center_points import CenterPoints
from .center_rcnn import CenterRCNN
from .center_rcnn_zero import CenterRCNNZero
from .center_rcnn_fix import CenterRCNNFix
from .e2enet import E2ENet



__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'PVRCNN_NUS': PVRCNN_NUS,
    'CenterPoints': CenterPoints,
    'CenterRCNN': CenterRCNN,
    'CenterRCNNZero': CenterRCNNZero,
    'CenterRCNNFix': CenterRCNNFix,
    'E2ENet': E2ENet,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
