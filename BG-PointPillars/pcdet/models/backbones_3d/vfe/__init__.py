from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, PillarVFE_onnx, PillarVFE_dual, Pillar_to_conv3D, PillarVFE_1x1conv,PillarVFE_input_norm,PillarVFE_input_norm_mean_plus,PillarVFE_mean_plus,PillarVFE_QAT,PillarVFE_QAT_1d,PillarVFE_batch2d,PillarVFE_QAT_input_norm,PillarVFE_complex_yolo,PillarVFE_skip,PillarVFE_forground_extraction,PillarVFE_forground_extraction_11D,PillarVFE_forground_extraction_dual_pointnet
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'PillarVFE_dual' : PillarVFE_dual,
    'PillarVFE_onnx': PillarVFE_onnx,
    'Pillar_to_conv3D' : Pillar_to_conv3D,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'PillarVFE_1x1conv': PillarVFE_1x1conv,
    'PillarVFE_input_norm' : PillarVFE_input_norm,
    'PillarVFE_input_norm_mean_plus' : PillarVFE_input_norm_mean_plus,
    'PillarVFE_mean_plus' : PillarVFE_mean_plus,
    'PillarVFE_QAT' : PillarVFE_QAT,
    'PillarVFE_QAT_1d' : PillarVFE_QAT_1d,
    'PillarVFE_batch2d' : PillarVFE_batch2d,
    'PillarVFE_QAT_input_norm' : PillarVFE_QAT_input_norm,
    'PillarVFE_complex_yolo' : PillarVFE_complex_yolo,
    'PillarVFE_skip' : PillarVFE_skip,
    'PillarVFE_forground_extraction' : PillarVFE_forground_extraction,
    'PillarVFE_forground_extraction_11D' : PillarVFE_forground_extraction_11D,
    'PillarVFE_forground_extraction_dual_pointnet' : PillarVFE_forground_extraction_dual_pointnet

}
