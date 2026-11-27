from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle, AnchorHeadSingle_for_dsp, AnchorHeadSingle_test,AnchorHeadSingle_QAT
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'AnchorHeadSingle_for_dsp': AnchorHeadSingle_for_dsp,
    'AnchorHeadSingle_test': AnchorHeadSingle_test,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'AnchorHeadSingle_QAT' : AnchorHeadSingle_QAT
}
