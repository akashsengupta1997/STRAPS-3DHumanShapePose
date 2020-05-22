# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from . import dataset  # just to register data
from .config import add_densepose_config
from .densepose_head import ROI_DENSEPOSE_HEAD_REGISTRY
from .roi_head import DensePoseROIHeads
from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
