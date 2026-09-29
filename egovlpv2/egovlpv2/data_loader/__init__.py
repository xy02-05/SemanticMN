# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "1.0.0"

# 导入主要的数据集类 - 使用正确的绝对路径
from egovlpv2.data_loader.FHO_dataset import *
from egovlpv2.data_loader.EgoClip_EgoMCQ_dataset import *
from egovlpv2.data_loader.CharadesEgo_dataset import *
from egovlpv2.data_loader.Ego4D_MQ_dataset import *
from egovlpv2.data_loader.EpicKitchens_MIR_dataset import *

# 导入数据加载器
from egovlpv2.data_loader.data_loader import *

# 导入变换函数
from egovlpv2.data_loader.transforms import *