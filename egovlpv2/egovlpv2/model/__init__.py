# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "1.0.0"

# 导入主要的模型类 - 使用正确的绝对路径
from egovlpv2.model.model import *
from egovlpv2.model.model_epic_charades import *
from egovlpv2.model.model_egohod import EgoHODModel
from egovlpv2.model.roberta import *
from egovlpv2.model.video_transformer import *
from egovlpv2.model.metric import *
from egovlpv2.model.fg_alignment_model import *

# 导入损失函数
from egovlpv2.model.loss import *

# 导入头部模块
from egovlpv2.model.heads import *