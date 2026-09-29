"""
用 EgoHOD Large 模型编码 SimplerEnv 4 个基础任务的 text embedding 并存储

重要: 必须使用 EgoHOD Large (ViT-L-14-336px + large_best.pt)，
因为 bridge_text_egohod_proj.npz 就是用 Large 编码的，Base 和 Large 的
embedding 空间完全不同，不可混用。

用法:
    conda activate gaze
    python evaluation/encode_simpler_text.py

输出:
    dataset/embedding/simpler_text_egohod.npz
    - embeddings: [4, 512]  L2 归一化后的投影后特征
    - task_names: [4] str
    - instructions: [4] str
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import clip

# 添加 egovlpv2 到 path
EGOHOD_ROOT = "/root/data/xuyuan1/Codes/mirror_neuron/egovlpv2"
sys.path.insert(0, EGOHOD_ROOT)

from egovlpv2.model.model_egohod import EgoHODModel

# SimplerEnv 4 个基础任务的精确语言指令 (来自 ManiSkill 环境定义)
SIMPLER_TASKS = {
    "widowx_spoon_on_towel": "put the spoon on the towel",
    "widowx_carrot_on_plate": "put carrot on plate",
    "widowx_stack_cube": "stack the green block on the yellow block",
    "widowx_put_eggplant_in_basket": "put eggplant into yellow basket",
}

# EgoHOD Large 模型路径（与 bridge_text_egohod_proj.npz 生成时完全一致）
CLIP_WEIGHT = os.path.join(EGOHOD_ROOT, "pretrain_weight/pretrain_weight/ViT-L-14-336px.pt")
EGOHOD_WEIGHT = os.path.join(EGOHOD_ROOT, "pretrain_weight/egohod/large_best.pt")

OUTPUT_PATH = "/root/data/xuyuan1/dataset/embedding/simpler_text_egohod.npz"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 构建 EgoHOD Large（与 encode_bridge_text_egohod.py 的 build_model 一致）
    print("加载 EgoHOD Large 模型...")
    model = EgoHODModel(
        video_params={"model": "SpaceTimeTransformer", "num_frames": 4},
        text_params={"model": "clip"},
        projection_dim=512,
        load_checkpoint=CLIP_WEIGHT,
        project_embed_dim=512,
        num_frames=4,
        use_fast_conv1=True,
        use_flash_attn=True,
        context_length=77,
        vocab_size=49408,
        freeze_temperature=True,
        egohod_checkpoint_path=EGOHOD_WEIGHT,
    )
    model = model.to(device).half().eval()
    print("  模型加载完成")

    task_names = list(SIMPLER_TASKS.keys())
    instructions = list(SIMPLER_TASKS.values())

    # CLIP tokenize → EgoHOD encode → L2 normalize（与 bridge 生成流程一致）
    tokens = clip.tokenize(instructions, truncate=True).to(device)
    text_data = {"input_ids": tokens}

    with torch.no_grad():
        raw_emb = model.compute_text(text_data)          # [4, 512]
        embeddings = F.normalize(raw_emb, dim=-1)         # L2 归一化

    embeddings = embeddings.cpu().float().numpy()
    print(f"  embeddings shape: {embeddings.shape}")

    for name, instr, emb in zip(task_names, instructions, embeddings):
        norm = np.linalg.norm(emb)
        print(f"  {name}: \"{instr}\" → norm={norm:.4f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH,
             embeddings=embeddings,
             task_names=np.array(task_names),
             instructions=np.array(instructions))
    print(f"\n保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
