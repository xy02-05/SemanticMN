"""
将 4 套 LIBERO raw RLDS 合并为 1 套新的 TFDS/RLDS 数据集。

这一步只做“拼接”，不改原始数据语义：
- 不改 action
- 不改 state
- 不改 language_instruction
- 不补 task_index / task_id

输出仍然是标准 TFDS/RLDS 目录，后续 openpi 可以继续走 RLDS 训练链。
"""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Iterable, Sequence

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.rlds import rlds_base
import tyro


RAW_DATASET_NAMES = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _source_builder_dir(data_dir: pathlib.Path, dataset_name: str, version: str) -> pathlib.Path:
    return data_dir / dataset_name / version


def _load_builder_from_directory(builder_dir: pathlib.Path):
    return tfds.builder_from_directory(str(builder_dir))


def _count_total_episodes(source_data_dir: pathlib.Path, dataset_names: Sequence[str], version: str) -> int:
    total_episodes = 0
    for dataset_name in dataset_names:
        builder = _load_builder_from_directory(_source_builder_dir(source_data_dir, dataset_name, version))
        total_episodes += int(builder.info.splits["train"].num_examples)
    return total_episodes


def _convert_episode_steps(steps) -> list[dict]:
    """把 RLDS episode["steps"] 转成 TFDS 期望的 step 列表。"""
    converted_steps: list[dict] = []

    for step in steps:
        converted_steps.append(
            {
                "observation": {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "joint_state": step["observation"]["joint_state"],
                },
                "action": step["action"],
                "discount": np.float32(step["discount"]),
                "reward": np.float32(step["reward"]),
                "is_first": bool(step["is_first"]),
                "is_last": bool(step["is_last"]),
                "is_terminal": bool(step["is_terminal"]),
                "language_instruction": _decode_text(step["language_instruction"]),
            }
        )

    return converted_steps


class LiberoMixNoNoops(tfds.core.GeneratorBasedBuilder):
    """本地 LIBERO merged RLDS builder。"""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial merged release from the four openvla LIBERO raw RLDS suites.",
    }

    def __init__(
        self,
        *,
        source_data_dir: str,
        source_dataset_names: Sequence[str],
        dataset_metadata: dict,
        **kwargs,
    ):
        self._source_data_dir = pathlib.Path(source_data_dir)
        self._source_dataset_names = tuple(source_dataset_names)
        self._dataset_metadata = dataset_metadata
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        dataset_config = rlds_base.DatasetConfig(
            name="default",
            description="Merged LIBERO raw RLDS dataset for OpenPI training.",
            overall_description=(
                "This dataset concatenates the four raw LIBERO RLDS suites from "
                "openvla/modified_libero_rlds without changing step semantics."
            ),
            homepage="https://huggingface.co/datasets/openvla/modified_libero_rlds",
            citation="",
            observation_info={
                "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            },
            action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
            reward_info=np.float32,
            discount_info=np.float32,
            step_metadata_info={
                "language_instruction": tfds.features.Text(),
            },
            episode_metadata_info={
                "file_path": tfds.features.Text(),
            },
            supervised_keys=None,
        )
        return rlds_base.build_info(dataset_config, self, ds_metadata=self._dataset_metadata)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self) -> Iterable[tuple[str, dict]]:
        global_episode_index = 0

        for dataset_name in self._source_dataset_names:
            builder_dir = _source_builder_dir(self._source_data_dir, dataset_name, str(self.VERSION))
            builder = _load_builder_from_directory(builder_dir)
            dataset = builder.as_dataset(split="train", shuffle_files=False)

            for local_episode_index, episode in enumerate(tfds.as_numpy(dataset)):
                key = f"{dataset_name}_{local_episode_index:06d}"
                steps = _convert_episode_steps(episode["steps"])
                file_path = _decode_text(episode["episode_metadata"]["file_path"])

                yield key, {
                    "steps": steps,
                    "file_path": file_path,
                }
                global_episode_index += 1


def main(
    source_data_dir: str = "/root/data/xuyuan1/dataset/libero_rlds",
    output_data_dir: str = "/root/data/xuyuan1/dataset/libero_rlds",
    *,
    overwrite: bool = False,
):
    source_data_dir = pathlib.Path(source_data_dir)
    output_data_dir = pathlib.Path(output_data_dir)
    output_dataset_dir = output_data_dir / "libero_mix_no_noops"

    if output_dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_dataset_dir}")
        shutil.rmtree(output_dataset_dir)

    total_episodes = _count_total_episodes(source_data_dir, RAW_DATASET_NAMES, "1.0.0")
    dataset_metadata = {
        "source_dataset_names": list(RAW_DATASET_NAMES),
        "total_episodes": total_episodes,
    }

    builder = LiberoMixNoNoops(
        source_data_dir=str(source_data_dir),
        source_dataset_names=RAW_DATASET_NAMES,
        dataset_metadata=dataset_metadata,
        data_dir=str(output_data_dir),
    )
    builder.download_and_prepare()


if __name__ == "__main__":
    tyro.cli(main)
