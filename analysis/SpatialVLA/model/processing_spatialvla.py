# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import List, Optional, Union, Dict
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import Unpack, _validate_images_text_input_order, ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.models.paligemma.processing_paligemma import (
    make_batched_images, 
    build_string_from_input, 
    _is_str_or_image, 
    PaliGemmaProcessorKwargs,
    IMAGE_TOKEN,
    EXTRA_TOKENS
)
from .action_tokenizer import SpatialActionTokenizer
logger = logging.get_logger(__name__)

class SpatialVLAProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        statistics: Optional[dict] = None,
        bin_policy=None,
        intrinsic_config=None,
        action_config=None,
        num_obs_steps=1,
        obs_delta=1,
        action_chunk_size=1,
        min_sigma=0.0,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id

        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        # action tokenizer
        self.statistics = statistics if statistics else {}
        self.bin_policy = bin_policy
        self.min_sigma = min_sigma
        self.intrinsic_config = intrinsic_config
        self.action_config = action_config
        self.num_obs_steps = num_obs_steps
        self.obs_delta = obs_delta
        self.action_chunk_size = action_chunk_size
        self.dataset_intrinsics = {}
        height, width = image_processor.size["height"], image_processor.size["width"]

        # scale intrinsic matrix
        for k, v in intrinsic_config.items():
            K = torch.tensor(v["intrinsic"]).float()
            K[:2] *= torch.tensor([width / v["width"], height / v["height"]])[:, None]
            self.dataset_intrinsics[k] = K
        
        self.action_tokenizer = SpatialActionTokenizer(
            tokenizer=tokenizer, num_bins=action_config["num_bins"], 
            bin_policy=bin_policy, use_spherical=action_config["use_spherical"],
            min_sigma=min_sigma,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        unnorm_key: Optional[str] = None,
        suffix_actions: Optional[np.array] = None, # (t e)
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if suffix_actions is not None:
            action_tokens = self.action_tokenizer(suffix_actions) # (n,3)
            suffix="".join(action_tokens.flatten())
        else:
            suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once( "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model.")
            text = ""

        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                        )
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")
                if suffix is not None and _is_str_or_image(suffix): suffix = [suffix]
                if suffix is not None: suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]
                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    for prompt, image_list in zip(text, images)
                ]
                images = make_batched_images(images)
            else:
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                        expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        intrinsic = self.dataset_intrinsics[unnorm_key] if unnorm_key in self.dataset_intrinsics else self.dataset_intrinsics["default"]
        return_data = {**inputs, "pixel_values": pixel_values, "intrinsic": intrinsic}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def decode_actions(
        self,
        generation_outputs: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        action_token_num = 3  # translation + rotation + gripper
        predicted_action_token_ids = generation_outputs[0, : action_token_num * self.action_chunk_size].detach().cpu().long().numpy()
        assert self.tokenizer.eos_token != predicted_action_token_ids[-1], "[error] actions contain EOS token, please check you truncation settings!"

        if predicted_action_token_ids.shape[0] < action_token_num * self.action_chunk_size:  # pad with zeros
            logger.warning(f"Padding zero action!")
            predicted_action_token_ids = np.concatenate(
                [
                    predicted_action_token_ids,
                    np.zeros(action_token_num * self.action_chunk_size - predicted_action_token_ids.shape[0], dtype=np.longlong),
                ]
            )
        predicted_action_token_ids = predicted_action_token_ids.reshape(-1, action_token_num)
        normalized_action_chunks = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids)

        if unnorm_key is None:
            logger.warning(f"unnorm_key {unnorm_key} is not in statistics, use next one")
            unnorm_key = next(self.statistics.keys())
        action_norm_stats = self.statistics[unnorm_key]["action"]

        action_dim = len(action_norm_stats["q01"])
        mask = np.array(action_norm_stats.get("mask", np.ones(action_dim)), dtype=bool)
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

        actions = []
        for normalized_actions in normalized_action_chunks:
            action = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            actions.append(action)
        actions = np.stack(actions)
        return {"actions": actions, "action_ids": predicted_action_token_ids}