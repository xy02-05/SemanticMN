import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import torch
from torch import nn
import torch.distributed as dist
from train.dist_utils import init_dist
from train.monkey_patch import (
    replace_train_dataloader,
    replace_compute_loss,
    concat_pad_data_collator,
    replace_train_sampler,
    SaveProcessorCallback
)
import transformers
from transformers import (
    AutoTokenizer,
    ZoeDepthConfig,
    ZoeDepthForDepthEstimation,
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from data.dataset import build_datasets
from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
    SpatialActionTokenizer,
    Gemma2ForCausalLM
)
replace_train_dataloader()
replace_compute_loss()
replace_train_sampler()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(default=None,
        metadata={"help": "Path to pretrained model or identifier for resume training."},
    )
    vision_zoe_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or identifier for zoe model."},
    )
    vlm_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or identifier for vlm model."},
    )
    use_vision_zoe: bool = field(
        default=True, metadata={"help": "Set to True to use vision zoe model."},
    )
    freeze_llm_embed: bool = field(
        default=True, metadata={"help": "Set to True to freeze the LLM embeddings."},
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    action_config: Path = field(
        default="scripts/action_config.json",
        metadata={"help": "path to the action config file."},
    )
    n_freqs: Optional[int] = field(
        default=8, metadata={"help": "Number of frequencies for ego3d."}
    )
    ego3d_patch_reso: Optional[int] = field(
        default=2, metadata={"help": "resoluation of ego3d."}
    )
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Set to True to use flash attention."},
    )
    min_sigma: float = field(
        default=0.0,
        metadata={"help": "Set the minimum sigma for creating action grids."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "The root directory of the dataset. Default is `data`."},
    )
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "The name of the dataset mixture. Default is `bridge`."},
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization. "},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=1000_000,
        metadata={"help": "The shuffle buffer size for the dataset. Default is 1000000."},
    )
    tsfm_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds transfom. Default is 1."},
    )
    read_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds reader. Default is 1."},
    )
    obs_backward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of backward steps in observation. 0 indicates current"},
    )
    obs_backward_delta: Optional[int] = field(
        default=1, metadata={"help": "Backward delta in observation."}
    )
    action_forward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of forward steps in action. 0 indicates current"},
    )
    fix_raw_length: Optional[int] = field(
        default=None, metadata={"help": "fix the iterable dataset iter length."}
    )
    use_raw_dataloader: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use raw dataloader"}
    )
    intrinsic_config_path: Path = field(
        default="scripts/intrinsics.json",
        metadata={"help": "path to the intrinsic config file."},
    )
    normalized_statistic_path: Path = field(
        default=None,
        metadata={"help": "path to the normalized statistic file."},
    )

def main():
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log: transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # 1. initializing models and load tokenizer
    if model_args.model_name_or_path is not None:
        processor = SpatialVLAProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
        spatial_token_num = processor.action_tokenizer.vocab_size
        action_tokenizer = processor.action_tokenizer
        tokenizer = processor.tokenizer
        tokenizer.model_max_length = data_args.max_seq_length
    else:
        # Load pretrained model, tokenizer, and image processor
        action_config = json.load(open(model_args.action_config))
        spatial_token_num = action_config["num_bins"]["total"]
        print(f"will add {spatial_token_num} spatial action tokens")
        tokenizer_path = model_args.model_name_or_path or model_args.vlm_path
        logger.info(f"Loading Tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True, local_files_only=True)
        tokenizer.tokenizer_path = tokenizer_path
        tokenizer.model_max_length = data_args.max_seq_length

    # load models
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    print(f"torch_dtype {torch_dtype}, {training_args.bf16}")
    if model_args.model_name_or_path is not None:
        logger.info("Loading SpatialVLA Model...")
        config = SpatialVLAConfig.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        config.use_spatial_token = model_args.freeze_llm_embed
        model = SpatialVLAForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if model_args.flash_attn:
            model.language_model.config._attn_implementation = model.config.text_config._attn_implementation_internal = "flash_attention_2"
            model.vision_tower.config._attn_implementation = model.config.vision_config._attn_implementation_internal = "flash_attention_2"
    else:
        logger.info("Loading Vision Model...")
        flash_attn_args = {"attn_implementation": "flash_attention_2"} if model_args.flash_attn else {}
        paligemma_config = PaliGemmaConfig.from_pretrained(
            model_args.vlm_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
            **flash_attn_args,
        )
        paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_args.vlm_path,
            config=paligemma_config,
            torch_dtype=torch_dtype,
            revision="bfloat16",
            local_files_only=True,
        )
        # flash-attn of gemma2 is problematic, we replace it with customized gemma2
        gemma2 = Gemma2ForCausalLM(paligemma_config.text_config)
        gemma2.load_state_dict(paligemma_model.language_model.state_dict())
        vision_zoe_config = ZoeDepthConfig.from_pretrained(
            model_args.vision_zoe_path, torch_dtype=torch_dtype, local_files_only=True
        )
        vision_zoe_model = ZoeDepthForDepthEstimation.from_pretrained(  # zoe does not support Flash Attention 2.0 yet.
            model_args.vision_zoe_path,
            config=vision_zoe_config,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        logger.info("Building SpatialVLAConfig...")
        config = SpatialVLAConfig(
            **paligemma_config.to_dict(),
            vision_zoe_config=vision_zoe_config.to_dict(),
            spatial_token_num=spatial_token_num,
            use_spatial_token=model_args.freeze_llm_embed,
            ego3d_patch_reso=model_args.ego3d_patch_reso,
            n_freqs=model_args.n_freqs,
            use_vision_zoe=model_args.use_vision_zoe,
            **flash_attn_args,
        )
        logger.info("Building SpatialVLA Model...")
        model = SpatialVLAForConditionalGeneration(
            config,
            vision_model=paligemma_model.vision_tower,
            vision_zoe_model=vision_zoe_model,
            projector_model=paligemma_model.multi_modal_projector,
            language_model=gemma2,
        )

    # 2. build datasets
    train_dataset, eval_dataset = build_datasets(
        data_args,
        training_args.output_dir,
        vla_processor=None,
    )

    # 3. build action tokenizer
    if model_args.model_name_or_path is None:
        gs_params = json.load(open(data_args.normalized_statistic_path))
        action_tokenizer = SpatialActionTokenizer(
            tokenizer,
            num_bins=action_config["num_bins"],
            gs_params=gs_params,
            use_spherical=action_config["use_spherical"],
            min_sigma=model_args.min_sigma,
        )
        num_new_tokens = action_tokenizer.vocab_size
        assert num_new_tokens == spatial_token_num, "new tokens {num_new_tokens} must equal to spatial tokens {spatial_token_num}"

        # modify language embedding and lm_head if train from scratch
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            output_embeddings = model.language_model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # replace the llm head, freeze embedding tokens
        model.language_model.config.tie_word_embeddings = model.config.text_config.tie_word_embeddings = False
        new_lm_head = nn.Linear(model.config.text_config.hidden_size, model.config.text_config.vocab_size, bias=False)
        new_lm_head.weight.data = model.language_model.lm_head.weight.detach().clone()
        model.language_model.lm_head = new_lm_head
    else:
        num_new_tokens = action_tokenizer.vocab_size

    # overwrite attributes
    model.action_token_begin_idx = model.config.action_token_begin_idx = action_tokenizer.action_token_begin_idx
    model.vision_tower.gradient_checkpointing = True

    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    
    # set freeze params
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_llm_embed:
        model.language_model.model.embed_tokens.weight.requires_grad = False
        model.spatial_embed_tokens.weight.data = (model.language_model.model.embed_tokens.weight.data[-num_new_tokens:])

    if model_args.freeze_vision_tower:
        model.vision_tower = model.vision_tower.eval()
        _freeze_params(model.vision_tower)

    model.vision_zoe_model = model.vision_zoe_model.eval()
    _freeze_params(model.vision_zoe_model)

    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad: logger.info(name)

    set_seed(training_args.seed)
    SpatialVLAConfig.register_for_auto_class() # register for auto save and map
    SpatialVLAForConditionalGeneration.register_for_auto_class()
    SpatialVLAProcessor.register_for_auto_class()

    # build processor
    statistic = train_dataset.ds_stats_pc
    if model_args.model_name_or_path is None:
        intrinsic_config = json.load(open(data_args.intrinsic_config_path))
        paligemma_processor = PaliGemmaProcessor.from_pretrained(model_args.vlm_path, local_files_only=True)
        paligemma_processor.image_processor.do_normalize = False  # we nomalize in model, instead of processor
        processor = SpatialVLAProcessor(
            image_processor=paligemma_processor.image_processor,
            tokenizer=tokenizer,
            statistics=statistic,
            bin_policy=action_tokenizer.bin_policy,
            intrinsic_config=intrinsic_config,
            action_config=action_config,
            num_obs_steps=data_args.obs_backward_steps + 1,
            obs_delta=data_args.obs_backward_delta,
            action_chunk_size=data_args.action_forward_steps + 1,
        )
    else:
        processor.statistics.update(statistic)  # merge
    model.action_tokenizer = action_tokenizer
    train_dataset.vla_processor = processor

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator,
        callbacks=[SaveProcessorCallback(processor=processor)],
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
