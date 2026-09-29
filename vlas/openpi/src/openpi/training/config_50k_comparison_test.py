import dataclasses

from openpi.training import config as training_config


BASELINE_NAME = "pi0_libero_full_50k_gbs32"
ALIGNMENT_NAME = "pi0_libero_full_qwen_50k_gbs32"
RECON001_PREPOOL_NAME = "pi0_libero_full_qwen_recon001_prepool_50k_gbs32"
RECON001_POSTPOOL_NAME = "pi0_libero_full_qwen_recon001_postpool_50k_gbs32"
BASELINE_CONTINUE_NAME = "pi0_libero_full_continue_30k_50k_gbs32"
ALIGNMENT_CONTINUE_NAME = "pi0_libero_full_qwen_continue_30k_50k_gbs32"


def test_50k_configs_only_differ_in_alignment_fields():
    baseline = dataclasses.asdict(training_config.get_config(BASELINE_NAME))
    alignment = dataclasses.asdict(training_config.get_config(ALIGNMENT_NAME))

    allowed_differences = {
        "name",
        "model.vlm_mode",
        "use_alignment",
        "egovlpv2_config_path",
        "alignment_loss_weight",
    }

    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                result.update(flatten(child, child_prefix))
        else:
            result[prefix] = value
        return result

    baseline_flat = flatten(baseline)
    alignment_flat = flatten(alignment)
    differences = {
        key
        for key in baseline_flat.keys() | alignment_flat.keys()
        if baseline_flat.get(key) != alignment_flat.get(key)
    }
    assert differences == allowed_differences


def test_50k_global_batch_and_schedule():
    for name in (
        BASELINE_NAME,
        ALIGNMENT_NAME,
        RECON001_PREPOOL_NAME,
        RECON001_POSTPOOL_NAME,
    ):
        config = training_config.get_config(name)
        assert config.batch_size == 32
        assert config.per_device_batch_size is None
        assert config.gradient_accumulation_steps == 1
        assert config.num_train_steps == 50_000
        assert config.save_interval == 5_000
        assert config.lr_schedule.warmup_steps == 1_000
        assert config.lr_schedule.peak_lr == 2.5e-5
        assert config.lr_schedule.decay_steps == 30_000
        assert config.lr_schedule.decay_lr == 2.5e-6


def test_recon001_configs_only_change_alignment_position():
    prepool = training_config.get_config(RECON001_PREPOOL_NAME)
    postpool = training_config.get_config(RECON001_POSTPOOL_NAME)
    assert prepool.egovlpv2_config_path != postpool.egovlpv2_config_path

    import json

    with open(prepool.egovlpv2_config_path) as handle:
        prepool_alignment = json.load(handle)
    with open(postpool.egovlpv2_config_path) as handle:
        postpool_alignment = json.load(handle)

    prepool_args = prepool_alignment["alignment"]["args"]
    postpool_args = postpool_alignment["alignment"]["args"]
    assert prepool_args["disentangle_config"]["position"] == "pre_pool"
    assert postpool_args["disentangle_config"]["position"] == "post_pool"
    assert prepool_args["disentangle_config"]["recon_weight"] == 0.001
    assert postpool_args["disentangle_config"]["recon_weight"] == 0.001

    prepool_args["disentangle_config"]["position"] = "post_pool"
    prepool_alignment["name"] = postpool_alignment["name"]
    assert prepool_alignment == postpool_alignment


def test_30k_continuation_configs():
    baseline = training_config.get_config(BASELINE_CONTINUE_NAME)
    alignment = training_config.get_config(ALIGNMENT_CONTINUE_NAME)
    for config in (baseline, alignment):
        assert config.resume
        assert config.allow_missing_optimizer_state
        assert config.optimizer_restart_step == 30_000
        assert config.optimizer_restart_warmup_steps == 200
        assert config.optimizer_restart_warmup_init_ratio == 0.1
        assert config.batch_size == 32
        assert config.per_device_batch_size is None
        assert config.num_train_steps == 50_000
        assert config.save_interval == 5_000
    assert not baseline.use_alignment
    assert alignment.use_alignment
    assert alignment.alignment_loss_weight == 0.1

    baseline_flat = dataclasses.asdict(baseline)
    alignment_flat = dataclasses.asdict(alignment)
    allowed_differences = {
        "name",
        "model.vlm_mode",
        "use_alignment",
        "egovlpv2_config_path",
        "alignment_loss_weight",
    }

    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                result.update(flatten(child, child_prefix))
        else:
            result[prefix] = value
        return result

    baseline_values = flatten(baseline_flat)
    alignment_values = flatten(alignment_flat)
    differences = {
        key
        for key in baseline_values.keys() | alignment_values.keys()
        if baseline_values.get(key) != alignment_values.get(key)
    }
    assert differences == allowed_differences
