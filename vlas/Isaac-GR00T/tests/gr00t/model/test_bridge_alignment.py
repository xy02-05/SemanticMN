import json
from types import SimpleNamespace

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.trainer import StopAfterStepsCallback
from gr00t.model.alignment import BridgeAlignmentAdapter
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import pytest
import torch


def build_adapter(tmp_path, loss_weight=2.5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    texts = np.asarray(["pick red", "pick crimson", "noise", "place blue"], dtype=object)
    embedding_path = tmp_path / "embedding.npz"
    np.savez(
        embedding_path,
        sentence_embeddings=np.arange(32, dtype=np.float32).reshape(4, 8),
        task_index=np.arange(4, dtype=np.int64),
        texts=texts,
    )
    task_meta_path = tmp_path / "tasks_with_id.jsonl"
    task_meta = [
        {"task_index": 0, "task": "pick red", "task_id": 5},
        {"task_index": 1, "task": "pick crimson", "task_id": 5},
        {"task_index": -1, "task": "noise", "task_id": -1},
        {"task_index": 3, "task": "place blue", "task_id": 6},
    ]
    task_meta_path.write_text("\n".join(json.dumps(row) for row in task_meta) + "\n")
    dataset_task_path = tmp_path / "tasks.jsonl"
    dataset_tasks = [
        {"task_index": 0, "task": "PICK RED"},
        {"task_index": 1, "task": "pick crimson"},
        {"task_index": 2, "task": "noise"},
        {"task_index": 3, "task": "place blue"},
    ]
    dataset_task_path.write_text(
        "\n".join(json.dumps(row) for row in dataset_tasks) + "\n"
    )

    config = {
        "arch": {
            "args": {
                "embeddings_path": str(embedding_path),
                "index_path": None,
                "task_meta_path": str(task_meta_path),
                "dataset_task_path": str(dataset_task_path),
            }
        },
        "alignment": {
            "use_alignment": True,
            "args": {
                "egovlpv2_dim": 8,
                "openvla_dim": 8,
                "projection_dim": 4,
                "token_egovlpv2_dim": 8,
                "temperature": 0.1,
                "dropout": 0.0,
                "layer_norm_eps": 1e-5,
                "layer_indices": [1],
                "text_proj_config": {
                    "use_fc_projection": True,
                    "proj_num_layers": 1,
                    "proj_hidden_dim": 8,
                },
                "action_proj_config": {
                    "use_fc_projection": False,
                    "proj_num_layers": 1,
                    "proj_hidden_dim": 8,
                },
                "use_feature_bank": False,
                "feature_bank_size": 8,
                "alignment_mode": "task_id",
                "vla_mode": "gr00t",
                "min_valid_ratio": 0.0,
                "use_distributed_negatives": False,
                "loss_type": "sigmoid",
                "sigmoid_bias": 1.5,
                "learnable_temperature": False,
                "learnbale_bias": False,
                "action_pool_mode": "mean",
                "action_pool_config": {},
                "use_disentangle": False,
                "mode_config": {
                    "as2ts": {"enabled": True, "weight": 1.0},
                    "as2tt": {"enabled": False, "weight": 0.0},
                    "at2tt": {"enabled": False, "weight": 0.0},
                    "at2tt_soft": {"enabled": False, "weight": 0.0},
                },
            },
            "training": {"backbone_update_start_pct": 0.1},
        },
    }
    config_path = tmp_path / "alignment.json"
    config_path.write_text(json.dumps(config))
    return BridgeAlignmentAdapter(str(config_path), loss_weight)


def make_inputs():
    hidden_states = [
        torch.randn(3, 5, 8, requires_grad=True),
        torch.randn(3, 5, 8, requires_grad=True),
    ]
    action_mask = torch.tensor(
        [
            [[1, 1], [1, 1], [1, 1], [0, 0]],
            [[1, 1], [1, 1], [1, 1], [0, 0]],
            [[1, 1], [1, 1], [1, 1], [0, 0]],
        ],
        dtype=torch.float32,
    )
    return hidden_states, action_mask


def test_mapping_padding_and_multi_positive(tmp_path):
    adapter = build_adapter(tmp_path)
    hidden_states, action_mask = make_inputs()
    selected = adapter._select_action_features(hidden_states, action_mask)
    torch.testing.assert_close(selected[:, 0], hidden_states[1][:, -4:-1])

    adapter.set_training_progress(global_step=100, max_steps=1000)
    output = adapter(hidden_states, action_mask, torch.tensor([0, 1, 3]))
    assert torch.isfinite(output["raw_loss"])
    torch.testing.assert_close(output["weighted_loss"], output["raw_loss"] * 2.5)
    assert output["loss_dict"]["as2ts_pos_samples"] == pytest.approx(5 / 3)


def test_backbone_detach_boundary_and_invalid_tasks(tmp_path):
    adapter = build_adapter(tmp_path)
    hidden_states, action_mask = make_inputs()
    adapter.set_training_progress(global_step=99, max_steps=1000)
    detached_output = adapter(hidden_states, action_mask, torch.tensor([0, 1, 3]))
    detached_output["weighted_loss"].backward()
    assert hidden_states[1].grad is None
    assert any(parameter.grad is not None for parameter in adapter.parameters())

    adapter.zero_grad(set_to_none=True)
    hidden_states, action_mask = make_inputs()
    adapter.set_training_progress(global_step=100, max_steps=1000)
    attached_output = adapter(hidden_states, action_mask, torch.tensor([0, 1, 3]))
    attached_output["weighted_loss"].backward()
    assert hidden_states[1].grad is not None
    assert hidden_states[1].grad.abs().sum() > 0

    adapter.zero_grad(set_to_none=True)
    hidden_states, action_mask = make_inputs()
    invalid_output = adapter(hidden_states, action_mask, torch.tensor([2, 2, 2]))
    assert invalid_output["raw_loss"].item() == 0.0
    assert torch.isfinite(invalid_output["raw_loss"])
    invalid_output["raw_loss"].backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    assert all(parameter.grad.abs().sum() == 0 for parameter in adapter.parameters())


def test_warmup_preserves_action_loss_backbone_gradient(tmp_path):
    adapter = build_adapter(tmp_path)
    hidden_states, action_mask = make_inputs()
    adapter.set_training_progress(global_step=99, max_steps=1000)

    # 用 hidden state 上的可微标量模拟 GR00T action loss；warmup 期间总梯度应与
    # 纯 action loss 完全一致，而 alignment head 仍应得到自身梯度。
    action_loss = hidden_states[1].square().mean()
    expected_action_grad = 2 * hidden_states[1].detach() / hidden_states[1].numel()
    alignment_output = adapter(hidden_states, action_mask, torch.tensor([0, 1, 3]))
    (action_loss + alignment_output["weighted_loss"]).backward()

    torch.testing.assert_close(hidden_states[1].grad, expected_action_grad)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adapter.parameters()
    )

    adapter.zero_grad(set_to_none=True)
    hidden_states, action_mask = make_inputs()
    adapter.set_training_progress(global_step=100, max_steps=1000)
    action_loss = hidden_states[1].square().mean()
    expected_action_grad = 2 * hidden_states[1].detach() / hidden_states[1].numel()
    alignment_output = adapter(hidden_states, action_mask, torch.tensor([0, 1, 3]))
    (action_loss + alignment_output["weighted_loss"]).backward()

    assert not torch.allclose(hidden_states[1].grad, expected_action_grad)


def test_mixed_valid_and_invalid_tasks(tmp_path):
    adapter = build_adapter(tmp_path)
    hidden_states, action_mask = make_inputs()
    adapter.set_training_progress(global_step=100, max_steps=1000)

    output = adapter(hidden_states, action_mask, torch.tensor([0, 2, 3]))
    assert torch.isfinite(output["raw_loss"])
    assert output["raw_loss"].item() > 0
    output["weighted_loss"].backward()
    assert hidden_states[1].grad is not None
    assert hidden_states[1].grad.abs().sum() > 0


def test_checkpoint_excludes_offline_tables(tmp_path):
    adapter = build_adapter(tmp_path)
    state_dict = adapter.state_dict()
    assert "text_embeddings" not in state_dict
    assert "task_ids" not in state_dict
    assert "dataset_to_embedding_index" not in state_dict

    restored = build_adapter(tmp_path / "restored")
    restored.load_state_dict(state_dict, strict=True)
    adapter.eval()
    restored.eval()
    hidden_states, action_mask = make_inputs()
    restored_hidden = [hidden.detach().clone().requires_grad_() for hidden in hidden_states]
    adapter.set_training_progress(global_step=100, max_steps=1000)
    restored.set_training_progress(global_step=100, max_steps=1000)
    output = adapter(hidden_states, action_mask, torch.tensor([0, 1, 3]))
    restored_output = restored(restored_hidden, action_mask, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(output["raw_loss"], restored_output["raw_loss"])


def test_stop_after_steps_callback():
    callback = StopAfterStepsCallback(stop_after_steps=2500)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    state = SimpleNamespace(global_step=2499)
    callback.on_step_end(None, state, control)
    assert not control.should_save
    assert not control.should_training_stop

    state.global_step = 2500
    callback.on_step_end(None, state, control)
    assert control.should_save
    assert control.should_training_stop


def test_inference_policy_disables_training_only_alignment(monkeypatch, tmp_path):
    calls = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, **kwargs):
            calls["model_to"] = kwargs
            return self

    class FakeProcessor:
        collator = object()

        def eval(self):
            return self

        def get_modality_configs(self):
            return {
                EmbodimentTag.OXE_WIDOWX.value: {
                    "language": SimpleNamespace(
                        modality_keys=["task"],
                        delta_indices=[0],
                    )
                }
            }

    def fake_model_from_pretrained(model_dir, **kwargs):
        calls["model_dir"] = model_dir
        calls["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        "gr00t.policy.gr00t_policy.AutoModel.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "gr00t.policy.gr00t_policy.AutoProcessor.from_pretrained",
        lambda model_dir: FakeProcessor(),
    )

    Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=str(tmp_path),
        device="cpu",
    )

    assert calls["model_kwargs"]["use_alignment"] is False
