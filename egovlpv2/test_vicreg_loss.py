import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parent / "egovlpv2/model/vicreg_loss.py"
SPEC = importlib.util.spec_from_file_location("vicreg_loss_under_test", MODULE_PATH)
VICREG_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VICREG_MODULE)
VICRegLoss = VICREG_MODULE.VICRegLoss


def _invariance_only_loss():
    return VICRegLoss(sim_coeff=1.0, std_coeff=0.0, cov_coeff=0.0)


def _multi_positive_mse(action, text, sample_ids):
    # 与 InfoNCE 一致：先平均每个 anchor 的正样本，再平均全部 anchors。
    positive_mask = sample_ids[:, None].eq(sample_ids[None, :])
    pairwise_mse = (action[:, None, :] - text[None, :, :]).pow(2).mean(dim=-1)
    positives_per_anchor = positive_mask.sum(dim=1)
    return (
        (pairwise_mse * positive_mask).sum(dim=1) / positives_per_anchor
    ).mean()


def test_sample_ids_expand_invariance_to_all_same_id_pairs():
    action = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    text = action.clone()
    sample_ids = torch.tensor([1, 1, 2])

    loss, stats = _invariance_only_loss()(
        action,
        text,
        sample_ids=sample_ids,
        gather_distributed=False,
    )

    expected = _multi_positive_mse(action, text, sample_ids)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(stats["vicreg_invariance_loss"], expected)
    assert stats["vicreg_positive_pairs"].item() == 5


def test_unique_sample_ids_keep_original_paired_vicreg_behavior():
    torch.manual_seed(7)
    action = torch.randn(5, 4)
    text = torch.randn(5, 4)
    sample_ids = torch.arange(5)
    loss_fn = VICRegLoss()

    paired_loss, _ = loss_fn(action, text, gather_distributed=False)
    grouped_loss, stats = loss_fn(
        action,
        text,
        sample_ids=sample_ids,
        gather_distributed=False,
    )

    torch.testing.assert_close(grouped_loss, paired_loss)
    assert stats["vicreg_positive_pairs"].item() == action.shape[0]


def test_invalid_samples_are_removed_before_id_matching():
    action = torch.tensor(
        [[0.0, 0.0], [4.0, 0.0], [1000.0, 1000.0], [8.0, 0.0]]
    )
    text = action.clone()
    sample_ids = torch.tensor([3, 3, 3, 9])
    valid_mask = torch.tensor([True, True, False, True])

    loss, stats = _invariance_only_loss()(
        action,
        text,
        valid_mask=valid_mask,
        sample_ids=sample_ids,
        gather_distributed=False,
    )

    expected = _multi_positive_mse(
        action[valid_mask], text[valid_mask], sample_ids[valid_mask]
    )
    torch.testing.assert_close(loss, expected)
    assert stats["vicreg_valid_pairs"].item() == 3
    assert stats["vicreg_positive_pairs"].item() == 5


def test_sample_ids_must_match_batch_shape():
    action = torch.randn(3, 2)
    text = torch.randn(3, 2)

    with pytest.raises(ValueError, match="sample_ids must have shape"):
        _invariance_only_loss()(
            action,
            text,
            sample_ids=torch.tensor([1, 2]),
            gather_distributed=False,
        )


def test_multi_positive_invariance_backpropagates_to_both_modalities():
    action = torch.randn(4, 3, requires_grad=True)
    text = torch.randn(4, 3, requires_grad=True)
    sample_ids = torch.tensor([2, 2, 5, 8])

    loss, _ = _invariance_only_loss()(
        action,
        text,
        sample_ids=sample_ids,
        gather_distributed=False,
    )
    loss.backward()

    assert torch.isfinite(action.grad).all()
    assert torch.isfinite(text.grad).all()
    assert action.grad.abs().sum() > 0
    assert text.grad.abs().sum() > 0


def test_vicreg_stats_support_alignment_model_mean_aggregation():
    action = torch.randn(3, 2)
    text = torch.randn(3, 2)
    sample_ids = torch.tensor([1, 1, 2])

    _, stats = _invariance_only_loss()(
        action,
        text,
        sample_ids=sample_ids,
        gather_distributed=False,
    )
    aggregated = {
        key: torch.stack([value, value]).mean().item()
        for key, value in stats.items()
        if key.startswith("vicreg_")
    }

    assert aggregated["vicreg_positive_pairs"] == 5.0


def test_multi_positive_mse_is_stable_with_large_common_offset():
    feature_dim = 512
    offset = torch.linspace(-0.2, 0.2, feature_dim)
    action = torch.stack(
        [10000.0 + offset, 10000.0 - offset, 10000.0 + 2 * offset]
    )
    text = action + torch.stack([offset, -offset, 0.5 * offset])
    sample_ids = torch.tensor([4, 4, 9])

    loss, _ = _invariance_only_loss()(
        action,
        text,
        sample_ids=sample_ids,
        gather_distributed=False,
    )
    expected = _multi_positive_mse(action, text, sample_ids)

    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
