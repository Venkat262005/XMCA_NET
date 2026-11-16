"""Unit tests for HALP activation extractor and layer-index tuple mapping."""

import torch
import pytest

from src.baselines.halp_extractor import HALPActivationExtractor, HALPExtractedSample


def test_halp_tuple_index_computation():
    # LLaVA (32 layers): 1-based layers {1, 8, 16, 24, 32} -> tuple indices {1, 8, 16, 24, 32}
    indices_llava = HALPActivationExtractor._compute_hidden_state_tuple_indices(32)
    assert indices_llava["pos_1"] == 1
    assert indices_llava["pos_L_div_4"] == 8
    assert indices_llava["pos_L_div_2"] == 16
    assert indices_llava["pos_3L_div_4"] == 24
    assert indices_llava["pos_L"] == 32

    # Qwen2-VL (28 layers): 1-based layers {1, 7, 14, 21, 28} -> tuple indices {1, 7, 14, 21, 28}
    indices_qwen = HALPActivationExtractor._compute_hidden_state_tuple_indices(28)
    assert indices_qwen["pos_1"] == 1
    assert indices_qwen["pos_L_div_4"] == 7
    assert indices_qwen["pos_L_div_2"] == 14
    assert indices_qwen["pos_3L_div_4"] == 21
    assert indices_qwen["pos_L"] == 28

    # InternVL2 (24 layers): 1-based layers {1, 6, 12, 18, 24} -> tuple indices {1, 6, 12, 18, 24}
    indices_intern = HALPActivationExtractor._compute_hidden_state_tuple_indices(24)
    assert indices_intern["pos_1"] == 1
    assert indices_intern["pos_L_div_4"] == 6
    assert indices_intern["pos_L_div_2"] == 12
    assert indices_intern["pos_3L_div_4"] == 18
    assert indices_intern["pos_L"] == 24


def test_halp_extraction_from_hidden_states():
    L = 32
    hidden_dim = 64
    seq_len = 20
    batch_size = 1

    # Simulate Hugging Face output: tuple of length L + 1 (embedding + 32 layers)
    hidden_states = tuple(
        torch.randn(batch_size, seq_len, hidden_dim) for _ in range(L + 1)
    )
    vf_pooled = torch.randn(batch_size, 32)

    extractor = HALPActivationExtractor(model_family="llava", num_layers=L, hidden_dim=hidden_dim)
    sample = extractor.extract_from_forward(
        hidden_states=hidden_states,
        vf_pooled_tensor=vf_pooled,
        vt_pos=5,
        qt_pos=15,
        sample_id="test_sample_01",
        dataset_id="pope",
        partition="dev_train",
    )

    assert isinstance(sample, HALPExtractedSample)
    assert sample.sample_id == "test_sample_01"
    assert sample.has_nan is False
    assert sample.has_inf is False
    assert len(sample.vf_pooled) == 32
    assert len(sample.vt_pos_1) == hidden_dim
    assert len(sample.vt_pos_L) == hidden_dim
    assert len(sample.qt_pos_1) == hidden_dim
    assert len(sample.qt_pos_L) == hidden_dim

