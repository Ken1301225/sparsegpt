import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


MODULE_PATH = Path(__file__).with_name("Qwen1.5_MoE.py")
SPEC = importlib.util.spec_from_file_location("qwen15_moe_module", MODULE_PATH)
qwen15_moe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qwen15_moe)


class FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)


class FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMlp()

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer()])
        self.embed_tokens = nn.Embedding(16, 4)
        self.norm = nn.LayerNorm(4)


class FakeQwen2MoeForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.config = SimpleNamespace(
            model_type="qwen2_moe",
            use_cache=True,
            hidden_size=4,
            max_position_embeddings=8,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.seqlen = 8

    def forward(self, input_ids):
        hidden_states = self.model.embed_tokens(input_ids)
        layer = self.model.layers[0]
        return layer(
            hidden_states,
            attention_mask=torch.ones(
                input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1]
            ),
            position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0),
        )


class FakeNonQwenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            layers=nn.ModuleList([FakeLayer()]),
            embed_tokens=nn.Embedding(16, 4),
            norm=nn.LayerNorm(4),
        )
        self.config = SimpleNamespace(model_type="llama", use_cache=True, hidden_size=4)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.seqlen = 8


def test_find_qwen_moe_expert_ffn_layers_only_returns_expert_ffn_projections():
    subset = qwen15_moe.find_qwen_moe_expert_ffn_layers(FakeLayer())

    assert sorted(subset) == [
        "mlp.experts.0.down_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.1.down_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.1.up_proj",
    ]


def test_get_qwen2_moe_components_uses_qwen2_moe_model_layout():
    model = FakeQwen2MoeForCausalLM()

    components = qwen15_moe.get_qwen2_moe_components(model)

    assert components.layers is model.model.layers
    assert components.embed_tokens is model.model.embed_tokens
    assert components.norm is model.model.norm


def test_get_qwen2_moe_components_rejects_non_qwen2_moe_models():
    with pytest.raises(TypeError, match="Qwen2-MoE"):
        qwen15_moe.get_qwen2_moe_components(FakeNonQwenModel())


def test_prepare_layer_kwargs_preserves_required_replay_inputs():
    model_kwargs = {
        "attention_mask": torch.ones(1, 1, 8, 8),
        "position_ids": torch.arange(8).unsqueeze(0),
        "cache_position": torch.arange(8),
        "unused": None,
    }

    layer_kwargs = qwen15_moe.prepare_layer_kwargs(model_kwargs)

    assert set(layer_kwargs) >= {"attention_mask", "position_ids", "cache_position"}
    assert layer_kwargs["attention_mask"] is model_kwargs["attention_mask"]
    assert layer_kwargs["position_ids"] is model_kwargs["position_ids"]


def test_replay_layer_returns_first_hidden_state_tensor():
    layer = FakeLayer()
    hidden_states = torch.randn(1, 8, 4)

    output = qwen15_moe.replay_decoder_layer(
        layer,
        hidden_states,
        {"attention_mask": torch.ones(1, 1, 8, 8)},
    )

    assert output.shape == hidden_states.shape


def test_qwen_sequential_uses_qwen2_moe_structure_without_decoder_access(monkeypatch):
    model = FakeQwen2MoeForCausalLM()
    args = SimpleNamespace(
        nsamples=1,
        minlayer=-1,
        maxlayer=1000,
        prune_only="",
        invert=False,
        wbits=16,
        sparsity=0.0,
        prunen=0,
        prunem=0,
        percdamp=0.01,
        blocksize=128,
    )
    dataloader = [(torch.randint(0, 16, (1, 8)),)]

    monkeypatch.setattr(qwen15_moe.torch.cuda, "empty_cache", lambda: None)

    qwen15_moe.qwen_sequential(model, dataloader, torch.device("cpu"), args)

    assert model.config.use_cache is True


def test_qwen2_moe_integration_target_discovery():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        decoder_sparse_step=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
    )
    model = transformers.Qwen2MoeForCausalLM(config)

    subset = qwen15_moe.find_qwen_moe_expert_ffn_layers(model.model.layers[0])

    assert sorted(subset) == [
        "mlp.experts.0.down_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.1.down_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.1.up_proj",
    ]
