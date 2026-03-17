import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *

# Hugging Face exposes Qwen1.5-MoE under the qwen2_moe model family.
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


@dataclass
class Qwen2MoeComponents:
    layers: nn.ModuleList
    embed_tokens: nn.Module
    norm: nn.Module


def find_qwen_moe_expert_ffn_layers(layer):
    if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'experts'):
        return {}

    subset = {}
    for expert_index, expert in enumerate(layer.mlp.experts):
        prefix = f'mlp.experts.{expert_index}'
        subset[f'{prefix}.gate_proj'] = expert.gate_proj
        subset[f'{prefix}.up_proj'] = expert.up_proj
        subset[f'{prefix}.down_proj'] = expert.down_proj
    return subset


def should_prune_qwen_moe_target(layer_idx, name, args):
    return (
        not (
            args.minlayer <= layer_idx < args.maxlayer
            and args.prune_only in name
        )
    ) != (not args.invert)


def get_qwen2_moe_components(model):
    config = getattr(model, 'config', None)
    if getattr(config, 'model_type', None) != 'qwen2_moe':
        raise TypeError('Expected a Qwen2-MoE model with config.model_type == "qwen2_moe".')

    backbone = getattr(model, 'model', None)
    if backbone is None:
        raise TypeError('Qwen2-MoE model is missing the top-level `model` module.')

    required_attrs = ('layers', 'embed_tokens', 'norm')
    missing = [name for name in required_attrs if not hasattr(backbone, name)]
    if missing:
        raise TypeError(
            'Qwen2-MoE backbone is missing required attributes: '
            + ', '.join(missing)
        )

    return Qwen2MoeComponents(
        layers=backbone.layers,
        embed_tokens=backbone.embed_tokens,
        norm=backbone.norm,
    )


def prepare_layer_kwargs(model_kwargs):
    layer_kwargs = {}
    for name, value in model_kwargs.items():
        if value is None:
            continue
        layer_kwargs[name] = value
    return layer_kwargs


def replay_decoder_layer(layer, hidden_states, layer_kwargs):
    output = layer(hidden_states, **layer_kwargs)
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, 'hidden_states'):
        return output.hidden_states
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state
    raise TypeError(f'Unsupported decoder layer output type: {type(output)!r}')


def get_qwen2_moe(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    get_qwen2_moe_components(model)
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def qwen_sequential(model, dataloader, dev, args):
    print('Starting ...')

    components = get_qwen2_moe_components(model)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = components.layers

    components.embed_tokens = components.embed_tokens.to(dev)
    model.model.embed_tokens = components.embed_tokens
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = prepare_layer_kwargs(kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs'] or {}

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_qwen_moe_expert_ffn_layers(layer)
        
        gpts = {}
        for name in subset:
            if not should_prune_qwen_moe_target(i, name, args):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = replay_decoder_layer(layer, inps[j].unsqueeze(0), layer_kwargs)
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = replay_decoder_layer(layer, inps[j].unsqueeze(0), layer_kwargs)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

@torch.no_grad()
def qwen_eval(model, testenc, dev, args, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    components = get_qwen2_moe_components(model)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = components.layers

    components.embed_tokens = components.embed_tokens.to(dev)
    model.model.embed_tokens = components.embed_tokens
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_kwargs'] = prepare_layer_kwargs(kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs'] or {}

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_qwen_moe_expert_ffn_layers(layer)
            for name in subset:
                if not should_prune_qwen_moe_target(i, name, args):
                    continue
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = replay_decoder_layer(layer, inps[j].unsqueeze(0), layer_kwargs)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, 
        help='Qwen1.5-MoE / Qwen2-MoE model to load.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_qwen2_moe(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        qwen_sequential(model, dataloader, DEV, args)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'mlp.experts.0.down_proj' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        qwen_eval(model, testloader, DEV, args, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
