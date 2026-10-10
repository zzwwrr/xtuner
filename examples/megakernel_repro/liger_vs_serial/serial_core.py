import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.distributed.nn.functional import all_gather, all_reduce


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, probabilities, indices, buffer, experts, state):
        layout = buffer.get_dispatch_layout(indices, experts)
        received, local_ids, weights, counts, handle, _ = buffer.dispatch(
            inputs, topk_idx=indices, topk_weights=probabilities,
            num_tokens_per_rank=layout[0], num_tokens_per_rdma_rank=layout[1],
            num_tokens_per_expert=layout[2], is_token_in_rank=layout[3],
            async_finish=False,
        )
        ctx.buffer, ctx.handle = buffer, handle
        state.update(handle=handle, counts=counts)
        ctx.mark_non_differentiable(local_ids)
        return received, weights, local_ids

    @staticmethod
    def backward(ctx, grad_inputs, grad_weights, grad_ids):
        combined, weights, _ = ctx.buffer.combine(
            grad_inputs.contiguous(), ctx.handle,
            topk_weights=grad_weights.contiguous(), async_finish=False,
        )
        return combined, weights, None, None, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, buffer, handle):
        ctx.buffer, ctx.handle = buffer, handle
        combined, _, _ = buffer.combine(inputs, handle, async_finish=False)
        return combined

    @staticmethod
    def backward(ctx, gradients):
        received, _, _, _, _, _ = ctx.buffer.dispatch(
            gradients.contiguous(), handle=ctx.handle, async_finish=False,
        )
        return received, None, None


def grouped_experts(inputs, probabilities, indices, up, down, local_experts):
    rows, slots = (indices >= 0).nonzero(as_tuple=True)
    expert_ids = indices[rows, slots].long()
    order = torch.argsort(expert_ids, stable=True)
    rows, slots = rows[order], slots[order]
    counts = torch.bincount(expert_ids, minlength=local_experts).tolist()
    gate, value = up(inputs[rows], counts).chunk(2, dim=-1)
    outputs = down(functional.silu(gate) * value, counts)
    weighted = (outputs.float() * probabilities[rows, slots, None]).to(inputs.dtype)
    return inputs.new_zeros(inputs.shape).index_add(0, rows, weighted)


def serial_forward(inputs, probabilities, indices, up, down, buffer, experts):
    state = {}
    received, weights, local_ids = Dispatch.apply(
        inputs, probabilities, indices, buffer, experts, state,
    )
    outputs = grouped_experts(received, weights, local_ids, up, down, up.num_gemms)
    return Combine.apply(outputs, buffer, state['handle'])


def reference_forward(inputs, probabilities, indices, up_weights, down_weights, group):
    gathered_inputs = torch.cat(all_gather(inputs, group=group))
    gathered_probabilities = torch.cat(all_gather(probabilities, group=group))
    gathered_ids = [torch.empty_like(indices) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered_ids, indices, group=group)
    gathered_ids = torch.cat(gathered_ids)
    offset = dist.get_rank(group) * len(up_weights)
    outputs = gathered_inputs * 0
    for expert, (up_weight, down_weight) in enumerate(zip(up_weights, down_weights)):
        rows, slots = (gathered_ids == offset + expert).nonzero(as_tuple=True)
        gate, value = functional.linear(gathered_inputs[rows], up_weight).chunk(2, dim=-1)
        values = functional.linear(functional.silu(gate) * value, down_weight)
        outputs = outputs.index_add(0, rows, values * gathered_probabilities[rows, slots, None])
    combined = all_reduce(outputs, group=group)
    return combined.chunk(dist.get_world_size(group))[dist.get_rank(group)]