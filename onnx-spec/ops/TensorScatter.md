# TensorScatter

First introduced in opset **24**

## Description

TensorScatter is a generic tensor update operation, motivated by the requirements for KV cache updates for Attention
ops commonly found in LLMs. It is a functional operation that models an in-place update to a KV cache buffer.

The past and present cache tensors have the same shape (batch_size, D1, D2, ..., max_sequence_length, ..., Dn), with
the sequence dimension (indicated by the `axis` attribute) being max_sequence_length, so the sizes of these tensors do
not need to grow between iterations. The `update` tensor's shape only differs from the cache tensors in the sequence
dimension: (batch_size, D1, D2, ..., sequence_length, ..., Dn), where sequence_length <= max_sequence_length.

The optional `write_indices` input indicates the write index for each sample in the batch, assumed to be zero
if not provided. When the `mode` attribute is set to "circular", the write index is modulo max_sequence_length.
The operation can be described using the following pseudocode:

```
for prefix_idx in np.ndindex(past_cache.shape[:axis]):
    batch_idx = prefix_idx[0]
    for sequence_idx in range(sequence_length):
        cache_idx = (*prefix_idx, write_indices[batch_idx] + sequence_idx)
        if mode == "circular":
            cache_idx = tuple(np.mod(np.asarray(cache_idx), max_sequence_length))
        update_idx = (*prefix_idx, sequence_idx)
        present_cache[cache_idx] = update[update_idx]
```

During the prefill phase of attention, only the first two inputs are needed. During the decode phase, `write_indices`
is also needed so that the incoming key or value update can be appended after the last valid token for each sample
in the batch.

## Attributes

- **axis** (INT, optional): Sequence dimension of the `past_cache` and `update` tensors. It cannot be 0 (the batch dimension). Default is -2.
- **mode** (STRING, optional): Write mode of cache update. Supported modes include `linear` and `circular`. `linear` mode requires write_indices+sequence_length<=max_sequence_length. For `circular` mode, the updates happen in wrap-around fashion, ie, the update index is modulo `max_sequence_length`

## Inputs (2 - 3)

- **past_cache** (T): Past state cache for key or value with shape `(batch_size, D1, D2, ..., max_sequence_length, ..., Dn)`.
- **update** (T): New update tensor with shape `(batch_size, D1, D2, ..., sequence_length, ..., Dn)`.
- **write_indices** (tensor(int64), optional): Write indices for the incoming update tensor in the cache. Shape is `(batch_size,)`. Assumed to be all zeros if not provided.

## Outputs (1 - 1)

- **present_cache** (T): Updated cache. Same shape as `past_cache`.

## Type Constraints

- **T**: tensor(bfloat16), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(float4e2m1), tensor(float8e4m3fn), tensor(float8e4m3fnuz), tensor(float8e5m2), tensor(float8e5m2fnuz), tensor(float8e8m0), tensor(int16), tensor(int32), tensor(int4), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint4), tensor(uint64), tensor(uint8)
  Constrain input and output types to any tensor type.
