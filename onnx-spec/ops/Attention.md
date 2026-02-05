# Attention

First introduced in opset **23**

All versions: 23, 24

## Description

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
2) If `is_causal` is set to `1`, attention scores above the diagonal are masked out, regardless of the `attn_mask` input.

With respect to KV cache update, this operator allows the following two use cases:

1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
key and value, respectively. Note that this only works correctly for the special case where the past key and value
do not contain padded tokens.
2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
as the maximum value of `nonpad_kv_seqlen`.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
 at_mask---Add              |
            |               |
  softcap (if provided)     |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```

## Attributes

- **is_causal** (INT, optional): If set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. The attention masking has the form of the upper left causal bias due to the alignment.
- **kv_num_heads** (INT, optional): Number of heads of key and value. Must be used with 3D inputs of Q, K and V.
- **q_num_heads** (INT, optional): Number of heads of query. Must be used with 3D inputs of Q, K and V.
- **qk_matmul_output_mode** (INT, optional): If set to `0`, qk_matmul_output is the output of qk matmul. If set to `1`, qk_matmul_output includes the addition of the attention mask to the output of qk matmul. If set to `2`, qk_matmul_output is the output after the softcap operation. If set to `3`, qk_matmul_output is the output after the softmax operation. Default value is 0.
- **scale** (FLOAT, optional): Scaling factor applied to $Q*K^T$. Default value is `1/sqrt(head_size)`. To prevent [numerical overflow](https://tinyurl.com/sudb9s96), scale `Q`, `K` by `sqrt(scale)` before matmul.
- **softcap** (FLOAT, optional): Softcap value for attention weights. Default value is 0.
- **softmax_precision** (INT, optional): The floating-point precision used in softmax computation. If softmax precision is not provided, the same precision as the input of softmax (Q and K) is used.

## Inputs (3 - 7)

- **Q** (T1): Query tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, q_hidden_size)`. For cases with a 3D input tensor, `q_hidden_size = q_num_heads * head_size`
- **K** (T1): Key tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, k_hidden_size)`. For cases with a 3D input tensor, `k_hidden_size = kv_num_heads * head_size`
- **V** (T2): Value tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, v_hidden_size)`. For cases with a 3D input tensor, `v_hidden_size = kv_num_heads * v_head_size`
- **attn_mask** (U, optional): Attention mask. Shape must be broadcastable to `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where `total_sequence_length = past_sequence_length + kv_sequence_length.` The last dimension can also be shorter than `total_sequence_length` and will be padded to `total_sequence_length` with negative infinity. Two types of masks are supported: a boolean mask where a value of `True` indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score.
- **past_key** (T1, optional): past state cache for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`
- **past_value** (T2, optional): past state cache for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`
- **nonpad_kv_seqlen** (tensor(int64), optional): A vector of integers of shape `(batch_size,)` that indicates the number of valid (ie, non-padding) tokens in each sample. A padding mask can be derived from this. This should not be used together with `past_key` and `past_value` inputs or `present_key` and `present_value` outputs (See the KV cache use cases in the operator description).

## Outputs (1 - 4)

- **Y** (T1): The output tensor . 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`
- **present_key** (T1, optional): Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, head_size)` where `total_sequence_length = past_sequence_length + kv_sequence_length`.
- **present_value** (T2, optional): Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, v_head_size)` where `total_sequence_length = past_sequence_length + kv_sequence_length`.
- **qk_matmul_output** (T1, optional): The output of QK matmul. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where `total_sequence_length = past_sequence_length + kv_sequence_length`.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain Q and K inputs types to float tensors.
- **T2**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain V input types to float tensors.
- **U**: tensor(bfloat16), tensor(bool), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain output 'mask' types to boolean tensors and input types.

## Version History

- **Opset 24**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 23**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
