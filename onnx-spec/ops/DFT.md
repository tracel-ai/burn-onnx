# DFT

First introduced in opset **17**

All versions: 17, 20

## Description

Computes the discrete Fourier Transform (DFT) of the input.

Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
DFT is computed and `M` denotes the conceptual "all other dimensions,"
the DFT `y[m, k]` of shape `[M, N]` is defined as

$$y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,$$

and the inverse transform is defined as

$$x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,$$

where $j$ is the imaginary unit.

The actual shape of the output is specified in the "output" section.

Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html

## Attributes

- **inverse** (INT, optional): Whether to perform the inverse discrete Fourier Transform. Default is 0, which corresponds to `false`.
- **onesided** (INT, optional): If `onesided` is `1` and input is real, only values for `k` in `[0, 1, 2, ..., floor(n_fft/2) + 1]` are returned because the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., `X[m, k] = X[m, n_fft-k]*`, where `m` denotes "all other dimensions" DFT was not applied on. If the input tensor is complex, onesided output is not possible. Value can be `0` or `1`. Default is `0`.

## Inputs (1 - 3)

- **input** (T1): For real input, the following shape is expected: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][1]`. For complex input, the following shape is expected: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]`. The final dimension represents the real and imaginary parts of the value in that order.
- **dft_length** (T2, optional): The length of the signal as a scalar. If greater than the axis dimension, the signal will be zero-padded up to `dft_length`. If less than the axis dimension, only the first `dft_length` values will be used as the signal.
- **axis** (tensor(int64), optional): The axis as a scalar on which to perform the DFT. Default is `-2` (last signal axis). Negative value means counting dimensions from the back. Accepted range is $[-r, -2] \cup [0, r-2]$ where `r = rank(input)`. The last dimension is for representing complex numbers and thus is an invalid axis.

## Outputs (1 - 1)

- **output** (T1): The Fourier Transform of the input vector. If `onesided` is `0`, the following shape is expected: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]`. If `axis=0` and `onesided` is `1`, the following shape is expected: `[floor(signal_dim0/2)+1][signal_dim1][signal_dim2]...[signal_dimN][2]`. If `axis=1` and `onesided` is `1`, the following shape is expected: `[signal_dim0][floor(signal_dim1/2)+1][signal_dim2]...[signal_dimN][2]`. If `axis=N` and `onesided` is `1`, the following shape is expected: `[signal_dim0][signal_dim1][signal_dim2]...[floor(signal_dimN/2)+1][2]`. The `signal_dim` at the specified `axis` is equal to the `dft_length`.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain input and output types to float tensors.
- **T2**: tensor(int32), tensor(int64)
  Constrain scalar length types to integers.

## Version History

- **Opset 20**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
- **Opset 17**: Types: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
