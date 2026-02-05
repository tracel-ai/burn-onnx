# STFT

First introduced in opset **17**

## Description

Computes the Short-time Fourier Transform of the signal.

## Attributes

- **onesided** (INT, optional): If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are returned because the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]*. Note if the input or window tensors are complex, then onesided output is not possible. Enabling onesided with real inputs performs a Real-valued fast Fourier transform (RFFT).When invoked with real or complex valued input, the default value is 1. Values can be 0 or 1.

## Inputs (2 - 4)

- **signal** (T1): Input tensor representing a real or complex valued signal. For real input, the following shape is expected: [batch_size][signal_length][1]. For complex input, the following shape is expected: [batch_size][signal_length][2], where [batch_size][signal_length][0] represents the real component and [batch_size][signal_length][1] represents the imaginary component of the signal.
- **frame_step** (T2): The number of samples to step between successive DFTs.
- **window** (T1, optional): A tensor representing the window that will be slid over the signal.The window must have rank 1 with shape: [window_shape]. It's an optional value.
- **frame_length** (T2, optional): A scalar representing the size of the DFT. It's an optional value.

## Outputs (1 - 1)

- **output** (T1): The Short-time Fourier Transform of the signals.If onesided is 1, the output has the shape: [batch_size][frames][dft_unique_bins][2], where dft_unique_bins is frame_length // 2 + 1 (the unique components of the DFT) If onesided is 0, the output has the shape: [batch_size][frames][frame_length][2], where frame_length is the length of the DFT.

## Type Constraints

- **T1**: tensor(bfloat16), tensor(double), tensor(float), tensor(float16)
  Constrain signal and output to float tensors.
- **T2**: tensor(int32), tensor(int64)
  Constrain scalar length types to int64_t.
