// The backend is now selected at runtime via Device::default(), driven by the
// `test-*` features (test-flex, test-wgpu, test-metal, test-tch, test-candle)
// which enable the corresponding burn backend feature. There is no per-backend
// compile-time type alias anymore; tests just use `Device` and `Tensor<N>`.
