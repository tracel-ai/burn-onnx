#[test]
fn rejects_unsupported_external_cache_lengths() {
    let bytes = include_bytes!("fixtures/attention/unsupported_lengths.onnx");
    let result = onnx_ir::OnnxGraphBuilder::new().parse_bytes(bytes);
    assert!(result.is_err(), "lengths must not be silently ignored");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("nonpad_kv_seqlen is not supported")
    );
}
