#[test]
fn accepts_supported_external_cache_lengths() {
    // #693 rejects this fixture while external caches are unsupported. Once the
    // translation is implemented, keep the same end-to-end parser regression.
    let bytes = include_bytes!("fixtures/attention/unsupported_lengths.onnx");
    onnx_ir::OnnxGraphBuilder::new().parse_bytes(bytes).unwrap();
}
