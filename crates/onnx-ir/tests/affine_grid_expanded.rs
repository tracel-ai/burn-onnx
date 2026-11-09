//! Regression coverage for the shape-dependent branches in ONNX's AffineGrid function.
use onnx_ir::OnnxGraphBuilder;

#[test]
fn expanded_affine_grid() {
    for dimension in ["2d", "3d"] {
        for alignment in ["", "align_corners_"] {
            let path = format!(
                "{}/../onnx-official-tests/vendor/node/test_affine_grid_{dimension}_{alignment}expanded/model.onnx",
                env!("CARGO_MANIFEST_DIR")
            );
            let graph = OnnxGraphBuilder::new()
                .shape_input("size")
                .parse_file(&path)
                .unwrap();
            assert_eq!(
                graph.outputs[0].ty.rank(),
                if dimension == "2d" { 4 } else { 5 }
            );
        }
    }
}

#[test]
fn shape_input_rejects_invalid_contracts() {
    use onnx_ir::ModelProto;
    use protobuf::Message;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-official-tests/vendor/node/test_affine_grid_2d_expanded/model.onnx"
    );
    let bytes = std::fs::read(path).unwrap();
    for name in ["missing", "theta"] {
        let error = OnnxGraphBuilder::new()
            .shape_input(name)
            .parse_bytes(&bytes)
            .unwrap_err();
        assert!(error.to_string().contains("Shape input"), "{error}");
    }
    let mut model = ModelProto::parse_from_bytes(&bytes).unwrap();
    let input = model.graph.as_mut().unwrap().input.get_mut(1).unwrap();
    let tensor = input.type_.as_mut().unwrap().mut_tensor_type();
    tensor.shape.as_mut().unwrap().dim[0].set_dim_param("length".into());
    let error = OnnxGraphBuilder::new()
        .shape_input("size")
        .parse_bytes(&model.write_to_bytes().unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("known length"), "{error}");
}
