use super::prelude::*;
use onnx_ir::node::affine_grid::AffineGridSize;
use proc_macro2::Literal;
use quote::format_ident;

impl NodeCodegen for onnx_ir::node::affine_grid::AffineGridNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let theta = scope.arg(self.inputs.first().unwrap());
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);
        // size has 4 values (N, C, H, W) or 5 (N, C, D, H, W), like the output rank
        let len = output_arg.ty.rank();
        let spatial = len - 2;

        // Spatial sizes, outermost first: [height, width] or [depth, height, width]
        let names: Vec<Ident> = ["depth", "height", "width"][3 - spatial..]
            .iter()
            .map(|name| format_ident!("{name}"))
            .collect();

        // The graph values (theta and a runtime size) are read before any temporary is
        // declared, so no temporary can shadow them. The batch and spatial sizes are checked
        // like ONNX Runtime: N non-negative, spatial sizes positive, C unused.
        let spatial_lit = Literal::usize_unsuffixed(spatial);
        let read_inputs = match &self.config.size {
            AffineGridSize::Static(size) => {
                // Checked in onnx-ir
                let batch = Literal::i64_unsuffixed(size[0]);
                let sizes = size[2..].iter().map(|&dim| Literal::i64_unsuffixed(dim));
                quote! {
                    let theta = #theta;
                    let batch: usize = #batch;
                    let [#(#names),*]: [usize; #spatial_lit] = [#(#sizes),*];
                }
            }
            AffineGridSize::Runtime(runtime_ref) => {
                let size_arg = &self.inputs[runtime_ref.input_index];
                let read_size = match &size_arg.ty {
                    ArgType::Shape(_) => {
                        let size = arg_to_ident(size_arg);
                        quote! { let (theta, size) = (#theta, #size); }
                    }
                    _ => {
                        let size = scope.arg(size_arg);
                        quote! {
                            let (theta, size_data) =
                                (#theta, #size.to_data().convert::<i64>());
                            let size = size_data.as_slice::<i64>().unwrap();
                        }
                    }
                };
                quote! {
                    #read_size
                    assert!(size[0] >= 0, "AffineGrid: N must be non-negative, got {}", size[0]);
                    let batch = size[0] as usize;
                    let [#(#names),*]: [usize; #spatial_lit] = core::array::from_fn(|i| {
                        let len = size[i + 2];
                        assert!(len > 0, "AffineGrid: spatial sizes must be positive, got {}", len);
                        len as usize
                    });
                }
            }
        };

        // Evenly spaced positions in [-1, 1]: with align_corners they are the corner pixel
        // centers, otherwise -1 and 1 are the outer edges and the positions the pixel
        // centers. A single position is -1 with align_corners, as in the ONNX reference
        // implementation.
        let spacing = if self.config.align_corners {
            quote! { (if len > 1 { 2.0 / (len - 1) as f64 } else { 0.0 }, -1.0) }
        } else {
            quote! { (2.0 / len as f64, 1.0 / len as f64 - 1.0) }
        };

        // The grid holds (x, y[, z]), innermost axis first: each coordinate varies along
        // its own axis and is broadcast over the others
        let coordinates = (0..spatial).rev().map(|axis| {
            let len = &names[axis];
            let shape = names.iter().enumerate().map(|(other, name)| {
                if other == axis {
                    quote! { #name }
                } else {
                    quote! { 1 }
                }
            });
            quote! { coordinates(#len).reshape([#(#shape),*]).expand([#(#names),*]) }
        });
        // Each point is (x, y[, z], 1): stacking the spatial-rank tensors gives rank
        // spatial + 1, with spatial + 1 values per point
        // Number of grid points: height * width or depth * height * width
        let points = names
            .iter()
            .map(|name| quote! { #name })
            .reduce(|acc, name| quote! { #acc * #name })
            .unwrap();
        let homogeneous = Literal::usize_unsuffixed(spatial + 1);

        quote! {
            let #output = {
                #read_inputs
                assert_eq!(
                    theta.dims()[0], batch,
                    "AffineGrid: theta has batch {} but size has N = {}", theta.dims()[0], batch
                );
                let device = theta.device();
                let dtype = theta.dtype();
                // Positions are computed in f32 (f64 for f64 theta) and cast afterwards: a
                // half-precision index overflows past 65504 and is inexact past 2048
                let working_dtype = if dtype == burn::tensor::DType::F64 {
                    burn::tensor::DType::F64
                } else {
                    burn::tensor::DType::F32
                };
                let coordinates = |len: usize| {
                    let (step, start) = #spacing;
                    Tensor::<1, Int>::arange(0..len as i64, (&device, burn::tensor::DType::I64))
                        .float()
                        .cast(working_dtype)
                        .mul_scalar(step)
                        .add_scalar(start)
                        .cast(dtype)
                };
                let points = #points;
                let stacked: Tensor<#homogeneous> = Tensor::stack(
                    alloc::vec::Vec::from([
                        #(#coordinates,)*
                        Tensor::ones([#(#names),*], (&device, dtype)),
                    ]),
                    #spatial_lit,
                );
                // One 2D matmul for the whole batch, without broadcasting it (an empty batch
                // can't be expanded): points [P, s + 1] @ theta as [s + 1, N * s]
                let base: Tensor<2> = stacked.reshape([points, #homogeneous]);
                let theta = theta
                    .permute([2, 0, 1])
                    .reshape([#homogeneous, batch * #spatial_lit]);
                base.matmul(theta)
                    .reshape([points, batch, #spatial_lit])
                    .permute([1, 0, 2])
                    .reshape([batch, #(#names,)* #spatial_lit])
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::node::affine_grid::{
        AffineGridConfig, AffineGridNode, AffineGridNodeBuilder, AffineGridSize,
    };

    fn runtime_size(input_index: usize) -> AffineGridSize {
        AffineGridSize::Runtime(RuntimeInputRef {
            name: "size".to_string(),
            input_index,
        })
    }

    #[test]
    fn test_affine_grid_2d_static_size() {
        let config = AffineGridConfig {
            align_corners: false,
            size: AffineGridSize::Static(vec![2, 1, 3, 4]),
        };
        let node = AffineGridNodeBuilder::new("affine_grid1")
            .input_tensor("theta", 3, DType::F32)
            .output_tensor("grid", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, theta: Tensor<3>) -> Tensor<4> {
            let grid = {
                let theta = theta;
                let batch: usize = 2;
                let [height, width]: [usize; 2] = [3, 4];
                assert_eq!(
                    theta.dims() [0], batch,
                    "AffineGrid: theta has batch {} but size has N = {}", theta.dims() [0], batch
                );
                let device = theta.device();
                let dtype = theta.dtype();
                let working_dtype = if dtype == burn::tensor::DType::F64 {
                    burn::tensor::DType::F64
                } else {
                    burn::tensor::DType::F32
                };
                let coordinates = |len: usize| {
                    let (step, start) = (2.0 / len as f64, 1.0 / len as f64 - 1.0);
                    Tensor::<1, Int>::arange(0..len as i64, (&device, burn::tensor::DType::I64))
                        .float()
                        .cast(working_dtype)
                        .mul_scalar(step)
                        .add_scalar(start)
                        .cast(dtype)
                };
                let points = height * width;
                let stacked: Tensor<3> = Tensor::stack(
                    alloc::vec::Vec::from([
                        coordinates(width).reshape([1, width]).expand([height, width]),
                        coordinates(height).reshape([height, 1]).expand([height, width]),
                        Tensor::ones([height, width], (&device, dtype)),
                    ]),
                    2,
                );
                let base: Tensor<2> = stacked.reshape([points, 3]);
                let theta = theta.permute([2, 0, 1]).reshape([3, batch * 2]);
                base.matmul(theta)
                    .reshape([points, batch, 2])
                    .permute([1, 0, 2])
                    .reshape([batch, height, width, 2])
            };
            grid
        }
        "#);
    }

    #[test]
    fn test_affine_grid_2d_runtime_size_align_corners() {
        let config = AffineGridConfig {
            align_corners: true,
            size: runtime_size(1),
        };
        let node = AffineGridNodeBuilder::new("affine_grid1")
            .input_tensor("theta", 3, DType::F32)
            .input_tensor("size", 1, DType::I64)
            .output_tensor("grid", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, theta: Tensor<3>, size: Tensor<1, Int>) -> Tensor<4> {
            let grid = {
                let (theta, size_data) = (theta, size.to_data().convert::<i64>());
                let size = size_data.as_slice::<i64>().unwrap();
                assert!(size[0] >= 0, "AffineGrid: N must be non-negative, got {}", size[0]);
                let batch = size[0] as usize;
                let [height, width]: [usize; 2] = core::array::from_fn(|i| {
                    let len = size[i + 2];
                    assert!(len > 0, "AffineGrid: spatial sizes must be positive, got {}", len);
                    len as usize
                });
                assert_eq!(
                    theta.dims() [0], batch,
                    "AffineGrid: theta has batch {} but size has N = {}", theta.dims() [0], batch
                );
                let device = theta.device();
                let dtype = theta.dtype();
                let working_dtype = if dtype == burn::tensor::DType::F64 {
                    burn::tensor::DType::F64
                } else {
                    burn::tensor::DType::F32
                };
                let coordinates = |len: usize| {
                    let (step, start) = (
                        if len > 1 { 2.0 / (len - 1) as f64 } else { 0.0 },
                        -1.0,
                    );
                    Tensor::<1, Int>::arange(0..len as i64, (&device, burn::tensor::DType::I64))
                        .float()
                        .cast(working_dtype)
                        .mul_scalar(step)
                        .add_scalar(start)
                        .cast(dtype)
                };
                let points = height * width;
                let stacked: Tensor<3> = Tensor::stack(
                    alloc::vec::Vec::from([
                        coordinates(width).reshape([1, width]).expand([height, width]),
                        coordinates(height).reshape([height, 1]).expand([height, width]),
                        Tensor::ones([height, width], (&device, dtype)),
                    ]),
                    2,
                );
                let base: Tensor<2> = stacked.reshape([points, 3]);
                let theta = theta.permute([2, 0, 1]).reshape([3, batch * 2]);
                base.matmul(theta)
                    .reshape([points, batch, 2])
                    .permute([1, 0, 2])
                    .reshape([batch, height, width, 2])
            };
            grid
        }
        "#);
    }

    #[test]
    fn test_affine_grid_3d_shape_size() {
        let config = AffineGridConfig {
            align_corners: false,
            size: runtime_size(1),
        };
        let node = AffineGridNodeBuilder::new("affine_grid1")
            .input_tensor("theta", 3, DType::F32)
            .input_shape("size", 5)
            .output_tensor("grid", 5, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, theta: Tensor<3>, size: [i64; 5]) -> Tensor<5> {
            let grid = {
                let (theta, size) = (theta, size);
                assert!(size[0] >= 0, "AffineGrid: N must be non-negative, got {}", size[0]);
                let batch = size[0] as usize;
                let [depth, height, width]: [usize; 3] = core::array::from_fn(|i| {
                    let len = size[i + 2];
                    assert!(len > 0, "AffineGrid: spatial sizes must be positive, got {}", len);
                    len as usize
                });
                assert_eq!(
                    theta.dims() [0], batch,
                    "AffineGrid: theta has batch {} but size has N = {}", theta.dims() [0], batch
                );
                let device = theta.device();
                let dtype = theta.dtype();
                let working_dtype = if dtype == burn::tensor::DType::F64 {
                    burn::tensor::DType::F64
                } else {
                    burn::tensor::DType::F32
                };
                let coordinates = |len: usize| {
                    let (step, start) = (2.0 / len as f64, 1.0 / len as f64 - 1.0);
                    Tensor::<1, Int>::arange(0..len as i64, (&device, burn::tensor::DType::I64))
                        .float()
                        .cast(working_dtype)
                        .mul_scalar(step)
                        .add_scalar(start)
                        .cast(dtype)
                };
                let points = depth * height * width;
                let stacked: Tensor<4> = Tensor::stack(
                    alloc::vec::Vec::from([
                        coordinates(width).reshape([1, 1, width]).expand([depth, height, width]),
                        coordinates(height)
                            .reshape([1, height, 1])
                            .expand([depth, height, width]),
                        coordinates(depth).reshape([depth, 1, 1]).expand([depth, height, width]),
                        Tensor::ones([depth, height, width], (&device, dtype)),
                    ]),
                    3,
                );
                let base: Tensor<2> = stacked.reshape([points, 4]);
                let theta = theta.permute([2, 0, 1]).reshape([4, batch * 3]);
                base.matmul(theta)
                    .reshape([points, batch, 3])
                    .permute([1, 0, 2])
                    .reshape([batch, depth, height, width, 3])
            };
            grid
        }
        "#);
    }

    fn node_with_inputs(theta: &str, size: &str, size_is_shape: bool) -> AffineGridNode {
        let config = AffineGridConfig {
            align_corners: false,
            size: AffineGridSize::Runtime(RuntimeInputRef {
                name: size.to_string(),
                input_index: 1,
            }),
        };
        let builder = AffineGridNodeBuilder::new("affine_grid1").input_tensor(theta, 3, DType::F32);
        let builder = if size_is_shape {
            builder.input_shape(size, 4)
        } else {
            builder.input_tensor(size, 1, DType::I64)
        };
        builder
            .output_tensor("grid", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn inputs_named_like_its_temporaries_are_accepted() {
        // theta and size are read before any temporary is declared
        let temporaries = [
            "theta",
            "size",
            "size_data",
            "batch",
            "height",
            "width",
            "device",
            "dtype",
            "coordinates",
            "points",
            "stacked",
            "base",
        ];
        for temporary in temporaries {
            for size_is_shape in [false, true] {
                let as_theta = node_with_inputs(temporary, "size_values", size_is_shape);
                assert_eq!(
                    shadow_check_result(&as_theta),
                    Ok(()),
                    "theta named {temporary}, shape size: {size_is_shape}"
                );
                if temporary != "theta" {
                    let as_size = node_with_inputs("theta", temporary, size_is_shape);
                    assert_eq!(
                        shadow_check_result(&as_size),
                        Ok(()),
                        "size named {temporary}, shape size: {size_is_shape}"
                    );
                }
            }
        }
    }
}
