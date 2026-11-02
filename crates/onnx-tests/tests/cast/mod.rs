// Import the shared macro
use crate::include_models;
include_models!(
    cast,
    cast_shape,
    cast_shape_to_float,
    cast_shape_to_bool,
    cast_partitioned_device
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn cast() {
        let device = Default::default();
        let model: cast::Model = cast::Model::new(&device);

        let input_bool = Tensor::<2, Bool>::from_bool(TensorData::from([[true], [true]]), &device);
        let input_int = Tensor::<2, Int>::from_ints([[1], [1]], &device);
        let input_float = Tensor::<2>::from_floats([[1f32], [1.]], &device);
        let input_scalar = 1f32;

        let (
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output_scalar,
        ) = model.forward(
            input_bool.clone(),
            input_int.clone(),
            input_float.clone(),
            input_scalar,
        );
        let expected_bool = input_bool.to_data();
        let expected_int = input_int.to_data();
        // The ONNX model casts to INT32 (TensorProto value 6), so cross-kind casts
        // produce I32 tensors via .int().cast(I32), not the backend's default IntElem.
        let expected_int_i32 = TensorData::from([[1i32], [1i32]]);
        let expected_float = input_float.to_data();
        let expected_scalar = 1;

        output1.to_data().assert_eq(&expected_bool, true);
        output2.to_data().assert_eq(&expected_int_i32, true); // Bool -> I32
        output3
            .to_data()
            .assert_approx_eq::<f32>(&expected_float, Tolerance::default());

        output4.to_data().assert_eq(&expected_bool, true);
        output5.to_data().assert_eq(&expected_int, true); // Int -> Int (noop)
        output6
            .to_data()
            .assert_approx_eq::<f32>(&expected_float, Tolerance::default());

        output7.to_data().assert_eq(&expected_bool, true);
        output8.to_data().assert_eq(&expected_int_i32, true); // Float -> I32
        output9
            .to_data()
            .assert_approx_eq::<f32>(&expected_float, Tolerance::default());

        assert_eq!(output_scalar, expected_scalar);
    }

    #[test]
    fn cast_shape() {
        // This test verifies that Shape types maintain their i64 representation
        // even when cast to other integer types in ONNX

        let device = Default::default();
        let model: cast_shape::Model = cast_shape::Model::new(&device);

        // Create test input
        let input = Tensor::<3>::ones([2, 3, 4], &device);

        // Run the model
        let (shape_original, shape_casted) = model.forward(input);

        // Verify both shapes are the same [i64; 3] arrays
        // In Burn, Shape types always use i64 regardless of ONNX cast operations
        assert_eq!(shape_original, [2i64, 3i64, 4i64]);
        assert_eq!(shape_casted, [2i64, 3i64, 4i64]); // Cast to int32 is a no-op for Shape types
    }

    #[test]
    fn cast_shape_to_float() {
        // This test verifies that Shape types can be cast to float tensors
        // When casting Shape to float, it should convert to a 1D tensor

        let device = Default::default();
        let model: cast_shape_to_float::Model = cast_shape_to_float::Model::new(&device);

        // Create test input with shape [2, 3, 4]
        let input = Tensor::<3>::ones([2, 3, 4], &device);
        let multiplier = Tensor::<1>::from_floats([2.0], &device);

        // Run the model - it extracts shape, casts to float, and multiplies by 2
        let output = model.forward(input, multiplier);

        // The output should be [2.0, 3.0, 4.0] * 2.0 = [4.0, 6.0, 8.0]
        let expected = Tensor::<1>::from_floats([4.0, 6.0, 8.0], &device);

        // Check values are close
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[cfg(all(feature = "test-flex", feature = "test-cpu"))]
    #[test]
    fn shape_allocation_follows_module_to_device() {
        use burn::module::Module;

        let device = Device::flex();
        let model = cast_shape_to_float::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/cast_shape_to_float.bpk"),
            &Device::cpu(),
        )
        .to_device(&device);

        let output = model.forward(
            // Only the shape is read: its device must not control allocation.
            Tensor::<3>::ones([2, 3, 4], &Device::cpu()),
            Tensor::<1>::from_floats([2.0], &device),
        );

        assert_eq!(output.device(), device);
        output
            .to_data()
            .assert_eq(&TensorData::from([4.0f32, 6.0, 8.0]), true);
    }

    #[cfg(all(feature = "test-flex", feature = "test-cpu"))]
    #[test]
    fn shape_allocation_follows_nested_partitioned_module_fork() {
        use burn::module::Module;

        #[derive(Module, Debug)]
        struct Nested {
            model: cast_partitioned_device::Model,
            #[module(skip)]
            construction_device: Device,
        }

        let device = Device::flex();
        let nested = Nested {
            model: cast_partitioned_device::Model::from_file(
                concat!(env!("OUT_DIR"), "/model/cast_partitioned_device.bpk"),
                &Device::cpu(),
            ),
            construction_device: Device::cpu(),
        }
        .fork(&device);

        assert_eq!(nested.construction_device, Device::cpu());
        assert_eq!(nested.devices().as_slice(), &[device.clone()]);
        assert_eq!(nested.num_params(), 2);
        // Only metadata flows through the first partitions.
        let output = nested
            .model
            .forward(Tensor::<2>::ones([2, 4], &Device::cpu()));
        assert_eq!(output.device(), device);
        output.to_data().assert_approx_eq::<f32>(
            &TensorData::from([2.0 * core::f32::consts::SQRT_2, 6.0]),
            Tolerance::default(),
        );
    }

    #[test]
    fn cast_shape_to_bool() {
        // This test verifies that Shape types can be cast to bool tensors
        // When casting Shape to bool, non-zero values become true

        let device = Default::default();
        let model: cast_shape_to_bool::Model = cast_shape_to_bool::Model::new(&device);

        // Create test input with shape [2, 3, 4] (all non-zero dimensions)
        let input = Tensor::<3>::ones([2, 3, 4], &device);

        // Run the model - it extracts shape and casts to bool
        let output = model.forward(input);

        // The output should be [true, true, true] since all dimensions are non-zero
        let expected = Tensor::<1, Bool>::from_bool(TensorData::from([true, true, true]), &device);

        // Check values match
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
