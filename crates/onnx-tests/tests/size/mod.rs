// Import the shared macro
use crate::include_models;
include_models!(size, size_shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[test]
    fn size() {
        let device = Default::default();
        let model: size::Model =
            size::Model::from_file(concat!(env!("OUT_DIR"), "/model/size.bpk"), &device);

        let input = Tensor::<4>::ones([2, 6, 2, 3], &device);
        let output = model.forward(input);

        // 2 * 6 * 2 * 3
        assert_eq!(output, 72);
    }

    #[test]
    fn size_shape() {
        let device = Default::default();
        let model: size_shape::Model = size_shape::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/size_shape.bpk"),
            &device,
        );

        let input = Tensor::<4>::ones([2, 6, 2, 3], &device);
        let output = model.forward(input);

        // Size of a Shape is the rank of the tensor it came from
        assert_eq!(output, 4);
    }
}
