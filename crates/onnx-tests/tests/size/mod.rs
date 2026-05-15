// Import the shared macro
use crate::include_models;
include_models!(size);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};


    #[test]
    fn size() {
        let model: size::Model = size::Model::default();
        let device = Default::default();

        let input =
            Tensor::<1>::arange(0..(1 * 2 * 3 * 4 * 5), &device).reshape([1, 2, 3, 4, 5]);
        let output = model.forward(input);
        let expected = TensorData::from([120]);

        output.to_data().assert_eq(&expected, true);
    }
}
