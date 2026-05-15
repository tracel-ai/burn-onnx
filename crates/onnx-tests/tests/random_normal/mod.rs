use crate::include_models;
include_models!(random_normal);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Shape;

    #[test]
    fn random_normal() {
        let device = Default::default();
        let model = random_normal::Model::new(&device);
        let expected_shape = Shape::from([2, 3]);
        let output = model.forward();
        assert_eq!(expected_shape, output.shape());
    }
}
