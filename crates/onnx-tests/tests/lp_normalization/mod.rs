use crate::include_models;
include_models!(
    lp_normalization_default,
    lp_normalization_l1_axis1,
    lp_normalization_l2_axis0,
    lp_normalization_l2_negative_axis
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    // Input generated via np.random.seed(42), shape [2, 3, 4]. Shared across all
    // tests so different axis/p configurations operate on the same data.
    fn test_input(device: &Device) -> Tensor<3> {
        Tensor::<3>::from_floats(
            [
                [
                    [0.49671414, -0.13826430, 0.64768857, 1.52302980],
                    [-0.23415338, -0.23413695, 1.57921280, 0.76743470],
                    [-0.46947438, 0.54256004, -0.46341768, -0.46572974],
                ],
                [
                    [0.24196227, -1.91328020, -1.72491790, -0.56228750],
                    [-1.01283110, 0.31424734, -0.90802410, -1.41230370],
                    [1.46564880, -0.22577630, 0.06752820, -1.42474820],
                ],
            ],
            device,
        )
    }

    #[test]
    fn lp_normalization_default() {
        let device = Default::default();
        let model: lp_normalization_default::Model = lp_normalization_default::Model::default();

        let output = model.forward(test_input(&device));

        let expected = TensorData::from([
            [
                [0.28654116, -0.07976099, 0.37363428, 0.87859530],
                [-0.13104902, -0.13103984, 0.88384080, 0.42951152],
                [-0.48257616, 0.55770147, -0.47635043, -0.47872700],
            ],
            [
                [0.09138335, -0.72260010, -0.65146020, -0.21236253],
                [-0.51001830, 0.15824148, -0.45724198, -0.71117556],
                [0.71232240, -0.10972992, 0.03281949, -0.69244426],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn lp_normalization_l1_axis1() {
        let device = Default::default();
        let model: lp_normalization_l1_axis1::Model = lp_normalization_l1_axis1::Model::default();

        let output = model.forward(test_input(&device));

        let expected = TensorData::from([
            [
                [0.41381055, -0.15111491, 0.24074787, 0.55258435],
                [-0.19507223, -0.25589820, 0.58699834, 0.27844000],
                [-0.39111720, 0.59298690, -0.17225380, -0.16897567],
            ],
            [
                [0.08894225, -0.77987903, -0.63874720, -0.16541082],
                [-0.37230384, 0.12809148, -0.33624664, -0.41546416],
                [0.53875387, -0.09202949, 0.02500609, -0.41912502],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn lp_normalization_l2_axis0() {
        let device = Default::default();
        let model: lp_normalization_l2_axis0::Model = lp_normalization_l2_axis0::Model::default();

        let output = model.forward(test_input(&device));

        let expected = TensorData::from([
            [
                [0.89900887, -0.07207762, 0.35152520, 0.93810886],
                [-0.22524594, -0.59746800, 0.86691180, 0.47745487],
                [-0.30505085, 0.92325230, -0.98954930, -0.31070673],
            ],
            [
                [0.43793040, -0.99739903, -0.93617845, -0.34634050],
                [-0.97430193, 0.80189276, -0.49846154, -0.87865620],
                [0.95233610, -0.38419430, 0.14419495, -0.95050585],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn lp_normalization_l2_negative_axis() {
        // axis=-2 on rank-3 resolves to axis=1.
        let device = Default::default();
        let model: lp_normalization_l2_negative_axis::Model =
            lp_normalization_l2_negative_axis::Model::default();

        let output = model.forward(test_input(&device));

        let expected = TensorData::from([
            [
                [0.68752480, -0.22782646, 0.36620232, 0.86148960],
                [-0.32410240, -0.38580164, 0.89288497, 0.43409330],
                [-0.64982100, 0.89400900, -0.26201580, -0.26343630],
            ],
            [
                [0.13457936, -0.98015590, -0.88435125, -0.26988563],
                [-0.56333643, 0.16098602, -0.46553650, -0.67787470],
                [0.81519353, -0.11566313, 0.03462116, -0.68384780],
            ],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
