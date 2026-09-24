// Import the shared macro
use crate::include_models;
include_models!(
    attention_4d,
    attention_3d,
    attention_attn_mask_bool,
    attention_attn_mask_int,
    attention_attn_mask_float,
    attention_softcap,
    attention_cache,
    attention_custom_scale,
    attention_is_causal,
    attention_qk_output_0,
    attention_qk_output_1,
    attention_qk_output_2,
    attention_qk_output_3,
    attention_gqa_causal,
    attention_kv_cache_causal,
    attention_padding_mask_causal,
    attention_softcap_bias
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Int, Tensor, TensorData, Tolerance};

    #[test]
    fn simple_4d() {
        let device = Default::default();
        let model: attention_4d::Model = attention_4d::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.283488f32, 0.566976], [0.266511, 0.533023]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn simple_3d() {
        let device = Default::default();
        let model: attention_3d::Model = attention_3d::Model::new(&device);

        let q = Tensor::<3>::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
        let k = Tensor::<3>::from_floats([[[0.0, 1.0], [1.0, 0.0]]], &device);
        let v = Tensor::<3>::from_floats([[[0.25, 0.5], [0.3, 0.6]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[0.283488f32, 0.566976], [0.266511, 0.533023]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_bool() {
        let device = Default::default();
        let model: attention_attn_mask_bool::Model = attention_attn_mask_bool::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask =
            Tensor::<2, Bool>::from_bool(TensorData::from([[true, false], [false, true]]), &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.25f32, 0.5], [0.3, 0.6]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_int() {
        let device = Default::default();
        let model: attention_attn_mask_int::Model = attention_attn_mask_int::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<2, Int>::from_ints([[2, 0], [0, 3]], &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.260768f32, 0.521536], [0.295414, 0.590828]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_float() {
        let device = Default::default();
        let model: attention_attn_mask_float::Model =
            attention_attn_mask_float::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<2>::from_floats([[2.0, 0.0], [0.0, 3.0]], &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.260768f32, 0.521536], [0.295414, 0.590828]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn softcap() {
        let device = Default::default();
        let model: attention_softcap::Model = attention_softcap::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.283176f32, 0.566352], [0.266823, 0.533647]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[allow(clippy::type_complexity)]
    fn cached_attn_inputs() -> (
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<2, Bool>,
        Tensor<4>,
        Tensor<4>,
    ) {
        let device = &Default::default();
        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], device);
        let k = Tensor::<4>::from_floats([[[[1.0, 0.0]]]], device);
        let v = Tensor::<4>::from_floats([[[[0.3, 0.6]]]], device);
        let attn_mask =
            Tensor::<2, Bool>::from_bool(TensorData::from([[true, true], [true, true]]), device);
        let past_k = Tensor::<4>::from_floats([[[[0.0, 1.0]]]], device);
        let past_v = Tensor::<4>::from_floats([[[[0.25, 0.5]]]], device);

        (q, k, v, attn_mask, past_k, past_v)
    }

    #[test]
    fn cache() {
        let device = Default::default();
        let model: attention_cache::Model = attention_cache::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (output, present_k, present_v) = model.forward(q, k, v, attn_mask, past_k, past_v);
        let expected = TensorData::from([[[[0.283488f32, 0.566976], [0.266511, 0.533023]]]]);
        let expected_k = TensorData::from([[[[0.0, 1.0], [1.0, 0.0]]]]);
        let expected_v = TensorData::from([[[[0.25, 0.5], [0.3, 0.6]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
        present_k
            .to_data()
            .assert_approx_eq::<f32>(&expected_k, Tolerance::default());
        present_v
            .to_data()
            .assert_approx_eq::<f32>(&expected_v, Tolerance::default());
    }

    #[test]
    fn custom_scale() {
        let device = Default::default();
        let model: attention_custom_scale::Model = attention_custom_scale::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.294039f32, 0.588079], [0.255960, 0.511920]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn is_causal() {
        let device = Default::default();
        let model: attention_is_causal::Model = attention_is_causal::Model::new(&device);

        let q = Tensor::<4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.25f32, 0.5], [0.266511, 0.533023]]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_0() {
        let device = Default::default();
        let model: attention_qk_output_0::Model = attention_qk_output_0::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.707106], [0.707106, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_1() {
        let device = Default::default();
        let model: attention_qk_output_1::Model = attention_qk_output_1::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        // Mode 1 is the product after the softcap, before the mask.
        let expected_qk = TensorData::from([[[[0.0f32, 0.67904], [0.67904, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_2() {
        let device = Default::default();
        let model: attention_qk_output_2::Model = attention_qk_output_2::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.67904], [0.67904, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_3() {
        let device = Default::default();
        let model: attention_qk_output_3::Model = attention_qk_output_3::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.336474f32, 0.663525], [0.663525, 0.336474]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<f32>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn attention_gqa_causal() {
        let device = Default::default();
        let model = attention_gqa_causal::Model::new(&device);
        let seq = |shape: [usize; 4], scale: f32| {
            let n = shape.iter().product::<usize>() as i64;
            Tensor::<1, burn::tensor::Int>::arange(0..n, &device)
                .float()
                .reshape(shape)
                .mul_scalar(scale)
                .remainder_scalar(1.7)
                .sub_scalar(0.8)
        };
        let mask = Tensor::<2>::from_floats([[0.0, -0.5, 0.3], [0.2, 0.0, -1.0]], &device);

        let y = model.forward(
            seq([1, 4, 2, 4], 0.37),
            seq([1, 2, 3, 4], 0.23),
            seq([1, 2, 3, 4], 0.41),
            mask,
        );

        y.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [
                    [-0.8f32, -0.39, 0.02, 0.43],
                    [-0.091_481, -0.415_921, -0.005_921, 0.404_078],
                ],
                [
                    [-0.8, -0.39, 0.02, 0.43],
                    [-0.102_567, -0.415_516, -0.005_516, 0.404_484],
                ],
                [
                    [0.72, -0.57, -0.16, 0.25],
                    [0.700_911, -0.589_089, -0.179_089, 0.230_911],
                ],
                [
                    [0.72, -0.57, -0.16, 0.25],
                    [0.685_064, -0.604_937, -0.194_936, 0.215_064],
                ],
            ]]),
            burn::tensor::Tolerance::absolute(1e-4),
        );
    }

    fn seq<const D: usize>(shape: [usize; D], scale: f32) -> Tensor<D> {
        let device = Default::default();
        let n = shape.iter().product::<usize>() as i64;
        Tensor::<1, Int>::arange(0..n, &device)
            .float()
            .reshape(shape)
            .mul_scalar(scale)
            .remainder_scalar(1.7)
            .sub_scalar(0.8)
    }

    #[test]
    fn attention_kv_cache_causal() {
        // A decode step: the causal mask is offset by the two cached keys, so the
        // single new query sees all three.
        let device = Default::default();
        let model = attention_kv_cache_causal::Model::new(&device);

        let (y, present_k, _) = model.forward(
            seq([1, 2, 1, 4], 0.37),
            seq([1, 2, 1, 4], 0.23),
            seq([1, 2, 1, 4], 0.41),
            seq([1, 2, 2, 4], 0.29),
            seq([1, 2, 2, 4], 0.53),
        );

        assert_eq!(present_k.dims(), [1, 2, 3, 4]);
        y.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [[-0.72138f32, -0.23987, 0.241639, 0.404923]],
                [[0.447403, -0.204451, -0.27166, 0.218809]],
            ]]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn attention_padding_mask_causal() {
        // A [batch, 1, 1, keys] padding mask broadcast against a non-square causal
        // mask. The second batch hides every key, so its rows are zeros.
        let device = Default::default();
        let model = attention_padding_mask_causal::Model::new(&device);
        let mask = Tensor::<4, Bool>::from_data(
            TensorData::from([[[[true, true, false]]], [[[false, false, false]]]]),
            &device,
        );

        let y = model.forward(
            seq([2, 2, 2, 4], 0.37),
            seq([2, 2, 3, 4], 0.23),
            seq([2, 2, 3, 4], 0.41),
            mask,
        );

        y.to_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [
                    [
                        [-0.8f32, -0.39, 0.02, 0.43],
                        [-0.010_162, -0.418_897, -0.008_897, 0.401_103],
                    ],
                    [
                        [0.72, -0.57, -0.16, 0.25],
                        [0.693_164, -0.596_837, -0.186_837, 0.223_164],
                    ],
                ],
                [[[0.0; 4], [0.0; 4]], [[0.0; 4], [0.0; 4]]],
            ]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn attention_softcap_bias() {
        // The softcap is applied before the additive mask.
        let device = Default::default();
        let model = attention_softcap_bias::Model::new(&device);
        let mask = Tensor::<2>::from_floats([[0.0, 3.0, -2.0], [2.5, 0.0, 1.0]], &device);

        let y = model.forward(
            seq([1, 1, 2, 4], 1.37),
            seq([1, 1, 3, 4], 1.23),
            seq([1, 1, 3, 4], 0.41),
            mask,
        );

        y.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[[
                [0.699_025f32, -0.445_199, -0.035_199, 0.374_801],
                [-0.461_619, -0.412_644, -0.002_644, 0.407_356],
            ]]]),
            Tolerance::absolute(1e-4),
        );
    }
}
