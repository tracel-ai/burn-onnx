use super::prelude::*;

/// Emit a float draw with an explicit dtype. Seed bits preserve fractional and signed
/// ONNX f32 seeds; widening to u64 is lossless. Repeated seeded calls reuse the key,
/// without consuming or resetting the device's implicit random stream.
pub(super) fn random_draw(
    seed: Option<f32>,
    shape: TokenStream,
    distribution: TokenStream,
    device: TokenStream,
    dtype: TokenStream,
) -> TokenStream {
    match seed {
        Some(seed) => {
            let key = u64::from(seed.to_bits());
            quote! {
                Tensor::from_dispatch(
                    <burn::backend::Dispatch as burn::backend::ops::FloatTensorOps<burn::backend::Dispatch>>::float_random(
                        (#shape).into(), #distribution, (#device).as_dispatch(),
                        (#dtype).into(), burn::backend::RandomKey::new(#key),
                    )
                )
            }
        }
        None => quote! { Tensor::random(#shape, #distribution, (#device, #dtype)) },
    }
}
