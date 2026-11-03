use crate::include_models;
include_models!(random_normal);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        module::Module,
        tensor::{Device, Shape},
    };

    fn load(device: &Device) -> random_normal::Model {
        random_normal::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/random_normal.bpk"),
            device,
        )
    }

    #[test]
    fn random_normal() {
        let device = Default::default();
        let model = random_normal::Model::new(&device);
        let expected_shape = Shape::from([2, 3]);
        let output = model.forward();
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn allocation_device_is_reported_by_parameterless_model() {
        let device = Device::default();
        let model = load(&device);

        // Module traversal skips a `#[module(skip)]` device, which would leave this empty.
        assert_eq!(model.devices().as_slice(), &[device]);
    }

    #[test]
    fn parameterless_model_trains_without_autodiff_feature() {
        let model = load(&Device::default());

        assert_eq!(model.train().forward().shape(), Shape::from([2, 3]));
    }

    #[cfg(all(feature = "test-flex", feature = "test-autodiff"))]
    #[test]
    fn allocation_device_follows_training_mode() {
        let device = Device::flex().autodiff().gradient_checkpointing();
        let model = load(&device);

        let training = model.clone().train().forward();
        assert!(training.is_autodiff());
        assert_eq!(
            training.gradient_checkpointing_strategy(),
            device.gradient_checkpointing_strategy(),
        );

        let valid = model.valid();
        assert!(!valid.forward().is_autodiff());
        assert!(model.forward().is_autodiff());

        // Moving to an autodiff-enabled destination must not re-enable training.
        for moved in [valid.clone().to_device(&device), valid.fork(&device)] {
            assert!(!moved.forward().is_autodiff());
            assert!(moved.train().forward().is_autodiff());
        }

        let plain = load(&Device::flex());
        assert!(plain.train().forward().is_autodiff());
    }

    #[cfg(all(feature = "test-flex", feature = "test-cpu"))]
    #[test]
    fn tensorless_allocation_follows_module_moves() {
        let source = Device::cpu();
        let target = Device::flex();
        let model = load(&source);

        for moved in [
            model.clone().to_device(&target),
            model.clone().fork(&target),
        ] {
            assert_eq!(moved.forward().device(), target);
            assert_eq!(moved.devices().as_slice(), &[target.clone()]);
            assert_eq!(moved.num_params(), 0);
            assert_eq!(moved.to_device(&source).forward().device(), source);
        }
        assert_eq!(model.forward().device(), source);
    }
}
