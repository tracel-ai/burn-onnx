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

    #[cfg(all(feature = "test-flex", feature = "test-autodiff"))]
    #[test]
    fn allocation_device_follows_training_mode() {
        use burn::{module::Module, tensor::Device};

        let device = Device::flex().autodiff().gradient_checkpointing();
        let model = random_normal::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/random_normal.bpk"),
            &device,
        );

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
            let training = moved.train().forward();
            assert!(training.is_autodiff());
            // Validation discards checkpointing; train uses the default strategy.
            assert_eq!(
                training.gradient_checkpointing_strategy(),
                Device::flex().autodiff().gradient_checkpointing_strategy(),
            );
        }

        let plain = random_normal::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/random_normal.bpk"),
            &Device::flex(),
        );
        assert!(plain.train().forward().is_autodiff());
    }

    #[cfg(all(feature = "test-flex", feature = "test-cpu"))]
    #[test]
    fn tensorless_allocation_follows_module_moves() {
        use burn::{module::Module, tensor::Device};

        let source = Device::cpu();
        let target = Device::flex();
        let model = random_normal::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/random_normal.bpk"),
            &source,
        );

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
