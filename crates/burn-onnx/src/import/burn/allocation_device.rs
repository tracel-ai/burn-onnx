use proc_macro2::TokenStream;
use quote::quote;

/// Generate non-persistent allocation state that participates in module movement.
pub(super) fn codegen() -> TokenStream {
    quote! {
        /// Runtime allocation device, moved alongside the model's parameters.
        #[derive(Clone, Debug)]
        struct AllocationDevice(Device);

        impl core::ops::Deref for AllocationDevice {
            type Target = Device;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl burn::module::Module for AllocationDevice {
            fn collect_devices(&self, mut devices: burn::module::Devices) -> burn::module::Devices {
                if !devices.contains(&self.0) {
                    devices.push(self.0.clone());
                }
                devices
            }

            fn to_device(self, device: &Device) -> Self {
                Self(device.clone().with_autodiff_context_from(&self.0))
            }

            fn fork(self, device: &Device) -> Self {
                self.to_device(device)
            }

            // Allocation state has no parameters to visit, persist, or train.
            fn visit<V: burn::module::ModuleVisitor>(&self, _visitor: &mut V) {}

            fn map<M: burn::module::ModuleMapper>(self, _mapper: &mut M) -> Self {
                self
            }

            fn materialize(self) -> Self {
                self
            }

            fn valid(&self) -> Self {
                self.clone()
            }

            fn train(self) -> Self {
                self
            }
        }

        impl burn::module::ModuleDisplayDefault for AllocationDevice {
            fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
                content.add_debug_attribute("device", &self.0).optional()
            }
        }

        impl burn::module::ModuleDisplay for AllocationDevice {}
    }
}
