pub mod label;
pub mod normalizer;
pub mod squeezenet;

pub fn weights() -> alloc::vec::Vec<u8> {
    include_bytes!(concat!(env!("OUT_DIR"), "/model/squeezenet1_opset16.bpk")).to_vec()
}
