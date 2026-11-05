//! Structural checks shared by linear quantization and dequantization.
use crate::ir::{ArgType, RawNode};
use crate::processor::ProcessError;

pub(super) fn validate(
    node: &RawNode,
    axis: Option<i64>,
    block: Option<i64>,
    opset: usize,
) -> Result<(), ProcessError> {
    let block = block.unwrap_or(0);
    if block < 0 || (block > 0 && opset < 21) {
        return Err(ProcessError::Custom(
            "block_size must be nonnegative and blocked quantization requires opset 21".into(),
        ));
    }
    if let Some(zero) = node.get_input(2)
        && let (ArgType::Tensor(scale), ArgType::Tensor(zero)) = (&node.inputs[1].ty, &zero.ty)
    {
        if scale.rank != zero.rank {
            return Err(ProcessError::Custom(
                "Scale and zero point must have the same shape".into(),
            ));
        }
        if let (Some(a), Some(b)) = (&scale.static_shape, &zero.static_shape)
            && a.iter()
                .zip(b)
                .any(|(a, b)| matches!((a,b),(Some(a),Some(b)) if a!=b))
        {
            return Err(ProcessError::Custom(
                "Scale and zero point must have the same shape".into(),
            ));
        }
    }
    if block == 0 {
        return Ok(());
    }
    let (ArgType::Tensor(x), ArgType::Tensor(scale)) = (&node.inputs[0].ty, &node.inputs[1].ty)
    else {
        return Err(ProcessError::Custom(
            "Blocked quantization requires tensor inputs".into(),
        ));
    };
    let axis = axis.unwrap_or(1);
    if x.rank == 0 || x.rank != scale.rank || axis < -(x.rank as i64) || axis >= x.rank as i64 {
        return Err(ProcessError::Custom(
            "Blocked quantization requires matching input/scale ranks and a valid axis".into(),
        ));
    }
    let axis = axis.rem_euclid(x.rank as i64) as usize;
    if let (Some(xs), Some(ss)) = (&x.static_shape, &scale.static_shape) {
        for (i, (x, s)) in xs.iter().zip(ss).enumerate() {
            if let (Some(x), Some(s)) = (x, s) {
                let expected = if i == axis {
                    x.div_ceil(block as usize)
                } else {
                    *x
                };
                if *s != expected {
                    return Err(ProcessError::Custom(
                        "Blocked quantization scale shape does not match the input and block_size"
                            .into(),
                    ));
                }
            }
        }
    }
    Ok(())
}
