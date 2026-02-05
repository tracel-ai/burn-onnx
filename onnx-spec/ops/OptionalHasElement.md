# OptionalHasElement

First introduced in opset **15**

All versions: 15, 18

## Description

Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.

## Inputs (0 - 1)

- **input** (O, optional): The optional input.

## Outputs (1 - 1)

- **output** (B): A scalar boolean tensor. If true, it indicates that optional-type input contains an element. Otherwise, it is empty.

## Type Constraints

- **O**: optional(seq(tensor(bool))), optional(seq(tensor(complex128))), optional(seq(tensor(complex64))), optional(seq(tensor(double))), optional(seq(tensor(float))), optional(seq(tensor(float16))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(int8))), optional(seq(tensor(string))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(uint8))), optional(tensor(bool)), optional(tensor(complex128)), optional(tensor(complex64)), optional(tensor(double)), optional(tensor(float)), optional(tensor(float16)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(int8)), optional(tensor(string)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(uint8)), seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8)), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
  Constrain input type to optional tensor and optional sequence types.
- **B**: tensor(bool)
  Constrain output to a boolean tensor.

## Version History

- **Opset 18**: Types: optional(seq(tensor(bool))), optional(seq(tensor(complex128))), optional(seq(tensor(complex64))), optional(seq(tensor(double))), optional(seq(tensor(float))), optional(seq(tensor(float16))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(int8))), optional(seq(tensor(string))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(uint8))), optional(tensor(bool)), optional(tensor(complex128)), optional(tensor(complex64)), optional(tensor(double)), optional(tensor(float)), optional(tensor(float16)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(int8)), optional(tensor(string)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(uint8)), seq(tensor(bool)), seq(tensor(complex128)), seq(tensor(complex64)), seq(tensor(double)), seq(tensor(float)), seq(tensor(float16)), seq(tensor(int16)), seq(tensor(int32)), seq(tensor(int64)), seq(tensor(int8)), seq(tensor(string)), seq(tensor(uint16)), seq(tensor(uint32)), seq(tensor(uint64)), seq(tensor(uint8)), tensor(bool), tensor(complex128), tensor(complex64), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(string), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
- **Opset 15**: Types: optional(seq(tensor(bool))), optional(seq(tensor(complex128))), optional(seq(tensor(complex64))), optional(seq(tensor(double))), optional(seq(tensor(float))), optional(seq(tensor(float16))), optional(seq(tensor(int16))), optional(seq(tensor(int32))), optional(seq(tensor(int64))), optional(seq(tensor(int8))), optional(seq(tensor(string))), optional(seq(tensor(uint16))), optional(seq(tensor(uint32))), optional(seq(tensor(uint64))), optional(seq(tensor(uint8))), optional(tensor(bool)), optional(tensor(complex128)), optional(tensor(complex64)), optional(tensor(double)), optional(tensor(float)), optional(tensor(float16)), optional(tensor(int16)), optional(tensor(int32)), optional(tensor(int64)), optional(tensor(int8)), optional(tensor(string)), optional(tensor(uint16)), optional(tensor(uint32)), optional(tensor(uint64)), optional(tensor(uint8))
