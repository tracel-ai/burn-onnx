# RegexFullMatch

First introduced in opset **20**

## Description

RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.

## Attributes

- **pattern** (STRING, optional): Regex pattern to match on. This must be valid RE2 syntax.

## Inputs (1 - 1)

- **X** (T1): Tensor with strings to match on.

## Outputs (1 - 1)

- **Y** (T2): Tensor of bools indicating if each input string fully matches the regex pattern specified.

## Type Constraints

- **T1**: tensor(string)
  Inputs must be UTF-8 strings
- **T2**: tensor(bool)
  Outputs are bools and are True where there is a full regex match and False otherwise.
