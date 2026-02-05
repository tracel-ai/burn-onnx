# ImageDecoder

First introduced in opset **20**

## Description

Loads and decodes and image from a file. If it can't decode for any reason (e.g. corrupted encoded
stream, invalid format, it will return an empty matrix).
The following image formats are supported:
* BMP
* JPEG (note: Lossless JPEG support is optional)
* JPEG2000
* TIFF
* PNG
* WebP
* Portable image format (PBM, PGM, PPM, PXM, PNM)
Decoded images follow a channel-last layout: (Height, Width, Channels).
**JPEG chroma upsampling method:**
When upsampling the chroma components by a factor of 2, the pixels are linearly interpolated so that the
centers of the output pixels are 1/4 and 3/4 of the way between input pixel centers.
When rounding, 0.5 is rounded down and up at alternative pixels locations to prevent bias towards
larger values (ordered dither pattern).
Considering adjacent input pixels A, B, and C, B is upsampled to pixels B0 and B1 so that
```
B0 = round_half_down((1/4) * A + (3/4) * B)
B1 = round_half_up((3/4) * B + (1/4) * C)
```
This method,  is the default chroma upsampling method in the well-established libjpeg-turbo library,
also referred as "smooth" or "fancy" upsampling.

## Attributes

- **pixel_format** (STRING, optional): Pixel format. Can be one of "RGB", "BGR", or "Grayscale".

## Inputs (1 - 1)

- **encoded_stream** (T1): Encoded stream

## Outputs (1 - 1)

- **image** (T2): Decoded image

## Type Constraints

- **T1**: tensor(uint8)
  Constrain input types to 8-bit unsigned integer tensor.
- **T2**: tensor(uint8)
  Constrain output types to 8-bit unsigned integer tensor.
