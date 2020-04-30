License Plate Recognition In Rust

A Chinese license plate regognition implemention in rust. This repo use:

1. [image](https://github.com/image-rs/image) and [impageproc](https://github.com/image-rs/imageproc) for image processing
2. [tensorflow-rust](https://github.com/tensorflow/rust) for machine learning
3. some pre-trained model from [HyperLPR](https://github.com/zeusees/HyperLPR)

## Quick Start

```shell
cargo run ./test_images/1.jpg
```

## Example

1. detect
2. finemapping
3. oblique_fix
4. recognize
5. recognize_dir
