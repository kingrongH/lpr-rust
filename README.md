License Plate Recognition In Rust

A Chinese license plate regognition implemention in rust. This repo use:

1. [image](https://github.com/image-rs/image) and [impageproc](https://github.com/image-rs/imageproc) for image processing
2. [tensorflow-rust](https://github.com/tensorflow/rust) for machine learning
3. some pre-trained model from [HyperLPR](https://github.com/zeusees/HyperLPR)
4. all the images for training and test comes from [CCPD](https://github.com/detectRecog/CCPD)

## Quick Start

Rust and `Cargo` is needed. If you don't have it yet, check [this](https://www.rust-lang.org/tools/install) out first!!

There is two way to use this repo: cli and lib

### cli

Use it as a cli tool, just run the following command.

```shell
cargo run ./test_images/1.jpg
```

Then you could see something like this:
![recognition](https://i.loli.net/2020/05/01/hLlxqUCVkDM8wAs.png)

### lib

Here is a simple example for using this as a library.

```Rust
use imageproc::window;

use std::error::Error;

use lpr_rust::Lpr;


fn main() -> Result<(), Box<dyn Error>>{

    let img = image::open(file_name)?;

    let lpr = Lpr::new("./models/detect.pb", "./models/ocr_plate_all_gru.pb", "./models/fine_mapping.pb")?;
    let res_img = lpr.recognize_and_draw(&img)?;
    window::display_image("res", &res_img.to_rgba(), 700, 700);

    Ok(())

}
```

## Example

1. detect - a simple detect example of Chinese license plate detect
2. oblique_fix - an example of fix oblique image
3. recognize - just recognize the image and show it in shell
4. recognize_dir - this is sort of benchmark thing

example for checking out examples

```Shell
cargo run --example detect test_images/2.jpg
```

![detect example](https://i.loli.net/2020/05/01/SWtMalhNILJmAfP.png)


## How

There should be three steps to making a license recognition tool:

1. detection
2. image processing
3. recognition

There is two main macheine learning model respectively used for detection and recognition

### Detection

This repo use [object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) for model training. And I use ssd_mobilenet_v3_small_coco for Transfer Learning

The model I used has been trained with 1167 pics, which is very little.

You dive into [this](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) for training more image

### Image processing

The main problem that I encoutered is how to fix oblique plate in image detected

Here is an image you may want to see how do I do this:

![image processing](https://i.loli.net/2020/05/01/dt7aWonlE5YSkGK.png)

If you found a better way, please let me know.

### Recognition

I use a keras model(ocr_plate_all_gru) from [HyperLPR](https://github.com/zeusees/HyperLPR) for recognition

But in Rust you don't have way to load keras model, so the thing i do is convert keras model to tensorflow model, then load this model to recognize.

Here is a [blog](https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/) found on internet, maybe you can take a look.

## Known issue

In some cases, the program will freeze, this happens when creating a Projection, I think this is a bug of [imageproc](https://github.com/image-rs/imageproc), then I filed an issue [#412](https://github.com/image-rs/imageproc/issues/412)
