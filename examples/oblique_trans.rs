use image::{ DynamicImage, GenericImageView };
use imageproc::window;

use std::error::Error;
use std::process;
use std::env::args;

use lpr_rust::image_process;


fn main() -> Result<(), Box<dyn Error>> {

    let mut args = args();
    args.next();
    let path = args.next();
    let path = match path {
        Some(path) => path,
        None => {
            eprintln!("didn't get a image from args");
            process::exit(1);
        }
    };

    let img = image::open(path)?;
    window::display_image("original imag", &img.to_rgb(), 500, 500);
    let rgb = img.to_rgb();
    let after_trans = image_process::perspective_trans(&rgb).expect("trans failed");
    window::display_image("after trans", &after_trans, 500, 500);

    Ok(())
}