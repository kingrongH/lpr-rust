use imageproc::window;
use image::GenericImageView;

use std::error::Error;
use std::env::args;
use std::process;

use lpr_rust::Lpr;

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
    
    let lpr = Lpr::new("../models/detect.pb", "../models/ocr_plate_all_gru.pb", "../models/fine_mapping.pb")?;
    let img = image::open(path)?;
    let (width, height) = img.dimensions();
    let mut rect = [0, 0, width, height];
    window::display_image("原图", &img.to_rgb(), width, height);
    let img = lpr.fine_mapping_vertical(&img, &mut rect)?;
    window::display_image("左右边界回归后", &img.to_rgb(), width, height);

    Ok(())
}
