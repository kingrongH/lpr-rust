use imageproc::{ window, drawing, rect };
use image::{ DynamicImage, GenericImageView };
use rusttype::{ Font, Scale };

use std::error::Error;
use std::env::args;
use std::process;
use std::time::{SystemTime, Duration};

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

    let lpr = Lpr::new("./models/detect.pb", "./models/ocr_plate_all_gru.pb", "./models/fine_mapping.pb")?;
    let img = image::open(path)?;
    //let (ocr_res, ocr_score) = lpr.get_ocr_result(&img)?;
    let before_time = SystemTime::now();
    let res = lpr.recognize(&img)?;
    let after_time = SystemTime::now();
    let duration = after_time.duration_since(before_time)?;
    let speed = duration.as_millis();
    println!("res: {:?}, spped: {}", res, speed);
    Ok(())
}
