use imageproc::window;
use clap::{Arg, App};

use std::error::Error;

use lpr_rust::Lpr;


fn main() -> Result<(), Box<dyn Error>>{
    let matches = App::new("LPR")
                    .version("0.1.0")
                    .author("kingrong")
                    .about("This program is meant for my graduation paper")
                    .arg(Arg::with_name("INPUT")
                        .help("image file with license plate")
                        .required(true)
                        .index(1))
                    .get_matches();
    let file_name = matches.value_of("INPUT").ok_or("image is required")?;

    let img = image::open(file_name)?;

    let lpr = Lpr::new("./models/detect.pb", "./models/ocr_plate_all_gru.pb", "./models/fine_mapping.pb")?;
    let res_img = lpr.recognize(&img)?;
    window::display_image("res", &res_img.to_rgba(), 700, 700);

    Ok(())
}


