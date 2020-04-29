use imageproc::{ window, drawing, rect };
use image::GenericImageView;
use rusttype::{ Font, Scale };

use std::error::Error;
use std::env::args;
use std::process;

use lpr_rust::Lpr;

const FONT_DATA: &[u8] = include_bytes!("../fonts/platech.ttf");

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
    let mut img = image::open(path)?;
    let (width, height) = img.dimensions();
    let res = lpr.get_boxes_and_scores(&img, 0.8)?;
    for (a_box, score) in res {
        let [x, y, width, height] = a_box;
        let rect = rect::Rect::at(x as i32, y as i32).of_size(width, height);
        let text = format!("{:.3}", score);
        let font = Font::from_bytes(FONT_DATA)?;
        let scale = Scale::uniform(24.0);
        drawing::draw_text_mut(&mut img, image::Rgba([0, 255, 0, 0]), x, y, scale, &font, &text);
        drawing::draw_hollow_rect_mut(&mut img, rect, image::Rgba([255, 0, 0, 0]));
    }
    window::display_image("detect result", &img.to_rgb(), width, height);
    Ok(())
}
