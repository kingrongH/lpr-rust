use imageproc::window;

use std::process;
use std::env::args;
use std::error::Error;

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
    // original image display
    let rgb = img.to_rgb();
    window::display_image("original image", &rgb, 500, 500);
    let after_fix = image_process::perspective_trans(&rgb).expect("fix failed");
    // after fix image display
    window::display_image("after fix", &after_fix, 500, 500);

    Ok(())
}
