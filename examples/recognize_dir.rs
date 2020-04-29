use std::error::Error;
use std::env::args;
use std::process;
use std::time::SystemTime;
use std::fs;

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
    let dir = fs::read_dir(path)?;

    let mut speeds = Vec::new();
    let mut scores = Vec::new();
    let mut total_amount = 0;
    let mut success = 0;
    for entry_result in dir {
        if let Ok(item) = entry_result {
            let path = item.path();
            print!("file: {:?},  ", path);
            let file_ext = path.extension();
            if let Some(ext) = file_ext {
                if ext == "jpg" {
                    let before_time = SystemTime::now();
                    let img = image::open(&path)?;
                    let res = lpr.recognize(&img)?;
                    let after_time = SystemTime::now();
                    let duration = after_time.duration_since(before_time)?;
                    let speed = duration.as_millis();
                    total_amount += 1;
                    if !res.is_empty() {
                        let (_, _, score) = res.first().unwrap();
                        scores.push(*score);
                        speeds.push(speed);
                        success += 1;
                    }
                    println!("res: {:?}, speed: {}",  res, speed);
                    if total_amount == 150 {
                        break;
                    }
                }
            }
        }
    }
    let total_score = scores.iter().fold(0.0, |mut score, item| {
        score += item;
        score
    });
    let total_speed = speeds.iter().fold(0, |mut speed, item| {
        speed += item;
        speed
    });
    let average_score = total_score/scores.len() as f32;
    let average_speed = total_speed/speeds.len() as u128;
    println!("total_amount: {}, success: {}, average_score: {}, average_speed: {}",
        total_amount, success, average_score, average_speed);
    Ok(())
}
