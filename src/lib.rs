use tensorflow::{ Tensor, Session, SessionOptions, Graph, SessionRunArgs, ImportGraphDefOptions, TensorType };
use image::{ GenericImageView, imageops::FilterType, DynamicImage };
use imageproc::{ drawing, rect, filter, contrast, window };
use rusttype::{ Font, Scale };

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use error::LprError;

pub mod utils;
pub mod error;
pub mod image_process;


// CHARS for Chinese license plate
const CHARS: [&str; 83] = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z","港","学","使","警","澳","挂","军","北","南","广","沈","兰","成","济","海","民","航","空"
             ];
const FONT_DATA: &[u8] = include_bytes!("../fonts/platech.ttf");

pub struct Lpr {
    detection: LpDetect,
    ocr: LprPart,
    fine_mapping: LprPart,
}

impl Lpr {

    pub fn new(detection_pb: impl AsRef<Path>, ocr_pb: impl AsRef<Path>, fine_mapping_pb: impl AsRef<Path>) -> Result<Self, LprError> {
        // read ocr proto
        let ocr = LprPart::new(ocr_pb, "input_1", "dense_2/truediv")?;
        let fine_mapping = LprPart::new(fine_mapping_pb, "input_1", "relu4/Relu")?;
        let detection = LpDetect::new(detection_pb, "image_tensor", "detection_boxes", "detection_scores")?;
        Ok(Lpr{ detection, ocr, fine_mapping })
    }

    /// Recognize one image
    pub fn recognize(&self, img: &DynamicImage) -> Result<DynamicImage, LprError> {
        let mut detected = img.clone().to_rgb();

        let score_needed = 0.6;
        let boxes_and_scores = self.get_boxes_and_scores(img, score_needed)?;
        for (a_box, _) in boxes_and_scores {
            let mut a_box = a_box;
            let [x, y, width, height] = a_box;
            let plate = img.view(x, y, width, height).to_image();
            let plate = self.fine_mapping_vertical(&DynamicImage::ImageRgba8(plate), &mut a_box)?;
            let (ocr_res, ocr_score) = self.get_ocr_result(&plate)?;
            let [x, y, width, height] = a_box;
            let rect = rect::Rect::at(x as i32, y as i32).of_size(width, height);
            drawing::draw_hollow_rect_mut(&mut detected, rect, image::Rgb([255, 0, 0]));
            let text = format!("{}--{}", ocr_res, ocr_score);
            let font = Font::from_bytes(FONT_DATA)?;
            let scale = Scale::uniform(24.0);
            drawing::draw_text_mut(&mut detected, image::Rgb([255, 0, 0]), x, y, scale, &font, &text);
        }
        
        Ok(DynamicImage::ImageRgb8(detected))
    }
    
    /// get boxes and scores
    /// box here is in a format of x,y,width,height
    /// the result which score is under score_needed will be failed
    //TODO here I just assume that boxes and scores from tensor is in pair, need more process
    pub fn get_boxes_and_scores(&self, img: &DynamicImage, score_needed: f32) -> Result<Vec<([u32; 4], f32)>, LprError> {
        let detection = &self.detection;

        let (width, height) = img.dimensions();
        let img_data = &img.to_rgb().to_vec();
        let img_tensor = Tensor::new(&[1, height as u64, width as u64, 3]);
        let img_tensor = img_tensor.with_values(&img_data)?;

        // box ymin xmin ymax xmax normalize 1
        let (boxes, scores) = detection.run(&img_tensor)?;
        // vec<(box, score)>  box: x,y,width,height
        let detect_res: Vec<([u32; 4], f32)> = boxes.chunks(4).enumerate().filter(|(index, _)| {
            scores[*index] > score_needed
        }).map(|(index, v)| {
            let ymin = (v[0]*(height as f32)) as u32;
            let xmin = (v[1]*(width as f32)) as u32;
            let ymax = (v[2]*(height as f32)) as u32;
            let xmax = (v[3]*(width as f32)) as u32;
            let a_box = [xmin, ymin, xmax - xmin, ymax - ymin];
            (a_box, scores[index])
        }).collect();
        Ok(detect_res)
    }
    

    pub fn get_ocr_result(&self, img: &DynamicImage) -> Result<(String, f32), LprError> {
        let img = utils::equalize_hist_in_gray(&img);

        // transpose
        let img = utils::transpose(&img);
        let img = img.resize_exact(48, 164, FilterType::Nearest);

        let img = img.to_luma();
        let mut img = filter::gaussian_blur_f32(&img, 1.4);
        contrast::equalize_histogram_mut(&mut img);
        let img = DynamicImage::ImageLuma8(img).to_bgr().to_vec();
        let img: Vec<f32> = img.into_iter().map(|v| v as f32).collect();
        let tensor_img: Tensor<f32> = Tensor::new(&[1, 164, 48u64, 3]);
        let tensor_img = tensor_img.with_values(&img[..])?;

        let ocr = &self.ocr;
        let ocr_res = ocr.run(&tensor_img)?;
        let res = Self::fast_decode(&ocr_res.to_vec(), [18, 84]);
        Ok(res)
    }
    
    //TODO
    // the process here is copied from a python code, but there seems has some puzzling thing
    pub fn fine_mapping_vertical(&self, img: &DynamicImage, rect: &mut [u32]) -> Result<DynamicImage, LprError> {
        let fine_mapping = &self.fine_mapping;

        let resized = img.resize_exact(66, 16, FilterType::Nearest);
        let resized = resized.to_bgr().to_vec();
        let resized: Vec<f32> = resized.into_iter().map(|v| v as f32/255.0).collect();

        let tensor: Tensor<f32> = Tensor::new(&[1, 16u64, 66, 3]);
        let tensor = tensor.with_values(&resized)?;
        let res_raw = fine_mapping.run(&tensor)?.to_vec();
        let img_shape = img.dimensions();
        let res: Vec<u32> = res_raw.iter().map(|v| (v * img_shape.0 as f32) as u32).collect();
        let h = res[0].checked_sub(3).unwrap_or(res[0]);
        let t = if res[1] + 2 >= img_shape.0 - 1 {
            img_shape.0 - 1
        } else {
            res[1] + 2
        };
        rect[2] -=  (rect[2] as f32*(1.0 - res_raw[1] + res_raw[0])) as u32;
        rect[0] += res[0] as u32;

        // try height
        let res_img = img.view(h, 0, t-h, img_shape.1).to_image();
        Ok(DynamicImage::ImageRgba8(res_img))
    }
    
    fn fast_decode(ocr_res: &[f32], shape: [usize; 2]) -> (String, f32){
        let argmax = utils::argmax_in_axis0(ocr_res, &shape);
        let (res, mut confidence) = argmax.iter().enumerate().skip(1).filter(|(i, v)| {
            **v < CHARS.len() && **v != argmax[i-1]
        }).fold((Vec::new(), 0.0), |(mut res, mut confidence), (i, v)| {
            res.push(CHARS[*v]);
            confidence += ocr_res[i*shape[1] + v];
            (res, confidence)
        });
        confidence = confidence/res.len() as f32;
        let res: String = res.join("");
        return (res, confidence);
    }

}

pub struct LpDetect {
    graph: Graph,
    session: Session,
    input_name: &'static str,
    box_name: &'static str,
    scores_name: &'static str,
}

impl LpDetect {

    pub fn new(pb_file: impl AsRef<Path>, input_name: &'static str, box_name: &'static str, scores_name: &'static str) -> Result<Self, LprError> {
        let mut pb_file = File::open(pb_file)?;
        let mut pb = Vec::new();
        pb_file.read_to_end(&mut pb)?;
        // import graph def
        let mut graph = Graph::new();
        let graph_def_options = ImportGraphDefOptions::new();
        graph.import_graph_def(&pb, &graph_def_options)?;
        // new session
        let session_option = SessionOptions::new();
        let session = Session::new(&session_option, &graph)?;
        Ok(Self { graph, session, input_name, box_name, scores_name })
    }

    /// get detection result
    /// return (boxes, scores)
    pub fn run(&self, input: &Tensor<u8>) -> Result<(Tensor<f32>, Tensor<f32>), LprError> {
        let graph = &self.graph;
        let session = &self.session;
        let input_name = self.input_name;
        let box_name = self.box_name;
        let scores_name = self.scores_name;
        let mut args = SessionRunArgs::new();
        args.add_feed(&graph.operation_by_name_required(input_name)?, 0, &input);
        //let res = args.request_fetch(&graph.operation_by_name_required("relu4/Relu")?, 0);
        let box_token = args.request_fetch(&graph.operation_by_name_required(box_name)?, 0);
        let scores_token = args.request_fetch(&graph.operation_by_name_required(scores_name)?, 0);
        session.run(&mut args)?;
        let boxes: Tensor<f32> = args.fetch(box_token)?;
        let scores: Tensor<f32> = args.fetch(scores_token)?;
        Ok((boxes, scores))
    }
    
}

pub struct LprPart {
    graph: Graph,
    session: Session,
    input_name: &'static str,
    output_name: &'static str,
}

impl LprPart {

    pub fn new(pb_file: impl AsRef<Path>, input_name: &'static str, output_name: &'static str) -> Result<Self, LprError> {
        // read fine_mapping proto
        let mut pb_file = File::open(pb_file)?;
        let mut pb = Vec::new();
        pb_file.read_to_end(&mut pb)?;
        // import graph def
        let mut graph = Graph::new();
        let graph_def_options = ImportGraphDefOptions::new();
        graph.import_graph_def(&pb, &graph_def_options)?;
        // new session
        let session_option = SessionOptions::new();
        let session = Session::new(&session_option, &graph)?;
        Ok(Self { graph, session, input_name, output_name })
    }

    pub fn run(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, LprError> {
        let graph = &self.graph;
        let session = &self.session;
        let input_name = self.input_name;
        let output_name = self.output_name;
        let mut args = SessionRunArgs::new();
        args.add_feed(&graph.operation_by_name_required(input_name)?, 0, &input);
        //let res = args.request_fetch(&graph.operation_by_name_required("relu4/Relu")?, 0);
        let res = args.request_fetch(&graph.operation_by_name_required(output_name)?, 0);
        session.run(&mut args)?;
        let res: Tensor<f32> = args.fetch(res)?;
        Ok(res)
    }
}


#[cfg(test)]
mod test {
    
    use image::{DynamicImage, ImageBuffer, Rgb, imageops::FilterType };
    use tensorflow::{ Tensor };

    use std::error::Error;
    use std::fs::File;

    use super::LprPart;

    fn transpose_out_place(input: &[u8], shape: [usize; 2]) -> Vec<u8> {
        let mut vec = vec![0; input.len()];
        let input: Vec<&[u8]> = input.chunks(3).collect();
        println!("chunked:{:?}", input);
        // 转置的方法 x*height + y
        input.chunks(shape[0]).enumerate().for_each(|(y, v1): (usize, &[&[u8]])| {
            v1.iter().enumerate().for_each(|(x, v2): (usize, &&[u8])| {
                v2.iter().enumerate().for_each(|(index3, v3)| {
                    let new_index = (y + x*shape[1])*3 + index3;
                    vec[new_index] = *v3;
                });
            });
        });
        return vec;
    }

    fn fine_mapping() -> Result<(), Box<dyn Error>>{
        let img = image::open("../test2.jpg")?;
        println!("this test will fail");
        let img = img.resize_exact(66, 16, FilterType::Nearest);
        let img = img.to_bytes();
        let img: Vec<f32> = img.into_iter().map(|v| v as f32/255.0).collect();
        
        let tensor: Tensor<f32> = Tensor::new(&[66u64, 16, 3]);
        let tensor = tensor.with_values(&img)?;
        let fine_mapping = LprPart::new("../fine_mapping.pb", "input_1", "relu4/Relu")?;
        let res = fine_mapping.run(&tensor)?;
        println!("res: {}", res);
        Ok(())
    }

    #[test]
    fn test() -> Result<(), Box<dyn Error>>{
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let tensor: Tensor<u8> = Tensor::new(&[3, 4]);
        let tensor = tensor.with_values(&v)?;
        println!("{}", tensor);
        Ok(())
    }

}
