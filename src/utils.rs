#[cfg(feature = "gtk-display")]
use gtk::prelude::*;
#[cfg(feature = "gtk-display")]
use gtk::{ Image, MessageType, MessageDialog, Window, ImageExt };
#[cfg(feature = "gtk-display")]
use gdk_pixbuf::{ Pixbuf, Colorspace };
#[cfg(feature = "gtk-display")]
use gio::prelude::*;

use image::{ DynamicImage, GenericImage, GenericImageView, ImageBuffer, Luma };
use tensorflow::{ Graph, Tensor, ImportGraphDefOptions };

use std::env;
use std::fs::File;
use std::path::Path;
use std::io::prelude::*;
use std::error::Error;

#[cfg(feature = "gtk-display")]
pub fn display_image(image: &DynamicImage, title: &str) {

    const BITS_PER_SAMPLE: i32 = 8;
    let uiapp = gtk::Application::new(Some("org.image.display"),
                                      gio::ApplicationFlags::FLAGS_NONE)
                                 .expect("Application::new failed");
    let (data, height, width, has_alpha, num_of_channel): (Vec<u8>, i32, i32, bool, i32) = match image {
        DynamicImage::ImageRgb8(img) => {
            (img.to_vec(), img.height() as i32, img.width() as i32, false, 3)
        },
        DynamicImage::ImageRgba8(img) => {{
            (img.to_vec(), img.height() as i32, img.width() as i32, true, 4)
        }},
        _ => {
            let img = image.to_rgb();
            (img.to_vec(), img.height() as i32, img.width() as i32, false, 3)
        }
    };
    let last_row_len = width * ((num_of_channel * BITS_PER_SAMPLE + 7) / 8);
    let rowstride = (data.len() as i32 - last_row_len)/(height - 1);
    let title = title.to_string();
    uiapp.connect_activate(move |app| {
        // We create the main window.
        let win = gtk::ApplicationWindow::new(app);

        // Then we set its size and a title.
        win.set_default_size(width, height);
        win.set_title(&title);

        let pix_buf = Pixbuf::new_from_mut_slice(data.clone(), Colorspace::Rgb, has_alpha,
            BITS_PER_SAMPLE, width, height, rowstride);
        let img = Image::new_from_pixbuf(Some(&pix_buf));
        win.add(&img);
        // Don't forget to make all widgets visible.
        win.show_all();
    });
    uiapp.run(&env::args().collect::<Vec<_>>());

}


// 这个是用于tensor的，tensor的展开机制是先y再x。
pub fn argmax_in_axis0(input: &[f32], shape: &[usize]) -> Vec<usize> {
    input.chunks(shape[1]).map(|v: &[f32]| {
        let mut max = v[0];
        let mut index = 0;
        v.iter().enumerate().for_each(|(i, v_in_v)| {
            if *v_in_v >= max {
                max = *v_in_v;
                index = i;
            }
        });
        index
    }).collect()
}


pub fn transpose(input: &DynamicImage) -> DynamicImage {
    let img = input.to_rgba();
    let mut output = DynamicImage::new_rgba8(img.height(), img.width());
    img.rows().enumerate().for_each(|(y, pixels)| {
        pixels.enumerate().for_each(|(x, pixel)| {
            output.put_pixel(y as u32, x as u32, *pixel);
        });
    });
    output
}

pub fn equalize_hist_in_gray(img: &DynamicImage) -> DynamicImage {
    let img = img.grayscale();
    let img = img.as_luma8().unwrap();
    let mut vec = img.to_vec();
    let len = vec.len();

    // 分布函数
    let mut df = [0; 256];
    for v in &vec {
        df[*v as usize] += 1;
    }
    // cdf
    let mut temp = df[0];
    df.iter_mut().skip(1).for_each(|v| {
        *v = *v + temp;
        temp = *v;
    });
    let mut iter = df.iter().filter(|v| **v != 0);
    let cdf_min = iter.next().unwrap();
    vec.iter_mut().for_each(|v| {
        let x = df[*v as usize] - cdf_min; 
        let y = len  - cdf_min;
        *v = ((x as f32/y as f32)*255.0).round() as u8;
    });
    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(img.width(), img.height(), vec).unwrap();
    DynamicImage::ImageLuma8(image_buffer)
}

pub fn get_all_operation_name(pb_file: impl AsRef<Path>) -> Result<Vec<String>, Box<dyn Error>> {
    let mut pb_file = File::open(pb_file)?;
    let mut pb = Vec::new();
    pb_file.read_to_end(&mut pb)?;
    // import graph def
    let mut graph = Graph::new();
    let graph_def_options = ImportGraphDefOptions::new();
    graph.import_graph_def(&pb, &graph_def_options)?;
    let graph_names: Vec<String> = graph.operation_iter().map(|op| op.name().unwrap()).collect();
    Ok(graph_names)
}
