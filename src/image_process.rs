/// This file contains function to do perspective transformation and some gradient and canny
/// calculation. But now only perspective transformation functions are needed

use image::{ DynamicImage, Rgb, RgbImage };
use imageproc::{ filter, edges, morphology, distance_transform::Norm,
    geometric_transformations::{self, Interpolation}, hough, hough::PolarLine };
use palette::{ Hsv, Srgb, Pixel };

use std::cmp::Ordering;

/// Perform perspective transformation for a oblique image, return None when projection cannot be
/// created
/// Steps:
/// 1. gaussian blur avoid noises
/// 2. pick all blue pixels out make then white, and other pixels black
/// 3. do morphology close operation to eliminate the char in plate, now get a binary img with plate
///    shape
/// 4. gaussian blur again to avoid noises
/// 5. canny img get edges of plate
/// 6. morphology dilate operation to make edge thicker
/// 7. detect hough lines using number of votes that is relative to the height
/// 8. get intersection points between lines(including border)
/// 9. get warp points(represent the plate four corners) by filtering some points
/// 10. perspective transformation from warp points to the four corner of this image
/// ### note 
/// Something that can used tweak the quality of transformation
/// 1. votes and suppression_radius in detecting hough lines
/// 2. corlor range in filter out blue pixels
/// 3. morphology distance 
/// 4. filter algorithm for getting warp points
pub fn perspective_trans(img: &RgbImage) -> Option<RgbImage> {
    let (width, height) = img.dimensions();
    // the img with proper width-height ratio don't need to be transformed
    //let ratio = width as f32/height as f32;
    //if ratio >= 0.9*NORM_BLUE_PLATE_RATIO && ratio <= 1.1*NORM_BLUE_PLATE_RATIO {
        //return img.clone();
    //}
    let img = filter::gaussian_blur_f32(&img, 1.4);
    let mut gray_img = DynamicImage::new_luma8(width, height).to_luma();
    // filter all the blue pixels
    img.rows().enumerate().for_each(|(y, pixels)| {
        pixels.enumerate().for_each(|(x, pixel)| {
            let data = pixel.0;
            let srgb = Srgb::from_raw(&data).into_format();
            let hsv: Hsv = srgb.into();
            let h = hsv.hue.to_positive_degrees();
            let s = hsv.saturation * 255.0;
            let v = hsv.value * 255.0;
            if h > 180.0 && h < 360.0 && s>=43.0 && s<=255.0 && v >=46.0 && v<=255.0 {
                gray_img.put_pixel(x as u32, y as u32, image::Luma([255]));
            }
        });
    });
    // 将中间的字去掉
    morphology::close_mut(&mut gray_img, Norm::L1, 3);
    //imageops::invert(&mut gray_img);
    //减少噪声
    let gray_img = filter::gaussian_blur_f32(&gray_img, 1.4);
    //window::display_image("white image", &gray_img, 500, 500);
    let mut gray_img = edges::canny(&gray_img, 0.13, 0.74);
    // expand border
    morphology::dilate_mut(&mut gray_img, Norm::LInf, 1);
    //window::display_image("canny", &gray_img, 500, 500);
    let options = hough::LineDetectionOptions {
        vote_threshold: (height as f32 * 0.4) as u32,
        suppression_radius: (height as f32 * 0.4) as u32,
    };
    let lines = hough::detect_lines(&gray_img, options);
    //let gray_img = DynamicImage::ImageLuma8(gray_img);
    //let mut rgb = gray_img.to_rgb();
    //hough::draw_polar_lines_mut(&mut rgb, &lines, Rgb([255, 0, 0]));
    //window::display_image("with lines", &rgb, width, height);
    let warp_points = get_warp_points(&lines, width, height);
    let to_points = [(0.0, 0.0), ((width - 1) as f32, 0.0), (0.0, (height - 1) as f32), ((width - 1) as f32, (height - 1) as f32)];
    println!("warp points: {:?}, to points: {:?}", warp_points, to_points);
    let projection = geometric_transformations::Projection::from_control_points(warp_points, to_points);
    if projection.is_none() {
        return None;
    }
    let img = geometric_transformations::warp(&img, &projection.unwrap(), Interpolation::Bicubic, Rgb([0, 0, 0]));
    //let img = DynamicImage::ImageRgb8(img).resize_exact(164, 48, imageops::FilterType::Nearest).to_rgb();
    Some(img)
}

/// get 4 points for warp projection: top_left, top_right, bottom_left, bottom_right
/// the point use order: intersection points between lines 、between line and border、corners
/// steps: 
/// 1. filter pointer in each corner area which defined by two variable: horizontal and vertical
/// 2. chose the points as close to the corder as much, othewise using the corder points (like
///    (0,0), (width-1, height-1))
fn get_warp_points(lines: &[PolarLine], width: u32, height: u32) -> [(f32, f32); 4] {
    // intersection points between lines
    let lines_points: Vec<(f32, f32)> = get_intersection_points(&lines).into_iter().filter(|(x, y)| {
        *x > 0.0 && *y > 0.0 && *x < width as f32 && *y < height as f32 
    }).collect();
    // intersection points between line and border
    let mut border_intersection_points = Vec::new();
    for line in lines {
        if let Some((p1, p2)) = intersection_points_with_border(line, width, height) {
            border_intersection_points.push(p1);
            border_intersection_points.push(p2);
        }
    }
    // horizontal and vertical borders for choose the corner points
    let horizontal = 0.25 * (width as f32/2.0);
    let vertical = 1.0 * (height as f32/2.0);
    let top_left = {
        let sort_func = |(x1, y1): &&(f32, f32), (x2, y2): &&(f32, f32)| {
            if x1 < x2 {
                Ordering::Less
            } else if x1 == x2 {
                if y1 < y2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else {
                Ordering::Greater
            }
        };
        let filter_func = |(x, y): &&(f32, f32)| {
            if *x < horizontal && *y < vertical {
                return true
            }
            false
        };
        let lines_point_option = lines_points.iter().filter(filter_func).min_by(sort_func);
        let border_point_option = border_intersection_points.iter().filter(filter_func).min_by(sort_func);
        if lines_point_option.is_some() {
            let p = lines_point_option.unwrap();
            (p.0, p.1)
        } else if border_point_option.is_some() {
            let p = border_point_option.unwrap();
            (p.0, p.1)
        } else {
            (0., 0.)
        }
    };
    let bottom_left = {
        let sort_func = |(x1, y1): &&(f32, f32), (x2, y2): &&(f32, f32)| {
            if x1 < x2 {
                Ordering::Less
            } else if x1 == x2 {
                if y1 > y2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else {
                Ordering::Greater
            }
        };
        let filter_func = |(x, y): &&(f32, f32)| {
            if *x < horizontal && *y > height as f32 - vertical {
                return true
            }
            false
        };
        let lines_point_option = lines_points.iter().filter(filter_func).min_by(sort_func);
        let border_point_option = border_intersection_points.iter().filter(filter_func).min_by(sort_func);
        if lines_point_option.is_some() {
            let p = lines_point_option.unwrap();
            (p.0, p.1)
        } else if border_point_option.is_some() {
            let p = border_point_option.unwrap();
            (p.0, p.1)
        } else {
            (0.0, (height - 1) as f32)
        }
    };
    let top_right = {
        let sort_func = |(x1, y1): &&(f32, f32), (x2, y2): &&(f32, f32)| {
            if x1 > x2 {
                Ordering::Less
            } else if x1 == x2 {
                if y1 < y2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else {
                Ordering::Greater
            }
        };
        let filter_func = |(x, y): &&(f32, f32)| {
            if *x > width as f32 - horizontal && *y < vertical {
                return true
            }
            false
        };
        let lines_point_option = lines_points.iter().filter(filter_func).min_by(sort_func);
        let border_point_option = border_intersection_points.iter().filter(filter_func).min_by(sort_func);
        if lines_point_option.is_some() {
            let p = lines_point_option.unwrap();
            (p.0, p.1)
        } else if border_point_option.is_some() {
            let p = border_point_option.unwrap();
            (p.0, p.1)
        } else {
            ((width - 1) as f32, 0.0)
        }
    };
    let bottom_right = {
        let sort_func = |(x1, y1): &&(f32, f32), (x2, y2): &&(f32, f32)| {
            if x1 > x2 {
                Ordering::Less
            } else if x1 == x2 {
                if y1 > y2 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else {
                Ordering::Greater
            }
        };
        let filter_func = |(x, y): &&(f32, f32)| {
            if *x > width as f32 - horizontal && *y > height as f32 - vertical {
                return true
            }
            false
        };
        let lines_point_option = lines_points.iter().filter(filter_func).min_by(sort_func);
        let border_point_option = border_intersection_points.iter().filter(filter_func).min_by(sort_func);
        if lines_point_option.is_some() {
            let p = lines_point_option.unwrap();
            (p.0, p.1)
        } else if border_point_option.is_some() {
            let p = border_point_option.unwrap();
            (p.0, p.1)
        } else {
            ((width - 1) as f32, (height - 1) as f32)
        }
    };

    let warp_points = [top_left, top_right, bottom_left, bottom_right];
    warp_points
}


/// Get all intersection points from lines
fn get_intersection_points(lines: &[PolarLine]) -> Vec<(f32, f32)> {
    let mut points = Vec::new();
    lines.iter().enumerate().for_each(|(index, line1)| {
        let len = lines.len();
        if index == len - 1 {
            return;
        }
        for line2 in &lines[index+1..len] {
            if let Some(point) = get_intersection_point(line1, line2) {
                points.push(point)
            }
        }
    });
    points
}

/// get intersection point of two line return None if two line are disjoint
fn get_intersection_point(line1: &PolarLine, line2: &PolarLine) -> Option<(f32, f32)> {
    // disjoint
    if line1.angle_in_degrees == line2.angle_in_degrees {
        return None;
    }
    let radian1 = degrees_to_radians(line1.angle_in_degrees as f32);
    let radian2 = degrees_to_radians(line2.angle_in_degrees as f32);

    let x1 = line1.r * radian1.cos();
    let y1 = line1.r * radian1.sin();
    let x2 = line2.r * radian2.cos();
    let y2 = line2.r * radian2.sin();
    // theta of the line1
    let theta1 = radian1 - std::f32::consts::FRAC_PI_2;
    let a1 = theta1.tan();
    let b1 = y1 - a1*x1;
    // theta of the line2
    let theta2 = radian2 - std::f32::consts::FRAC_PI_2;
    let a2 = theta2.tan();
    let b2 = y2 - a2*x2;
    //intersection point
    let x0 = (b2 - b1)/(a1 - a2);
    let y0 = a1 * x0 + b1;
    //println!("a1: {}, b1: {}, a2: {}, b2: {}, x0: {}, y0: {}, b2-b1: {}, a1-a2: {}", a1, a2, a2, b2, x0, y0, b2-b1, a1-a2);
    Some((x0, y0))
}

/// Returns the intersection points of a `PolarLine` with an image of given width and height,
/// or `None` if the line and image bounding box are disjoint. The x value of an intersection
/// point lies within the closed interval [0, image_width] and the y value within the closed
/// interval [0, image_height].
fn intersection_points_with_border(
    line: &PolarLine,
    image_width: u32,
    image_height: u32,
) -> Option<((f32, f32), (f32, f32))> {
    let r = line.r;
    let m = line.angle_in_degrees;
    let w = image_width as f32;
    let h = image_height as f32;

    // Vertical line
    if m == 0 {
        return if r >= 0.0 && r <= w {
            Some(((r, 0.0), (r, h)))
        } else {
            None
        };
    }

    // Horizontal line
    if m == 90 {
        return if r >= 0.0 && r <= h {
            Some(((0.0, r), (w, r)))
        } else {
            None
        };
    }

    let theta = degrees_to_radians(m as f32);
    let sin = theta.sin();
    let cos = theta.cos();

    let right_y = cos.mul_add(-w, r) / sin;
    let left_y = r / sin;
    let bottom_x = sin.mul_add(-h, r) / cos;
    let top_x = r / cos;

    let mut start = None;

    if right_y >= 0.0 && right_y <= h {
        let right_intersect = (w, right_y);
        if let Some(s) = start {
            return Some((s, right_intersect));
        }
        start = Some(right_intersect);
    }

    if left_y >= 0.0 && left_y <= h {
        let left_intersect = (0.0, left_y);
        if let Some(s) = start {
            return Some((s, left_intersect));
        }
        start = Some(left_intersect);
    }

    if bottom_x >= 0.0 && bottom_x <= w {
        let bottom_intersect = (bottom_x, h);
        if let Some(s) = start {
            return Some((s, bottom_intersect));
        }
        start = Some(bottom_intersect);
    }

    if top_x >= 0.0 && top_x <= w {
        let top_intersect = (top_x, 0.0);
        if let Some(s) = start {
            return Some((s, top_intersect));
        }
    }

    None
}

fn degrees_to_radians(degrees: f32) -> f32 {
    let pi = std::f32::consts::PI;
    degrees *  pi / 180.0
}

// gradient_theta, in radian
struct Gradients {
    gradient_x: Vec<i16>,
    gradient_y: Vec<i16>,
    gradient_amp: Vec<f32>,
    shape: (usize, usize),
}

#[allow(dead_code)]
impl Gradients {

    fn from_image(img: &DynamicImage) -> Self {
        //let img = img.blur(1.4);
        let gray_img = img.to_luma();

        let (width, height) = gray_img.dimensions();
        // gradient in x axis
        let mut gradient_x = vec![0i16; (width as usize - 1)*(height as usize - 1)];
        // gradient in y axis
        let mut gradient_y = vec![0i16; (width as usize - 1)*(height as usize - 1)];
        // gradient_amplitude
        let mut gradient_amp = vec![0f32; (width as usize - 1)*(height as usize - 1)];
        gray_img.rows().enumerate().for_each(|(y, pixels)| {
            pixels.enumerate().for_each(|(x, pixel)| {
                let next_x = x as u32 + 1;
                let next_y = y as u32 + 1;
                // 最后一行和最后一列没有gradient
                if x != width as usize - 1 && y != height as usize - 1 {
                    let next_x_pixel_value = gray_img[(next_x, y as u32)].0[0] as i16;
                    let next_y_pixel_value = gray_img[(x as u32, next_y)].0[0] as i16;

                    let gradient_x_value = next_x_pixel_value - pixel.0[0] as i16;
                    let gradient_y_value = next_y_pixel_value - pixel.0[0] as i16;
                    let gradient_amp_value = ((gradient_x_value.pow(2) + gradient_y_value.pow(2)) as f32).sqrt();

                    gradient_x[y*(height as usize) + x] = gradient_x_value;
                    gradient_y[y*(height as usize) + x] = gradient_y_value;
                    gradient_amp[y*(height as usize) + x] = gradient_amp_value;
                }
            });
        });
        let shape = (width as usize - 1, height as usize - 1);
        Self { gradient_x, gradient_y, gradient_amp, shape }
    }

    fn get_x(&self, (x, y): (usize, usize)) -> i16 {
        let index = y*self.shape.0 + x;
        self.gradient_x[index]
    }

    fn get_y(&self, (x, y): (usize, usize)) -> i16 {
        let index = y*self.shape.0 + x;
        self.gradient_y[index]
    }

    fn get_amp(&self, (x, y): (usize, usize)) -> f32 {
        let index = y*self.shape.0 + x;
        let gradient_x_value = self.gradient_x[index];
        let gradient_y_value = self.gradient_y[index];
        let gradient_amp_value = ((gradient_x_value.pow(2) + gradient_y_value.pow(2)) as f32).sqrt();
        gradient_amp_value
    }
    
    fn get_theta(&self, (x, y): (usize, usize)) -> f32 {
        let index = y*self.shape.0 + x;
        let gradient_x_value = self.gradient_x[index];
        let gradient_y_value = self.gradient_y[index];
        let gradient_theta_value = (gradient_y_value as f32/(gradient_x_value as f32 + 0.000000001)).atan();
        gradient_theta_value
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    fn get_amp_all(&self) -> &[f32] {
        &self.gradient_amp
    }
    
}

#[allow(dead_code)]
struct NonMaxSuppress {
    pub data: Vec<f32>,
    shape: (usize, usize)
}

#[allow(dead_code)]
impl NonMaxSuppress {

    fn from_gradient(gradient: &Gradients) -> Self {
        let (width, height) = gradient.get_shape();
        //let gradient_amp = gradient.get_amp_all();
        let mut nms = vec![0.0; width*height]; 
        nms.chunks_mut(width).enumerate().for_each(|(y, y_values)| {
            y_values.iter_mut().enumerate().for_each(|(x, v)| {
                //TODO why
                // 边缘取0
                if x == 0 || y == 0 || x == width-1 || y == height - 1 {
                    *v = 0.0;
                    return;
                }
                // calc
                // 如果当前梯度为0，该点就不是边缘点
                if gradient.get_amp((x, y)) == 0.0 {
                    *v = 0.0;
                    return;
                }
                let grad_x = gradient.get_x((x, y));
                let grad_y = gradient.get_y((x, y));
                let grad_current = gradient.get_amp((x, y));

                //TODO double check if here is some thing wrong
                // 求出四个相邻的点的梯度幅值
                // 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                let (grad1, grad2, grad3, grad4, weight) = if grad_y.abs() > grad_x.abs() {
                    let weight = grad_x.abs() as f32 / grad_y.abs() as f32;
                    let grad2 = gradient.get_amp((x-1, y));
                    let grad4 = gradient.get_amp((x+1, y));

                    // 如果 x, y 方向导数符号一致
                    // 像素点位置关系
                    //   g1 g2
                    //      c
                    //      g4 g3
                    let (grad1, grad3) = if grad_x*grad_y > 0 {
                        let grad1 = gradient.get_amp((x-1, y-1));
                        let grad3 = gradient.get_amp((x+1, y+1));
                        (grad1, grad3)
                    // 如果 x，y 方向导数符号相反
                    // 像素点位置关系
                    //    g2 g1
                    //    c
                    // g3 g4
                    } else {
                        let grad1 = gradient.get_amp((x-1, y+1));
                        let grad3 = gradient.get_amp((x+1, y-1));
                        (grad1, grad3)
                    };
                    (grad1, grad2, grad3, grad4, weight)

                // 如果 y 方向梯度值比较大，说明导数方向趋向于 x 分量
                } else {
                    let weight = grad_y.abs() as f32 / grad_x.abs() as f32;
                    let grad2 = gradient.get_amp((x, y-1));
                    let grad4 = gradient.get_amp((x, y+1));
                    
                    // 如果 x, y 方向导数符号一致
                    // 像素点位置关系
                    //      g3
                    // g2 c g4
                    // g1
                    let (grad1, grad3) = if grad_x*grad_y > 0 {
                        let grad1 = gradient.get_amp((x+1, y-1));
                        let grad3 = gradient.get_amp((x-1, y+1));
                        (grad1, grad3)
                    // 如果 x，y 方向导数符号相反
                    // 像素点位置关系
                    // g1
                    // g2 c g4
                    //      g3
                    } else {
                        let grad1 = gradient.get_amp((x-1, y-1));
                        let grad3 = gradient.get_amp((x+1, y+1));
                        (grad1, grad3)
                    };
                    (grad1, grad2, grad3, grad4, weight)
                };

                // 利用 grad1-grad4 对梯度进行插值
                let grad_temp1 = weight*grad1 + (1.0-weight)*grad2;
                let grad_temp2 = weight*grad3 + (1.0-weight)*grad4;
                // 当前像素的梯度是局部的最大值，可能是边缘点
                if grad_current >= grad_temp1 && grad_current >= grad_temp2 {
                    *v = grad_current;
                } else {
                    *v = 0.0
                }
            });
        });
        return Self{ data: nms, shape: (width, height) };
    } 

    fn get_value(&self, (x, y): (usize, usize)) -> f32 {
        let index = y*self.shape.0 + x;
        self.data[index]
    }

    fn get_max(&self) -> f32 {
        let nms = &self.data;
        let mut max_nms = nms[0];
        for v in nms {
            if *v > max_nms {
                max_nms = *v;
            }
        }
        max_nms
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

}

#[allow(dead_code)]
// 双阈值边界选取
fn double_threshold(img: &DynamicImage, threshold_low: f32, threshold_high: f32) -> Vec<u8> {
    let gradient = Gradients::from_image(&img);
    let nms = NonMaxSuppress::from_gradient(&gradient);
    
    let max_nms = nms.get_max();
    let threshold_low = threshold_low*max_nms;
    let threshold_high = threshold_high*max_nms;
    let (width, height) = nms.get_shape();
    let mut double_threshold = vec![0; width*height];
    double_threshold.chunks_mut(width).enumerate().for_each(|(y, values)| {
        values.iter_mut().enumerate().for_each(|(x, value)| {
            if x==0 || y==0  || x==width-1 || y==height-1 {
                return;
            } 
            let nms_current = nms.get_value((x, y));
            let g1 = nms.get_value((x-1, y-1));
            let g2 = nms.get_value((x-1, y));
            let g3 = nms.get_value((x-1, y+1));
            let g4 = nms.get_value((x, y-1));
            let g5 = nms.get_value((x, y));
            let g6 = nms.get_value((x, y+1));
            let g7 = nms.get_value((x+1, y-1));
            let g8 = nms.get_value((x+1, y));
            let g9 = nms.get_value((x+1, y+1));
            
            let area = [g1, g2, g3, g4, g5, g6, g7, g8, g9];
            *value = if nms_current < threshold_low {
                0
            } else if nms_current > threshold_high {
                1
            } else if area.iter().any(|v| *v < threshold_high) {
                1
            } else {
                0
            }
        });
    });

    return double_threshold;
}

#[allow(dead_code)]
// 非极大值抑制
fn non_maximum_suppression(img: &DynamicImage) -> Vec<f32> {
    let gradient = Gradients::from_image(img);
    let (width, height) = gradient.get_shape();
    //let gradient_amp = gradient.get_amp_all();
    let mut nms = vec![0.0; width*height]; 
    nms.chunks_mut(width).enumerate().for_each(|(y, y_values)| {
        y_values.iter_mut().enumerate().for_each(|(x, v)| {
            //TODO why
            // 边缘取0
            if x == 0 || y == 0 || x == width-1 || y == height - 1 {
                *v = 0.0;
                return;
            }
            // calc
            // 如果当前梯度为0，该点就不是边缘点
            if gradient.get_amp((x, y)) == 0.0 {
                *v = 0.0;
                return;
            }
            let grad_x = gradient.get_x((x, y));
            let grad_y = gradient.get_y((x, y));
            let grad_current = gradient.get_amp((x, y));

            //TODO double check if here is some thing wrong
            // 求出四个相邻的点的梯度幅值
            // 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
            let (grad1, grad2, grad3, grad4, weight) = if grad_y.abs() > grad_x.abs() {
                let weight = grad_x.abs() as f32 / grad_y.abs() as f32;
                let grad2 = gradient.get_amp((x-1, y));
                let grad4 = gradient.get_amp((x+1, y));

                // 如果 x, y 方向导数符号一致
                // 像素点位置关系
                //   g1 g2
                //      c
                //      g4 g3
                let (grad1, grad3) = if grad_x*grad_y > 0 {
                    let grad1 = gradient.get_amp((x-1, y-1));
                    let grad3 = gradient.get_amp((x+1, y+1));
                    (grad1, grad3)
                // 如果 x，y 方向导数符号相反
                // 像素点位置关系
                //    g2 g1
                //    c
                // g3 g4
                } else {
                    let grad1 = gradient.get_amp((x-1, y+1));
                    let grad3 = gradient.get_amp((x+1, y-1));
                    (grad1, grad3)
                };
                (grad1, grad2, grad3, grad4, weight)

            // 如果 y 方向梯度值比较大，说明导数方向趋向于 x 分量
            } else {
                let weight = grad_y.abs() as f32 / grad_x.abs() as f32;
                let grad2 = gradient.get_amp((x, y-1));
                let grad4 = gradient.get_amp((x, y+1));
                
                // 如果 x, y 方向导数符号一致
                // 像素点位置关系
                //      g3
                // g2 c g4
                // g1
                let (grad1, grad3) = if grad_x*grad_y > 0 {
                    let grad1 = gradient.get_amp((x+1, y-1));
                    let grad3 = gradient.get_amp((x-1, y+1));
                    (grad1, grad3)
                // 如果 x，y 方向导数符号相反
                // 像素点位置关系
                // g1
                // g2 c g4
                //      g3
                } else {
                    let grad1 = gradient.get_amp((x-1, y-1));
                    let grad3 = gradient.get_amp((x+1, y+1));
                    (grad1, grad3)
                };
                (grad1, grad2, grad3, grad4, weight)
            };

            // 利用 grad1-grad4 对梯度进行插值
            let grad_temp1 = weight*grad1 + (1.0-weight)*grad2;
            let grad_temp2 = weight*grad3 + (1.0-weight)*grad4;
            // 当前像素的梯度是局部的最大值，可能是边缘点
            if grad_current >= grad_temp1 && grad_current >= grad_temp2 {
                *v = grad_current;
            } else {
                *v = 0.0
            }
        });
    });
    return nms;
}


#[allow(dead_code)]
// 图像梯度，梯度幅值，梯度方向计算
fn gradients(img: &DynamicImage) {
    //utils::display_image(img, "before blur");
    let img = img.blur(1.4);
    //utils::display_image(&img, "after blur");
    let gray_img = img.to_luma();
    //utils::display_image(&gray_img, "after gray");
    let (width, height) = gray_img.dimensions();
    // gradient in x axis
    let mut gradient_x = vec![0i16; (width as usize - 1)*(height as usize - 1)];
    // gradient in y axis
    let mut gradient_y = vec![0i16; (width as usize - 1)*(height as usize - 1)];
    // gradient_amplitude
    let mut gradient_amp = vec![0f32; (width as usize - 1)*(height as usize - 1)];
    // gradient_theta
    let mut gradient_theta = vec![0f32; (width as usize - 1)*(height as usize - 1)];
    gray_img.rows().enumerate().for_each(|(y, pixels)| {
        pixels.enumerate().for_each(|(x, pixel)| {
            // 最后一行和最后一列没有gradient
            let next_x = x as u32 + 1;
            let next_y = y as u32 + 1;
            if x != width as usize - 1 && y != height as usize - 1 {
                let next_x_pixel_value = gray_img[(next_x, y as u32)].0[0] as i16;
                let next_y_pixel_value = gray_img[(x as u32, next_y)].0[0] as i16;

                let gradient_x_value = next_x_pixel_value - pixel.0[0] as i16;
                let gradient_y_value = next_y_pixel_value - pixel.0[0] as i16;
                let gradient_amp_value = ((gradient_x_value.pow(2) + gradient_y_value.pow(2)) as f32).sqrt();
                let gradient_theta_value = (gradient_y_value as f32/gradient_x_value as f32).atan();
                gradient_x[y*(height as usize) + x] = gradient_x_value;
                gradient_y[y*(height as usize) + x] = gradient_y_value;
                gradient_amp[y*(height as usize) + x] = gradient_amp_value;
                gradient_theta[y*(height as usize) + x] = gradient_theta_value;
            }
        });
    });
}
