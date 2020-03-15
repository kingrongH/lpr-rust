/// This file is here because first I don't there is a imageproc lib which has a lot of image
/// process functions.
/// bug now this file is abandoned

use image::DynamicImage;

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
