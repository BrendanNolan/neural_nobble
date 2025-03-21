use image::{GrayImage, Luma};
use mnist::*;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use neural_nobble::{
    activation_functions::*, cost_functions::*, feed_forward::*, mini_batch::*, neural_network::*,
    one_hot::*, train::*,
};

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_size = 28 * 28;

    let train_data = Array2::from_shape_vec((50_000, image_size), trn_img)
        .expect("Error converting traininig images")
        .t()
        .map(|x| *x as f64 / 256.0);

    let train_labels =
        Array1::from_shape_vec(50_000, trn_lbl.clone()).expect("Error converting training labels");
    let train_labels_one_hot_encoded = one_hot_encode(&train_labels, 10).map(|x| *x as f64);

    let test_data = Array2::from_shape_vec((10_000, image_size), tst_img)
        .expect("Error converting test images")
        .t()
        .map(|x| *x as f64 / 256.0);

    let test_labels =
        Array1::from_shape_vec(10_000, tst_lbl.clone()).expect("Error converting test labels");
    let test_labels_one_hot_encoded = one_hot_encode(&test_labels, 10).map(|x| *x as f64);

    let mut network = builder::NeuralNetworkBuilder::new(image_size)
        .add_layer_random(32, ActivationFunction::Relu)
        .unwrap()
        .add_layer_random(32, ActivationFunction::Relu)
        .unwrap()
        .add_layer_random(10, ActivationFunction::SoftMax)
        .unwrap()
        .build();

    let training_options = TrainingOptions {
        cost_function: CrossEntropyCost,
        batch_size: 64,
        learning_rate: 0.02,
        gradient_magnitude_stopping_criterion: 0.0001,
        cost_difference_stopping_criterion: 0.0001,
        epoch_limit: 100,
    };

    train(
        &mut network,
        &train_data,
        &train_labels_one_hot_encoded,
        &training_options,
    );

    let whole_test_data = MiniBatch {
        inputs: test_data.clone(),
        targets: test_labels_one_hot_encoded.clone(),
    };
    let feed_forward_result = feed_forward(&network, &whole_test_data);
    let prediction_matrix = feed_forward_result.activations.last().unwrap();
    // print_details(&feed_forward_result, &whole_test_data.targets, 20);
    let cost = training_options.cost_function.cost(
        feed_forward_result.activations.last().unwrap(),
        &whole_test_data.targets,
    );
    println!("Cost on the test data: {}", cost);
    let mut predictions = vec![];
    for example in 0..prediction_matrix.dim().1 {
        let mut max = None;
        for row in 0..prediction_matrix.dim().0 {
            if max.is_none() {
                max = Some(row as u8);
                continue;
            }
            if let Some(old_max) = max {
                let current = prediction_matrix[(row, example)];
                if current > prediction_matrix[(old_max as usize, example)] {
                    max = Some(row as u8);
                }
            }
        }
        assert!(max.is_some());
        predictions.push(max.unwrap());
    }
    let mut hit_count = 0;
    let mut miss_count = 0;
    let mut terminal_print_count = 0;
    assert!(predictions.len() == tst_lbl.len());
    for image_index in 0..tst_lbl.len() {
        if predictions[image_index] == tst_lbl[image_index] {
            hit_count += 1;
        } else {
            miss_count += 1;
            _print_image(
                &test_data,
                image_index,
                &format!(
                    "misidentified_images/{image_index}_thought_it_was_a{}.png",
                    predictions[image_index]
                ),
            );
            if terminal_print_count < 1 {
                print_grayscale_image_to_terminal(
                    test_data
                        .column(image_index)
                        .into_iter()
                        .map(|x| (x * 256.0) as u8),
                    28,
                );
                terminal_print_count += 1;
            }
        }
    }
    print!("Hits: {hit_count}, Misses: {miss_count}");
}

pub fn print_grayscale_image_to_terminal<I>(image: I, column_count: usize)
where
    I: IntoIterator<Item = u8>,
{
    for (index, pixel) in image.into_iter().enumerate() {
        let x_coordinate = index % column_count;
        if x_coordinate == column_count {
            println!();
        }
        print!("\x1b[48;2;{0};{0};{0};m  ", pixel);
    }
}

fn _print_image(image_array: &Array2<f64>, image_col: usize, image_file_name: &str) {
    let mut data = Vec::new();
    for i in 0..28 * 28 {
        data.push(image_array[(i, image_col)]);
    }

    let width = 28;
    let height = 28;
    let mut img = GrayImage::new(width, height);

    for (i, &value) in data.iter().enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        img.put_pixel(x, y, Luma([(value * 256.0) as u8]));
    }

    img.save(image_file_name).expect("Failed to save image");
}
