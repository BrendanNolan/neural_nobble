use image::{GrayImage, Luma};
use mnist::*;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use neural_nobble::activation_functions::ReluFunc;
use neural_nobble::activation_functions::SigmoidFunc;
use neural_nobble::{
    cost_functions::*, feed_forward::*, mini_batch::*, neural_network::*, one_hot::*, train::*,
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
        .map(|x| *x as f64 / 32.0);

    let train_labels =
        Array1::from_shape_vec(50_000, trn_lbl.clone()).expect("Error converting training labels");
    let train_labels_one_hot_encoded = one_hot_encode(&train_labels, 10).map(|x| *x as f64);

    let test_data = Array2::from_shape_vec((10_000, image_size), tst_img)
        .expect("Error converting test images")
        .t()
        .map(|x| *x as f64 / 32.0);

    let test_labels =
        Array1::from_shape_vec(10_000, tst_lbl.clone()).expect("Error converting test labels");
    let test_labels_one_hot_encoded = one_hot_encode(&test_labels, 10).map(|x| *x as f64);

    let mut network = builder::NeuralNetworkBuilder::new(image_size)
        .add_layer(
            Array::random(
                (32, image_size),
                Normal::new(0.0, 2.0 / (image_size as f64)).unwrap(),
            ),
            Array::zeros(32),
        )
        .unwrap()
        .add_layer(
            Array::random((32, 32), Normal::new(0.0, 2.0 / 32.0).unwrap()),
            Array::zeros(32),
        )
        .unwrap()
        .add_layer(
            Array::random((10, 32), Normal::new(0.0, 2.0 / 32.0).unwrap()),
            Array::zeros(10),
        )
        .unwrap()
        .build();

    let training_options = TrainingOptions {
        cost_function: HalfSSECostFunction,
        batch_size: 64,
        learning_rate: 0.01,
        gradient_magnitude_stopping_criterion: 0.0001,
        cost_difference_stopping_criterion: 0.0001,
    };

    let activation = ReluFunc::default();

    train(
        &mut network,
        activation,
        &train_data,
        &train_labels_one_hot_encoded,
        &training_options,
    );

    let inputs = MiniBatch {
        inputs: test_data.clone(),
        targets: test_labels_one_hot_encoded.clone(),
    };
    let feed_forward_result = feed_forward(&network, activation, &inputs);
    print_details(&feed_forward_result, &inputs.targets, 20);
    let prediction_matrix = feed_forward_result.activations.last().unwrap();
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
    assert!(predictions.len() == tst_lbl.len());
    for image in 0..tst_lbl.len() {
        if predictions[image] == tst_lbl[image] {
            hit_count += 1;
        } else {
            miss_count += 1;
        }
    }
    print!("Hits: {hit_count}, Misses: {miss_count}");
}

fn print_first_image(image_array: &Array2<f64>, image_size: usize, image_file_name: &str) {
    println!("Rows: {}", image_array.dim().0);
    println!("Columns: {}", image_array.dim().1);
    let mut data = Vec::new();
    for i in 0..image_size {
        data.push(image_array[(i, 0)]);
    }

    let width = 28;
    let height = 28;
    let mut img = GrayImage::new(width, height);

    for (i, &value) in data.iter().enumerate() {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        img.put_pixel(x, y, Luma([value as u8]));
    }

    img.save(image_file_name).expect("Failed to save image");
}
