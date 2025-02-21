use image::{GrayImage, Luma};
use mnist::*;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use neural_nobble::activation_functions::SigmoidFunc;
use neural_nobble::{
    cost_functions::*, feed_forward::*, mini_batch::*, neural_network::*, train::*,
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
        Array1::from_shape_vec(50_000, trn_lbl).expect("Error converting training labels");
    let train_labels = one_hot_encode(&train_labels, 10).map(|x| *x as f64 / 256.0);

    let test_data = Array2::from_shape_vec((10_000, image_size), tst_img)
        .expect("Error converting test images")
        .t()
        .map(|x| *x as f64 / 256.0);

    let test_labels =
        Array1::from_shape_vec(10_000, tst_lbl).expect("Error converting test labels");
    let test_labels = one_hot_encode(&test_labels, 10).map(|x| *x as f64 / 256.0);

    let normal_dist = Normal::new(0.5, 1.0).unwrap();
    let mut network = builder::NeuralNetworkBuilder::new(image_size)
        .add_layer(
            Array::random((32, image_size), normal_dist),
            Array::random(32, normal_dist),
        )
        .unwrap()
        .add_layer(
            Array::random((32, 32), normal_dist),
            Array::random(32, normal_dist),
        )
        .unwrap()
        .add_layer(
            Array::random((32, 32), normal_dist),
            Array::random(32, normal_dist),
        )
        .unwrap()
        .add_layer(
            Array::random((32, 32), normal_dist),
            Array::random(32, normal_dist),
        )
        .unwrap()
        .add_layer(
            Array::random((10, 32), normal_dist),
            Array::random(10, normal_dist),
        )
        .unwrap()
        .build();

    let training_options = TrainingOptions {
        cost_function: HalfSSECostFunction,
        batch_size: 100,
        learning_rate: 1000.0,
        gradient_magnitude_stopping_criterion: 0.0001,
        cost_difference_stopping_criterion: 0.0001,
    };

    let activation = SigmoidFunc::default();

    train(
        &mut network,
        activation,
        &train_data,
        &train_labels,
        &training_options,
    );

    let inputs = MiniBatch {
        inputs: test_data.clone(),
        targets: test_labels.clone(),
    };
    let feed_forward_result = feed_forward(&network, activation, &inputs);
    let cost = training_options.cost_function.cost(
        feed_forward_result.activations.last().unwrap(),
        &mini_batch.targets,
    );
}

fn one_hot_encode(row_matrix: &Array1<u8>, limit: u8) -> Array2<u8> {
    let col_count = row_matrix.dim();
    let mut one_hot = Array2::<u8>::zeros((limit as usize, col_count));
    for col in 0..col_count {
        let number = row_matrix[col];
        one_hot[(number as usize, col)] = 1;
    }
    one_hot
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
