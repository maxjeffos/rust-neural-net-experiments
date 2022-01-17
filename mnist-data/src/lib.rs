use common::column_vector;
use common::datapoints::NDTrainingDataPoint;
use mnist::*;
use ndarray::prelude::*;

use common::linalg::ColumnVector;

pub fn get_mnist_data(
    training_set_size: usize,
    test_set_size: usize,
) -> (Vec<NDTrainingDataPoint>, Vec<NDTrainingDataPoint>) {
    // Deconstruct the returned Mnist struct.
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

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!(
        "The first digit is a {:?}",
        train_labels.slice(s![image_num, ..])
    );

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    ////////////////////////////////////
    // Reshape this into stuff for my NN
    ////////////////////////////////////

    // Training Data
    let mut my_training_data = Vec::new();
    for i in 0..training_set_size {
        // image
        let img = train_data.slice(s![i, .., ..]);
        // println!("{:#.3?}\n", img);
        let mut flat_img = Vec::<f32>::new();
        for i_row in 0..28 {
            for i_col in 0..28 {
                let x = img.get((i_row, i_col)).unwrap();
                flat_img.push(*x);
            }
        }
        // println!("flat_img: {:?}", flat_img);
        let flat_img = flat_img.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let flat_image_column_vec = ColumnVector::from_vec(flat_img);

        // label
        let label = train_labels.row(i)[0];
        let label_cv = get_output_v(label);

        let data_point = NDTrainingDataPoint::new(flat_image_column_vec, label_cv);
        my_training_data.push(data_point);
    }

    // Test Data
    let mut my_test_data = Vec::new();
    for i in 0..test_set_size {
        // image
        let img = test_data.slice(s![i, .., ..]);
        // println!("{:#.3?}\n", img);
        let mut flat_img = Vec::<f32>::new();
        for i_row in 0..28 {
            for i_col in 0..28 {
                let x = img.get((i_row, i_col)).unwrap();
                flat_img.push(*x);
            }
        }
        // println!("flat_img: {:?}", flat_img);
        let flat_img = flat_img.iter().map(|x| *x as f64).collect::<Vec<f64>>();
        let flat_image_column_vec = ColumnVector::from_vec(flat_img);

        // label
        let label = test_labels.row(i)[0];
        let label_cv = get_output_v(label);

        let data_point = NDTrainingDataPoint::new(flat_image_column_vec, label_cv);
        my_test_data.push(data_point);
    }

    (my_training_data, my_test_data)
}

fn get_output_v(label: f32) -> ColumnVector {
    match label {
        0.0 => column_vector!(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        1.0 => column_vector!(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2.0 => column_vector!(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        3.0 => column_vector!(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        4.0 => column_vector!(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        5.0 => column_vector!(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        6.0 => column_vector!(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        7.0 => column_vector!(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        8.0 => column_vector!(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        9.0 => column_vector!(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        _ => panic!("unexpected"),
    }
}
