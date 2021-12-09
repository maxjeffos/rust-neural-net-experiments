use common::matrix::{Matrix, RowsMatrixBuilder};

use common::column_vector;

fn main() {
    let m = column_vector![1.0, 2.0];
    println!("{:?}", m.rows);
    println!("{:?}", m.columns);
    println!("{:?}", m.data);

    let theta = Matrix::new_column_vector(&[0.0, 1.0, 3.0]);
    let x = Matrix::new_column_vector(&[7.0, 11.0, 13.0]);
    let theta_t = theta.transpose();

    assert_eq!(theta_t.rows, 1);
    assert_eq!(theta_t.columns, 3);
    assert_eq!(theta_t.data, vec![0.0, 1.0, 3.0]);

    let y = theta_t.multiply(&x);
    println!("{:?}", y.data);
    println!("{}", y);

    println!("\nmatrix visualization");
    let m = RowsMatrixBuilder::new()
        .with_row(&[1.0, 2.0, 3.0])
        .with_row(&[4.0, 5.0, 6.0])
        .with_row(&[7.0, 8.0, 9.0])
        .build();
    println!("{}", m);
}
