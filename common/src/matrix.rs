use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use rand_distr::Normal;
use std::fmt;

#[macro_export]
macro_rules! column_vector {
    ($($y:expr),+) => (
        Matrix::new_column_vector(&[$($y),+])
    );
}

#[macro_export]
macro_rules! row_matrix {
    ($($y:expr),+) => (
        Matrix::new_row_vector(&[$($y),+])
    );
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn empty_with_num_rows(rows: usize) -> Matrix {
        Matrix {
            rows: rows,
            columns: 0,
            data: vec![],
        }
    }

    pub fn empty_with_num_cols(rows: usize) -> Matrix {
        Matrix {
            rows: 0,
            columns: rows,
            data: vec![],
        }
    }

    pub fn init(rows: usize, columns: usize, init_value: f64) -> Matrix {
        Matrix {
            rows,
            columns,
            data: vec![init_value; rows * columns],
        }
    }

    pub fn new_column_vector(items: &[f64]) -> Self {
        Self {
            rows: items.len(),
            columns: 1,
            data: items.to_vec(),
        }
    }

    pub fn new_row_vector(items: &[f64]) -> Self {
        Self {
            rows: 1,
            columns: items.len(),
            data: items.to_vec(),
        }
    }

    pub fn new_identity_matrix(size: usize) -> Self {
        let mut m = Self::init(size, size, 0.0);
        for i in 0..size {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn new_zero_matrix(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            data: vec![0.0; rows * columns],
        }
    }

    pub fn new_matrix_with_random_values_from_normal_distribution(
        rows: usize,
        columns: usize,
        mean: f64,
        std_dev: f64,
    ) -> Self {
        let mut matrix = Self::new_zero_matrix(rows, columns);

        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();

        for m in 0..rows {
            for n in 0..columns {
                let x = normal.sample(&mut rng);
                matrix.set(m, n, x);
            }
        }

        matrix
    }

    // TODO: implement macros like this:
    // from_columns![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)];
    // from_rows![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)];

    // this isn't great. The macro version could be nicer
    pub fn from_columns(columns: Vec<Vec<f64>>) -> Self {
        // check that the dimension of each inner vec is the same

        if columns.len() == 0 {
            panic!("columns must have at least one column");
        }

        let columns_lengths = columns.iter().map(|v| v.len()).collect::<Vec<usize>>();
        if columns_lengths.iter().any(|&x| x != columns_lengths[0]) {
            panic!("columns must have the same length");
        }

        let rows = columns_lengths[0];

        let mut matrix = Matrix::empty_with_num_rows(rows);
        for c in columns {
            matrix.push_column(&c);
        }

        matrix
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[row * self.columns + column]
    }

    pub fn set(&mut self, row: usize, column: usize, value: f64) {
        self.data[row * self.columns + column] = value;
    }

    pub fn push_row(&mut self, row: &[f64]) {
        self.data.extend_from_slice(row);
    }

    pub fn push_column(&mut self, column: &[f64]) {
        let mut i = 0_usize;
        for item in column {
            i += self.columns;
            self.data.insert(i, *item);
            i += 1
        }

        self.columns += 1;
    }

    pub fn transpose(&self) -> Self {
        if self.columns == 1 && self.rows > 0 {
            Self::new_row_vector(&self.data)
        } else if self.rows == 1 && self.columns > 0 {
            Self::new_column_vector(&self.data)
        } else {
            let mut transposed = Self::new_zero_matrix(self.columns, self.rows);
            for i in 0..self.rows {
                for j in 0..self.columns {
                    transposed.set(j, i, self.get(i, j));
                }
            }

            transposed
        }
    }

    pub fn multiply_by_scalar(&self, scalar: f64) -> Self {
        Self {
            rows: self.rows,
            columns: self.columns,
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }

    pub fn multiply_by_scalar_in_place(&mut self, scalar: f64) {
        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self * scalar;
        }
    }

    pub fn divide_by_scalar(&self, scalar: f64) -> Self {
        Self {
            rows: self.rows,
            columns: self.columns,
            data: self.data.iter().map(|x| x / scalar).collect(),
        }
    }

    pub fn divide_by_scalar_in_place(&mut self, scalar: f64) {
        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self / scalar;
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        if self.columns != other.rows {
            panic!("Matrix dimensions are not compatible for multiplication. The number of columns in self must equal the number of rows in other.");
        }
        let mut result = Self::new_zero_matrix(self.rows, other.columns);
        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Matrix dimensions are not compatible for hadamard product (element-wise multiplication). Both matricies must be the same dimensions.");
        }

        let mut data = Vec::new();
        for i in 0..self.data.len() {
            data.push(self.data[i] * other.data[i]);
        }

        Self {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }

    /// Computes the hadamard product, updating self.
    pub fn hadamard_product_in_place(&mut self, other: &Self) {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Matrix dimensions are not compatible for hadamard product (element-wise multiplication). Both matricies must be the same dimensions.");
        }

        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self * other.data[i];
        }
    }

    pub fn plus(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.columns != other.columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.rows, self.columns, other.rows, other.columns
            );
            panic!("Matrix dimensions are not compatible for addition. Both matricies must have the same dimensions.");
        }

        let mut data = Vec::new();
        for i in 0..self.data.len() {
            data.push(self.data[i] + other.data[i]);
        }

        Self {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }

    /// Add another matrix to self, updating self
    pub fn add_in_place(&mut self, other: &Self) {
        if self.rows != other.rows || self.columns != other.columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.rows, self.columns, other.rows, other.columns
            );
            panic!("Matrix dimensions are not compatible for addition. Both matricies must have the same dimensions.");
        }

        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self + other.data[i];
        }
    }

    /// Create a new Matrix which is self minus other
    pub fn minus(&self, other: &Self) -> Self {
        if self.rows != other.rows || self.columns != other.columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.rows, self.columns, other.rows, other.columns
            );
            panic!("Matrix dimensions are not compatible for subtraction. Both matricies must have the same dimensions.");
        }

        let mut data = Vec::new();
        for i in 0..self.data.len() {
            data.push(self.data[i] - other.data[i]);
        }

        Self {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }

    /// Subtract another matrix from self, updating self.
    pub fn subtract_in_place(&mut self, other: &Self) {
        if self.rows != other.rows || self.columns != other.columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.rows, self.columns, other.rows, other.columns
            );
            panic!("Matrix dimensions are not compatible for subtraction. Both matricies must have the same dimensions.");
        }

        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self - other.data[i];
        }
    }

    pub fn into_value(self) -> f64 {
        if self.rows == 1 && self.columns == 1 {
            let x = self.get(0, 0);
            return x;
        }
        panic!("into_value is not valid for a non-1x1 matrix");
    }

    pub fn vec_length(&self) -> f64 {
        if self.columns == 1 {
            let mut sum = 0.0;
            for i in 0..self.data.len() {
                sum += self.data[i] * self.data[i];
            }
            sum.sqrt()
        } else {
            panic!("vec_length is not valid for a non- column vector");
        }
    }

    // TODO there's probably a better way to impelement this
    pub fn extract_column_vector(&self, column_index: usize) -> Matrix {
        if column_index >= self.columns {
            panic!("column_index must be less than the number of columns");
        }

        let mut data = Vec::new();

        for i_row in 0..self.rows {
            data.push(self.get(i_row, column_index));
        }

        Matrix {
            rows: self.rows,
            columns: 1,
            data,
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut width_pad = String::new();
        let width_per_column = 9;
        let pad_width = width_per_column * self.columns + 1; // +1 for the leading space
        width_pad.push_str(&" ".repeat(pad_width));

        let mut result = String::new();
        result.push_str(&format!("╭{}╮\n", width_pad));

        for i in 0..self.rows {
            result.push_str("│ ");
            for j in 0..self.columns {
                result.push_str(&format!("{:.6} ", self.get(i, j)));
            }
            result.push_str("│");
            result.push_str("\n");
        }

        result.push_str(&format!("╰{}╯\n", width_pad));
        // result.push_str("╰\n");

        write!(f, "{}", result)
    }
}

// struct Shape {
//     rows: usize,
//     columns: usize,
// }

pub struct RowsMatrixBuilder {
    num_columns: Option<usize>,
    rows: Vec<Vec<f64>>,
}

impl RowsMatrixBuilder {
    pub fn new() -> Self {
        Self {
            num_columns: None,
            rows: Vec::new(),
        }
    }

    pub fn new_with_num_columns(num_columns: usize) -> Self {
        Self {
            num_columns: Some(num_columns),
            rows: Vec::new(),
        }
    }

    /// For non-chaining use
    pub fn push_row(&mut self, row: &[f64]) {
        if let Some(num_columns) = self.num_columns {
            if row.len() != num_columns {
                panic!("row must have the same number of columns as previous row or the predefined number of columns");
            }
        } else {
            self.num_columns = Some(row.len());
        }

        let v = Vec::from(row);
        self.rows.push(v);
    }

    /// For chaining use
    pub fn with_row(mut self, row: &[f64]) -> Self {
        if let Some(num_columns) = self.num_columns {
            if row.len() != num_columns {
                panic!("row must have the same number of columns as previous columns or the predefined number of columns");
            }
        } else {
            self.num_columns = Some(row.len());
        }

        let v = Vec::from(row);
        self.rows.push(v);
        self
    }

    pub fn build(self) -> Matrix {
        if let Some(num_columns) = self.num_columns {
            if self.rows.len() == 0 {
                panic!("rows must have at least one row");
            }

            let num_rows = self.rows.len();

            let mut data = Vec::new();
            for row in self.rows {
                data.extend_from_slice(&row);
            }

            Matrix {
                rows: num_rows,
                columns: num_columns,
                data,
            }
        } else {
            panic!("num_columns must be established before building the matrix");
        }
    }
}

pub struct ColumnsMatrixBuilder {
    num_rows: Option<usize>,
    columns: Vec<Vec<f64>>,
}

impl ColumnsMatrixBuilder {
    pub fn new() -> Self {
        Self {
            num_rows: None,
            columns: Vec::new(),
        }
    }

    pub fn push_column(&mut self, column: &[f64]) {
        if let Some(num_rows) = self.num_rows {
            if column.len() != num_rows {
                panic!("column must have the same number of rows as previous columns");
            }
        } else {
            self.num_rows = Some(column.len());
        }

        let v = Vec::from(column);
        self.columns.push(v);
    }

    /// For chaining use
    pub fn with_column(mut self, column: &[f64]) -> Self {
        if let Some(num_rows) = self.num_rows {
            if column.len() != num_rows {
                panic!("column must have the same number of rows as previous columns");
            }
        } else {
            self.num_rows = Some(column.len());
        }

        let v = Vec::from(column);
        self.columns.push(v);
        self
    }

    pub fn build(self) -> Matrix {
        if let Some(num_rows) = self.num_rows {
            if self.columns.len() == 0 {
                panic!("columns must have at least one column");
            }
            let num_columns = self.columns.len();
            let mut data = Vec::new();

            for i_row in 0..num_rows {
                for c in self.columns.iter() {
                    data.push(c[i_row]);
                }
            }

            Matrix {
                rows: num_rows,
                columns: num_columns,
                data,
            }
        } else {
            panic!("num_columns must be established before building the matrix");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get_work() {
        let mut m = Matrix::init(3, 3, 0.0);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);
        m.set(2, 0, 7.0);
        m.set(2, 1, 8.0);
        m.set(2, 2, 9.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(1, 2), 6.0);
        assert_eq!(m.get(2, 0), 7.0);
        assert_eq!(m.get(2, 1), 8.0);
        assert_eq!(m.get(2, 2), 9.0);
    }

    #[test]
    fn new_identity_matrix_works() {
        let m = Matrix::new_identity_matrix(3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.columns, 3);
        assert_eq!(m.data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0);
        assert_eq!(m.get(2, 1), 0.0);
        assert_eq!(m.get(2, 2), 1.0);
    }

    #[test]
    fn can_add_rows_and_get_values_at_specified_indexes() {
        let mut matrix = Matrix::empty_with_num_cols(3);
        matrix.push_row(&[1.0, 2.0, 3.0]);
        matrix.push_row(&[4.0, 5.0, 6.0]);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(0, 2), 3.0);
        assert_eq!(matrix.get(1, 0), 4.0);
        assert_eq!(matrix.get(1, 1), 5.0);
        assert_eq!(matrix.get(1, 2), 6.0);
    }

    #[test]
    fn push_column_works() {
        let mut matrix = Matrix::empty_with_num_rows(3);
        matrix.push_column(&[1.0, 2.0, 3.0]);
        matrix.push_column(&[4.0, 5.0, 6.0]);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 0), 2.0);
        assert_eq!(matrix.get(2, 0), 3.0);

        assert_eq!(matrix.get(0, 1), 4.0);
        assert_eq!(matrix.get(1, 1), 5.0);
        assert_eq!(matrix.get(2, 1), 6.0);
    }

    #[test]
    fn transpose_works_for_column_vector() {
        let theta = Matrix::new_column_vector(&[0.0, 1.0, 3.0]);
        let theta_t = theta.transpose();

        assert_eq!(theta_t.rows, 1);
        assert_eq!(theta_t.columns, 3);
        assert_eq!(theta_t.data, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn transpose_of_column_vector_mult_by_column_vector_works() {
        let theta = Matrix::new_column_vector(&[0.0, 1.0, 3.0]);
        let x = Matrix::new_column_vector(&[7.0, 11.0, 13.0]);
        let theta_t = theta.transpose();

        assert_eq!(theta_t.rows, 1);
        assert_eq!(theta_t.columns, 3);
        assert_eq!(theta_t.data, vec![0.0, 1.0, 3.0]);

        let y = theta_t.multiply(&x);

        assert_eq!(y.rows, 1);
        assert_eq!(y.columns, 1);
        assert_eq!(y.data, vec![50.0]);

        let y_into_value = y.into_value();
        assert_eq!(y_into_value, 50.0);
    }

    #[test]
    fn transpose_works() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m_t = m.transpose();

        assert_eq!(m_t.rows, 3);
        assert_eq!(m_t.columns, 2);
        assert_eq!(m_t.get(0, 0), 1.0);
        assert_eq!(m_t.get(1, 0), 2.0);
        assert_eq!(m_t.get(2, 0), 3.0);

        assert_eq!(m_t.get(0, 1), 4.0);
        assert_eq!(m_t.get(1, 1), 5.0);
        assert_eq!(m_t.get(2, 1), 6.0);
    }

    #[test]
    fn multiply_works() {
        let mut m1 = Matrix::init(2, 3, 0.0);
        m1.set(0, 0, 1.0);
        m1.set(0, 1, 2.0);
        m1.set(0, 2, 3.0);
        m1.set(1, 0, 4.0);
        m1.set(1, 1, 5.0);
        m1.set(1, 2, 6.0);

        let mut m2 = Matrix::init(3, 2, 0.0);
        m2.set(0, 0, 10.0);
        m2.set(0, 1, 11.0);
        m2.set(1, 0, 20.0);
        m2.set(1, 1, 21.0);
        m2.set(2, 0, 30.0);
        m2.set(2, 1, 31.0);

        let m3 = m1.multiply(&m2);

        assert_eq!(m3.rows, 2);
        assert_eq!(m3.columns, 2);
        assert_eq!(m3.get(0, 0), 140.0);
        assert_eq!(m3.get(0, 1), 146.0);
        assert_eq!(m3.get(1, 0), 320.0);
        assert_eq!(m3.get(1, 1), 335.0);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        // 90 degree counterclockwise rotation
        let mut m_rotation = Matrix::empty_with_num_rows(2);
        m_rotation.push_column(&[0.0, 1.0]);
        m_rotation.push_column(&[-1.0, 0.0]);

        let v = Matrix::new_column_vector(&[1.0, 2.0]);
        let v_out = m_rotation.multiply(&v);

        assert_eq!(v_out.rows, 2);
        assert_eq!(v_out.columns, 1);
        assert_eq!(v_out.get(0, 0), -2.0);
        assert_eq!(v_out.get(0, 1), 1.0);

        // shear
        let mut m_shear = Matrix::empty_with_num_rows(2);
        m_shear.push_column(&[1.0, 0.0]);
        m_shear.push_column(&[1.0, 1.0]);

        let v = Matrix::new_column_vector(&[1.0, 2.0]);
        let v_out = m_shear.multiply(&v);
        assert_eq!(v_out.rows, 2);
        assert_eq!(v_out.columns, 1);
        assert_eq!(v_out.get(0, 0), 3.0);
        assert_eq!(v_out.get(0, 1), 2.0);

        let m1 = ColumnsMatrixBuilder::new().with_column(&[1.0, 2.0]).build();

        let m2 = RowsMatrixBuilder::new().with_row(&[3.0, 4.0]).build();

        let m = m1.multiply(&m2);
        println!("m: \n{}", m);

        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 2);
        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }

    #[test]
    fn plus_works() {
        let mut m1 = Matrix::empty_with_num_rows(3);
        m1.push_column(&[1.0, 2.0, 3.0]);
        m1.push_column(&[4.0, 5.0, 6.0]);

        let mut m2 = Matrix::empty_with_num_rows(3);
        m2.push_column(&[1.0, 2.0, 3.0]);
        m2.push_column(&[4.0, 5.0, 6.0]);

        let m = m1.plus(&m2);

        assert_eq!(m.rows, 3);
        assert_eq!(m.columns, 2);

        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 8.0);

        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 1), 10.0);

        assert_eq!(m.get(2, 0), 6.0);
        assert_eq!(m.get(2, 1), 12.0);
    }

    #[test]
    fn add_in_place_works() {
        let mut m1 = Matrix::empty_with_num_rows(3);
        m1.push_column(&[1.0, 2.0, 3.0]);
        m1.push_column(&[4.0, 5.0, 6.0]);

        let mut m2 = Matrix::empty_with_num_rows(3);
        m2.push_column(&[1.0, 2.0, 3.0]);
        m2.push_column(&[4.0, 5.0, 6.0]);

        m1.add_in_place(&m2);

        assert_eq!(m1.rows, 3);
        assert_eq!(m1.columns, 2);

        assert_eq!(m1.get(0, 0), 2.0);
        assert_eq!(m1.get(0, 1), 8.0);

        assert_eq!(m1.get(1, 0), 4.0);
        assert_eq!(m1.get(1, 1), 10.0);

        assert_eq!(m1.get(2, 0), 6.0);
        assert_eq!(m1.get(2, 1), 12.0);
    }

    #[test]
    fn minus_works() {
        let mut m1 = column_vector![1.0, 2.0, 3.0];
        let mut m2 = column_vector![1.0, 2.0, 3.0];

        let m = m1.minus(&m2);

        assert_eq!(m.rows, 3);
        assert_eq!(m.columns, 1);
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
        assert_eq!(m.get(2, 0), 0.0);
    }

    #[test]
    fn subtract_works() {
        let mut m1 = column_vector![1.0, 2.0, 3.0];
        let m2 = column_vector![1.0, 2.0, 3.0];

        m1.subtract_in_place(&m2);

        assert_eq!(m1.rows, 3);
        assert_eq!(m1.columns, 1);
        assert_eq!(m1.get(0, 0), 0.0);
        assert_eq!(m1.get(1, 0), 0.0);
        assert_eq!(m1.get(2, 0), 0.0);
    }

    #[test]
    fn test_from_columns() {
        let m_build_from_columns = Matrix::from_columns(vec![vec![0.0, 1.0], vec![-1.0, 0.0]]);

        println!("{:?}", m_build_from_columns.data);
        println!("rows: {:?}", m_build_from_columns.rows);
        println!("columns: {:?}", m_build_from_columns.columns);

        assert_eq!(m_build_from_columns.rows, 2);
        assert_eq!(m_build_from_columns.columns, 2);

        assert_eq!(m_build_from_columns.get(0, 0), 0.0);
        assert_eq!(m_build_from_columns.get(0, 1), -1.0);
        assert_eq!(m_build_from_columns.get(1, 0), 1.0);
        assert_eq!(m_build_from_columns.get(1, 1), 0.0);
    }

    #[test]
    fn test_hadamard_product() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m2 = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m_hadamard = m.hadamard_product(&m2);
        assert_eq!(m_hadamard.rows, 2);
        assert_eq!(m_hadamard.columns, 3);
        assert_eq!(m_hadamard.get(0, 0), 1.0);
        assert_eq!(m_hadamard.get(0, 1), 4.0);
        assert_eq!(m_hadamard.get(0, 2), 9.0);
        assert_eq!(m_hadamard.get(1, 0), 16.0);
        assert_eq!(m_hadamard.get(1, 1), 25.0);
        assert_eq!(m_hadamard.get(1, 2), 36.0);
    }

    #[test]
    fn test_hadamard_product_in_place() {
        let mut m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m2 = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        m.hadamard_product_in_place(&m2);
        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(0, 2), 9.0);
        assert_eq!(m.get(1, 0), 16.0);
        assert_eq!(m.get(1, 1), 25.0);
        assert_eq!(m.get(1, 2), 36.0);
    }

    #[test]
    fn test_vec_length() {
        assert_eq!((column_vector![0.0]).vec_length(), 0.0);
        assert_eq!((column_vector![1.0]).vec_length(), 1.0);
        assert_eq!((column_vector![4.0]).vec_length(), 4.0);
        assert_eq!((column_vector![3.0, 4.0]).vec_length(), 5.0);
    }

    #[test]
    fn test_extract_column_vector() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .with_row(&[7.0, 8.0, 9.0])
            .build();

        let v0 = m.extract_column_vector(0);
        assert_eq!(v0.rows, 3);
        assert_eq!(v0.columns, 1);
        assert_eq!(v0.get(0, 0), 1.0);
        assert_eq!(v0.get(1, 0), 4.0);
        assert_eq!(v0.get(2, 0), 7.0);

        let v1 = m.extract_column_vector(1);
        assert_eq!(v1.rows, 3);
        assert_eq!(v1.columns, 1);
        assert_eq!(v1.get(0, 0), 2.0);
        assert_eq!(v1.get(1, 0), 5.0);
        assert_eq!(v1.get(2, 0), 8.0);

        let v2 = m.extract_column_vector(2);
        assert_eq!(v2.rows, 3);
        assert_eq!(v2.columns, 1);
        assert_eq!(v2.get(0, 0), 3.0);
        assert_eq!(v2.get(1, 0), 6.0);
        assert_eq!(v2.get(2, 0), 9.0);
    }

    #[test]
    pub fn test_multiply_by_scalar() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m2 = m.multiply_by_scalar(2.0);
        assert_eq!(m2.rows, 2);
        assert_eq!(m2.columns, 3);

        assert_eq!(m2.get(0, 0), 2.0);
        assert_eq!(m2.get(0, 1), 4.0);
        assert_eq!(m2.get(0, 2), 6.0);

        assert_eq!(m2.get(1, 0), 8.0);
        assert_eq!(m2.get(1, 1), 10.0);
        assert_eq!(m2.get(1, 2), 12.0);
    }

    #[test]
    pub fn test_multiply_by_scalar_in_place() {
        let mut m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        m.multiply_by_scalar_in_place(2.0);
        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 3);

        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(0, 2), 6.0);

        assert_eq!(m.get(1, 0), 8.0);
        assert_eq!(m.get(1, 1), 10.0);
        assert_eq!(m.get(1, 2), 12.0);
    }

    #[test]
    pub fn test_divide_by_scalar() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m2 = m.divide_by_scalar(2.0);
        assert_eq!(m2.rows, 2);
        assert_eq!(m2.columns, 3);

        assert_eq!(m2.get(0, 0), 0.5);
        assert_eq!(m2.get(0, 1), 1.0);
        assert_eq!(m2.get(0, 2), 1.5);

        assert_eq!(m2.get(1, 0), 2.0);
        assert_eq!(m2.get(1, 1), 2.5);
        assert_eq!(m2.get(1, 2), 3.0);
    }

    #[test]
    pub fn test_divide_by_scalar_in_place() {
        let mut m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        m.divide_by_scalar_in_place(2.0);
        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 3);

        assert_eq!(m.get(0, 0), 0.5);
        assert_eq!(m.get(0, 1), 1.0);
        assert_eq!(m.get(0, 2), 1.5);

        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(1, 1), 2.5);
        assert_eq!(m.get(1, 2), 3.0);
    }
}

#[cfg(test)]
mod rows_matrix_builder_tests {
    use super::*;

    #[test]
    fn test_row_matrix_builder_with_non_chaining() {
        let mut rmb = RowsMatrixBuilder::new();
        rmb.push_row(&[1.0, 2.0, 3.0]);
        rmb.push_row(&[4.0, 5.0, 6.0]);
        let m = rmb.build();

        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn test_rows_matrix_builder_with_chaining() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(1, 2), 6.0);
    }
}

#[cfg(test)]
mod columns_matrix_builder_tests {
    use super::*;

    #[test]
    fn test_columns_matrix_builder_with_non_chaining() {
        let mut cmb = ColumnsMatrixBuilder::new();
        cmb.push_column(&[1.0, 2.0, 3.0]);
        cmb.push_column(&[4.0, 5.0, 6.0]);
        let m = cmb.build();

        assert_eq!(m.rows, 3);
        assert_eq!(m.columns, 2);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(2, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(2, 1), 6.0);
    }

    #[test]
    fn test_columns_matrix_builder_with_chaining() {
        let m = ColumnsMatrixBuilder::new()
            .with_column(&[1.0, 2.0, 3.0])
            .with_column(&[4.0, 5.0, 6.0])
            .build();

        assert_eq!(m.rows, 3);
        assert_eq!(m.columns, 2);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(2, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(2, 1), 6.0);
    }
}
