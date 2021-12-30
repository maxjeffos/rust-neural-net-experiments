use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use rand_distr::Normal;
use std::fmt;
use std::ops::Deref;
use std::ops::DerefMut;

fn square_ref(x: &f64) -> f64 {
    x * x
}

#[macro_export]
macro_rules! column_vector_matrix {
    ($($y:expr),+) => (
        Matrix::new_column_vector(&[$($y),+])
    );
}

#[macro_export]
macro_rules! column_vector {
    ($($y:expr),+) => (
        ColumnVector::new(&[$($y),+])
    );
}

#[derive(Debug, Clone)]
pub struct Matrix {
    num_rows: usize,
    num_columns: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub fn empty_with_num_rows(num_rows: usize) -> Matrix {
        Matrix {
            num_rows,
            num_columns: 0,
            data: vec![],
        }
    }

    pub fn empty_with_num_cols(num_columns: usize) -> Matrix {
        Matrix {
            num_rows: 0,
            num_columns,
            data: vec![],
        }
    }

    pub fn init(num_rows: usize, num_columns: usize, init_value: f64) -> Matrix {
        Matrix {
            num_rows: num_rows,
            num_columns: num_columns,
            data: vec![init_value; num_rows * num_columns],
        }
    }

    pub fn new_column_vector(items: &[f64]) -> Self {
        Self {
            num_rows: items.len(),
            num_columns: 1,
            data: items.to_vec(),
        }
    }

    pub fn new_row_vector(items: &[f64]) -> Self {
        Self {
            num_rows: 1,
            num_columns: items.len(),
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

    pub fn new_zero_matrix(num_rows: usize, num_columns: usize) -> Self {
        Self {
            num_rows,
            num_columns,
            data: vec![0.0; num_rows * num_columns],
        }
    }

    pub fn new_matrix_with_random_values_from_normal_distribution(
        num_rows: usize,
        num_columns: usize,
        mean: f64,
        std_dev: f64,
    ) -> Self {
        let mut matrix = Self::new_zero_matrix(num_rows, num_columns);

        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();

        for m in 0..num_rows {
            for n in 0..num_columns {
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

        let columns_lengths = columns
            .iter()
            .map(|v| v.len())
            .collect::<Vec<usize>>();
        if columns_lengths
            .iter()
            .any(|&x| x != columns_lengths[0])
        {
            panic!("columns must have the same length");
        }

        let num_rows = columns_lengths[0];

        let mut matrix = Matrix::empty_with_num_rows(num_rows);
        for c in columns {
            matrix.push_column(&c);
        }

        matrix
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.data[row * self.num_columns + column]
    }

    pub fn set(&mut self, row: usize, column: usize, value: f64) {
        self.data[row * self.num_columns + column] = value;
    }

    pub fn push_row(&mut self, row: &[f64]) {
        self.data.extend_from_slice(row);
    }

    pub fn push_column(&mut self, column: &[f64]) {
        let mut i = 0_usize;
        for item in column {
            i += self.num_columns;
            self.data.insert(i, *item);
            i += 1
        }

        self.num_columns += 1;
    }

    pub fn transpose(&self) -> Self {
        if self.num_columns == 1 && self.num_rows > 0 {
            Self::new_row_vector(&self.data)
        } else if self.num_rows == 1 && self.num_columns > 0 {
            Self::new_column_vector(&self.data)
        } else {
            let mut transposed = Self::new_zero_matrix(self.num_columns, self.num_rows);
            for i in 0..self.num_rows {
                for j in 0..self.num_columns {
                    transposed.set(j, i, self.get(i, j));
                }
            }

            transposed
        }
    }

    pub fn multiply_by_scalar(&self, scalar: f64) -> Self {
        Self {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            data: self
                .data
                .iter()
                .map(|x| x * scalar)
                .collect(),
        }
    }

    pub fn multiply_by_scalar_in_place(&mut self, scalar: f64) {
        self.data
            .iter_mut()
            .for_each(|x| *x *= scalar);
    }

    pub fn divide_by_scalar(&self, scalar: f64) -> Self {
        Self {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            data: self
                .data
                .iter()
                .map(|x| x / scalar)
                .collect(),
        }
    }

    pub fn divide_by_scalar_in_place(&mut self, scalar: f64) {
        self.data
            .iter_mut()
            .for_each(|x| *x /= scalar);
    }

    pub fn multiply(&self, other: &Self) -> Self {
        if self.num_columns != other.num_rows {
            panic!("Matrix dimensions are not compatible for multiplication. The number of columns in self must equal the number of rows in other.");
        }
        let mut result = Self::new_zero_matrix(self.num_rows, other.num_columns);
        for i in 0..self.num_rows {
            for j in 0..other.num_columns {
                let mut sum = 0.0;
                for k in 0..self.num_columns {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn mult_vector(&self, v: &ColumnVector) -> ColumnVector {
        if self.num_columns != v.num_elements() {
            panic!("Matrix dimensions are not compatible for multiplication. The number of columns in self must equal the number of elements (rows) in v.");
        }
        let mut res = ColumnVector::empty();
        for i in 0..self.num_rows {
            let mut sum = 0.0;
            for k in 0..self.num_columns {
                sum += self.get(i, k) * v.get(k);
            }
            res.push(sum);
        }
        res
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        if self.num_rows != other.num_rows || self.num_columns != other.num_columns {
            panic!("Matrix dimensions are not compatible for hadamard product (element-wise multiplication). Both matricies must be the same dimensions.");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x * y)
            .collect();

        Self {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            data,
        }
    }

    /// Computes the hadamard product, updating self.
    pub fn hadamard_product_in_place(&mut self, other: &Self) {
        if self.num_rows != other.num_rows || self.num_columns != other.num_columns {
            panic!("Matrix dimensions are not compatible for hadamard product (element-wise multiplication). Both matricies must be the same dimensions.");
        }

        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self * other.data[i];
        }
    }

    pub fn plus(&self, other: &Self) -> Self {
        if self.num_rows != other.num_rows || self.num_columns != other.num_columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.num_rows, self.num_columns, other.num_rows, other.num_columns
            );
            panic!("Matrix dimensions are not compatible for addition. Both matricies must have the same dimensions.");
        }

        let mut data = Vec::new();
        for i in 0..self.data.len() {
            data.push(self.data[i] + other.data[i]);
        }

        Self {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            data,
        }
    }

    /// Add another matrix to self, updating self
    pub fn add_in_place(&mut self, other: &Self) {
        if self.num_rows != other.num_rows || self.num_columns != other.num_columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.num_rows, self.num_columns, other.num_rows, other.num_columns
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
        if self.num_rows != other.num_rows || self.num_columns != other.num_columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.num_rows, self.num_columns, other.num_rows, other.num_columns
            );
            panic!("Matrix dimensions are not compatible for subtraction. Both matricies must have the same dimensions.");
        }

        let mut data = Vec::new();
        for i in 0..self.data.len() {
            data.push(self.data[i] - other.data[i]);
        }

        Self {
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            data,
        }
    }

    /// Subtract another matrix from self, updating self.
    pub fn subtract_in_place(&mut self, other: &Self) {
        if self.num_rows != other.num_rows || self.num_columns != other.num_columns {
            println!(
                "self: {}x{} other: {}x{}",
                self.num_rows, self.num_columns, other.num_rows, other.num_columns
            );
            panic!("Matrix dimensions are not compatible for subtraction. Both matricies must have the same dimensions.");
        }

        for i in 0..self.data.len() {
            let val_in_self = self.data.get_mut(i).unwrap();
            *val_in_self = *val_in_self - other.data[i];
        }
    }

    pub fn into_value(self) -> f64 {
        if self.num_rows == 1 && self.num_columns == 1 {
            let x = self.get(0, 0);
            return x;
        }
        panic!("into_value is not valid for a non-1x1 matrix");
    }

    pub fn vec_length(&self) -> f64 {
        if self.num_columns == 1 {
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
    pub fn extract_column_vector_as_matrix(&self, column_index: usize) -> Matrix {
        if column_index >= self.num_columns {
            panic!("column_index must be less than the number of columns");
        }

        let mut data = Vec::new();

        for i_row in 0..self.num_rows {
            data.push(self.get(i_row, column_index));
        }

        Matrix {
            num_rows: self.num_rows,
            num_columns: 1,
            data,
        }
    }

    pub fn extract_column_vector(&self, column_index: usize) -> ColumnVector {
        if column_index >= self.num_columns {
            panic!("column_index must be less than the number of columns");
        }

        let mut res = ColumnVector::empty();
        for i_row in 0..self.num_rows {
            res.push(self.get(i_row, column_index));
        }

        res
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut width_pad = String::new();
        let width_per_column = 9;
        let pad_width = width_per_column * self.num_columns + 1; // +1 for the leading space
        width_pad.push_str(&" ".repeat(pad_width));

        let mut result = String::new();
        result.push_str(&format!("╭{}╮\n", width_pad));

        for i in 0..self.num_rows {
            result.push_str("│ ");
            for j in 0..self.num_columns {
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

impl From<ColumnVector> for Matrix {
    fn from(column_vector: ColumnVector) -> Matrix {
        column_vector.inner_matrix
    }
}

#[derive(Debug, Clone)]
pub struct ColumnVector {
    inner_matrix: Matrix,
}

pub struct IterWith<'a> {
    v_0_values: &'a Vec<f64>,
    v_1_values: &'a Vec<f64>,
    index: usize,
}

impl<'a> Iterator for IterWith<'a> {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.v_0_values.len() {
            let x_0 = self.v_0_values[self.index];
            let x_1 = self.v_1_values[self.index];
            self.index += 1;
            Some((x_0, x_1))
        } else {
            None
        }
    }
}

impl ColumnVector {
    pub fn new(data: &[f64]) -> ColumnVector {
        ColumnVector {
            inner_matrix: Matrix::new_column_vector(data),
        }
    }

    pub fn new_zero_vector(num_rows: usize) -> ColumnVector {
        ColumnVector {
            inner_matrix: Matrix::new_zero_matrix(num_rows, 1),
        }
    }

    pub fn fill_new(value: f64, size: usize) -> Self {
        let mut cv = ColumnVector::empty();
        for _ in 0..size {
            cv.push(value);
        }
        cv
    }

    pub fn empty() -> Self {
        Self {
            inner_matrix: Matrix::new_column_vector(&[]),
        }
    }

    pub fn num_elements(&self) -> usize {
        self.inner_matrix.num_rows
    }

    pub fn get(&self, row_index: usize) -> f64 {
        self.inner_matrix.get(row_index, 0)
    }

    pub fn set(&mut self, row_index: usize, value: f64) {
        self.inner_matrix.set(row_index, 0, value);
    }

    pub fn iter_with<'a>(&'a self, other: &'a ColumnVector) -> IterWith {
        if self.inner_matrix.data.len() != other.inner_matrix.data.len() {
            panic!("self and other must have the same length");
        }

        IterWith {
            v_0_values: &self.inner_matrix.data,
            v_1_values: &other.inner_matrix.data,
            index: 0,
        }
    }

    pub fn plus(&self, other: &ColumnVector) -> ColumnVector {
        let mut data = Vec::new();
        for i in 0..self.num_elements() {
            data.push(self.get(i) + other.get(i));
        }
        ColumnVector::new(&data)
    }

    pub fn plus_in_place(&mut self, other: &ColumnVector) {
        for i in 0..self.num_elements() {
            self.set(i, self.get(i) + other.get(i));
        }
    }

    /// Variant of addition for use in chaining
    pub fn add(mut self, other: &ColumnVector) -> ColumnVector {
        for i in 0..self.num_elements() {
            self.set(i, self.get(i) + other.get(i));
        }
        self
    }

    pub fn minus(&self, other: &ColumnVector) -> ColumnVector {
        let mut data = Vec::new();
        for i in 0..self.num_elements() {
            data.push(self.get(i) - other.get(i));
        }
        ColumnVector::new(&data)
    }

    pub fn minus_in_place(&mut self, other: &ColumnVector) {
        for i in 0..self.num_elements() {
            self.set(i, self.get(i) - other.get(i));
        }
    }

    pub fn multiply_by_scalar(&self, scalar: f64) -> ColumnVector {
        let data = self
            .inner_matrix
            .data
            .iter()
            .map(|x| *x * scalar)
            .collect::<Vec<f64>>();
        ColumnVector::new(&data)
    }

    pub fn multiply_by_scalar_in_place(&mut self, scalar: f64) {
        self.inner_matrix
            .data
            .iter_mut()
            .for_each(|x| *x *= scalar);
    }

    pub fn divide_by_scalar(&self, scalar: f64) -> ColumnVector {
        let data = self
            .inner_matrix
            .data
            .iter()
            .map(|x| *x / scalar)
            .collect::<Vec<f64>>();

        ColumnVector::new(&data)
    }

    pub fn divide_by_scalar_in_place(&mut self, scalar: f64) {
        self.inner_matrix
            .data
            .iter_mut()
            .for_each(|x| *x /= scalar);
    }

    pub fn mult_matrix(&self, other: &Matrix) -> Matrix {
        // TODO: re-implement without cloning
        let self_clone = self.clone();
        let m: Matrix = self_clone.into();
        let res = m.multiply(&other);
        res
    }

    pub fn dot_product(&self, other: &ColumnVector) -> f64 {
        if self.num_elements() != other.num_elements() {
            panic!("dot_product requires two vectors of the same length");
        }

        let mut sum = 0.0;
        for (x, y) in self.iter_with(other) {
            sum += x * y;
        }
        sum
    }

    pub fn hadamard_product(&self, other: &ColumnVector) -> ColumnVector {
        if self.num_elements() != other.num_elements() {
            panic!("hadamard_product on column vectors requires that the two vectors have of the same length");
        }

        let mut data = Vec::new();
        for (x, y) in self.iter_with(other) {
            data.push(x * y);
        }
        ColumnVector::new(&data)
    }

    pub fn hadamard_product_in_place(&mut self, other: &ColumnVector) {
        if self.num_elements() != other.num_elements() {
            panic!("hadamard_product on column vectors requires that the two vectors have of the same length");
        }

        for i in 0..self.num_elements() {
            self.set(i, self.get(i) * other.get(i));
        }
    }

    pub fn vec_length(&self) -> f64 {
        self.iter()
            .map(square_ref)
            .sum::<f64>()
            .sqrt()
    }

    pub fn push(&mut self, value: f64) {
        self.inner_matrix.data.push(value);
        self.inner_matrix.num_rows += 1;
    }

    pub fn transpose(&self) -> Matrix {
        self.inner_matrix.transpose()
    }

    pub fn into_value(self) -> f64 {
        if self.num_elements() != 1 {
            panic!("into_value is only valid for a column vector with one element");
        }
        self.get(0)
    }
}

impl From<Matrix> for ColumnVector {
    fn from(matrix: Matrix) -> ColumnVector {
        if matrix.num_columns != 1 {
            panic!(
                "Cannot convert a {}x{} matrix into a column vector",
                matrix.num_rows, matrix.num_columns
            );
        }

        ColumnVector {
            inner_matrix: matrix,
        }
    }
}

impl fmt::Display for ColumnVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut width_pad = String::new();
        let width_per_column = 9;
        let pad_width = width_per_column + 1; // +1 for the leading space
        width_pad.push_str(&" ".repeat(pad_width));

        let mut result = String::new();
        result.push_str(&format!("╭{}╮\n", width_pad));

        for i in 0..self.inner_matrix.num_rows {
            result.push_str("│ ");
            for j in 0..self.inner_matrix.num_columns {
                result.push_str(&format!("{:.6} ", self.inner_matrix.get(i, j)));
            }
            result.push_str("│");
            result.push_str("\n");
        }

        result.push_str(&format!("╰{}╯\n", width_pad));
        // result.push_str("╰\n");

        write!(f, "{}", result)
    }
}

impl Deref for ColumnVector {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.inner_matrix.data
    }
}

impl DerefMut for ColumnVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner_matrix.data
    }
}

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
                num_rows,
                num_columns,
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
                num_rows,
                num_columns,
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
        assert_eq!(m.num_rows, 3);
        assert_eq!(m.num_columns, 3);
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

        assert_eq!(theta_t.num_rows, 1);
        assert_eq!(theta_t.num_columns, 3);
        assert_eq!(theta_t.data, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn transpose_of_column_vector_mult_by_column_vector_works() {
        let theta = Matrix::new_column_vector(&[0.0, 1.0, 3.0]);
        let x = Matrix::new_column_vector(&[7.0, 11.0, 13.0]);
        let theta_t = theta.transpose();

        assert_eq!(theta_t.num_rows, 1);
        assert_eq!(theta_t.num_columns, 3);
        assert_eq!(theta_t.data, vec![0.0, 1.0, 3.0]);

        let y = theta_t.multiply(&x);

        assert_eq!(y.num_rows, 1);
        assert_eq!(y.num_columns, 1);
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

        assert_eq!(m_t.num_rows, 3);
        assert_eq!(m_t.num_columns, 2);
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

        assert_eq!(m3.num_rows, 2);
        assert_eq!(m3.num_columns, 2);
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

        assert_eq!(v_out.num_rows, 2);
        assert_eq!(v_out.num_columns, 1);
        assert_eq!(v_out.get(0, 0), -2.0);
        assert_eq!(v_out.get(1, 0), 1.0); // TODO: I had this backwards and it still worked - was something wrong with get(). probably need to add bounds checking on the indexes

        // shear
        let mut m_shear = Matrix::empty_with_num_rows(2);
        m_shear.push_column(&[1.0, 0.0]);
        m_shear.push_column(&[1.0, 1.0]);

        let v = Matrix::new_column_vector(&[1.0, 2.0]);
        let v_out = m_shear.multiply(&v);
        assert_eq!(v_out.num_rows, 2);
        assert_eq!(v_out.num_columns, 1);
        assert_eq!(v_out.get(0, 0), 3.0);
        assert_eq!(v_out.get(0, 1), 2.0);

        let m1 = ColumnsMatrixBuilder::new()
            .with_column(&[1.0, 2.0])
            .build();

        let m2 = RowsMatrixBuilder::new()
            .with_row(&[3.0, 4.0])
            .build();

        let m = m1.multiply(&m2);
        println!("m: \n{}", m);

        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 2);
        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }

    #[test]
    fn test_matrix_vector_multiplication_with_column_vector_type() {
        // 90 degree counterclockwise rotation
        let mut m_rotation = Matrix::empty_with_num_rows(2);
        m_rotation.push_column(&[0.0, 1.0]);
        m_rotation.push_column(&[-1.0, 0.0]);

        let v = ColumnVector::new(&[1.0, 2.0]);
        let v_out = m_rotation.mult_vector(&v);
        assert_eq!(v_out.num_elements(), 2);
        assert_eq!(v_out.get(0), -2.0);
        assert_eq!(v_out.get(1), 1.0);

        // shear
        let mut m_shear = Matrix::empty_with_num_rows(2);
        m_shear.push_column(&[1.0, 0.0]);
        m_shear.push_column(&[1.0, 1.0]);

        let v = ColumnVector::new(&[1.0, 2.0]);
        let v_out = m_shear.mult_vector(&v);
        assert_eq!(v_out.num_elements(), 2);
        assert_eq!(v_out.get(0), 3.0);
        assert_eq!(v_out.get(1), 2.0);
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

        assert_eq!(m.num_rows, 3);
        assert_eq!(m.num_columns, 2);

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

        assert_eq!(m1.num_rows, 3);
        assert_eq!(m1.num_columns, 2);

        assert_eq!(m1.get(0, 0), 2.0);
        assert_eq!(m1.get(0, 1), 8.0);

        assert_eq!(m1.get(1, 0), 4.0);
        assert_eq!(m1.get(1, 1), 10.0);

        assert_eq!(m1.get(2, 0), 6.0);
        assert_eq!(m1.get(2, 1), 12.0);
    }

    #[test]
    fn minus_works() {
        let mut m1 = column_vector_matrix![1.0, 2.0, 3.0];
        let mut m2 = column_vector_matrix![1.0, 2.0, 3.0];

        let m = m1.minus(&m2);

        assert_eq!(m.num_rows, 3);
        assert_eq!(m.num_columns, 1);
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
        assert_eq!(m.get(2, 0), 0.0);
    }

    #[test]
    fn subtract_works() {
        let mut m1 = column_vector_matrix![1.0, 2.0, 3.0];
        let m2 = column_vector_matrix![1.0, 2.0, 3.0];

        m1.subtract_in_place(&m2);

        assert_eq!(m1.num_rows, 3);
        assert_eq!(m1.num_columns, 1);
        assert_eq!(m1.get(0, 0), 0.0);
        assert_eq!(m1.get(1, 0), 0.0);
        assert_eq!(m1.get(2, 0), 0.0);
    }

    #[test]
    fn test_from_columns() {
        let m_build_from_columns = Matrix::from_columns(vec![vec![0.0, 1.0], vec![-1.0, 0.0]]);

        println!("{:?}", m_build_from_columns.data);
        println!("num_rows: {:?}", m_build_from_columns.num_rows);
        println!("num_columns: {:?}", m_build_from_columns.num_columns);

        assert_eq!(m_build_from_columns.num_rows, 2);
        assert_eq!(m_build_from_columns.num_columns, 2);

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
        assert_eq!(m_hadamard.num_rows, 2);
        assert_eq!(m_hadamard.num_columns, 3);
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
        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(0, 2), 9.0);
        assert_eq!(m.get(1, 0), 16.0);
        assert_eq!(m.get(1, 1), 25.0);
        assert_eq!(m.get(1, 2), 36.0);
    }

    #[test]
    fn test_vec_length() {
        assert_eq!((column_vector_matrix![0.0]).vec_length(), 0.0);
        assert_eq!((column_vector_matrix![1.0]).vec_length(), 1.0);
        assert_eq!((column_vector_matrix![4.0]).vec_length(), 4.0);
        assert_eq!((column_vector_matrix![3.0, 4.0]).vec_length(), 5.0);
    }

    #[test]
    fn test_extract_column_vector_as_matrix() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .with_row(&[7.0, 8.0, 9.0])
            .build();

        let v0 = m.extract_column_vector_as_matrix(0);
        assert_eq!(v0.num_rows, 3);
        assert_eq!(v0.num_columns, 1);
        assert_eq!(v0.get(0, 0), 1.0);
        assert_eq!(v0.get(1, 0), 4.0);
        assert_eq!(v0.get(2, 0), 7.0);

        let v1 = m.extract_column_vector_as_matrix(1);
        assert_eq!(v1.num_rows, 3);
        assert_eq!(v1.num_columns, 1);
        assert_eq!(v1.get(0, 0), 2.0);
        assert_eq!(v1.get(1, 0), 5.0);
        assert_eq!(v1.get(2, 0), 8.0);

        let v2 = m.extract_column_vector_as_matrix(2);
        assert_eq!(v2.num_rows, 3);
        assert_eq!(v2.num_columns, 1);
        assert_eq!(v2.get(0, 0), 3.0);
        assert_eq!(v2.get(1, 0), 6.0);
        assert_eq!(v2.get(2, 0), 9.0);
    }

    #[test]
    fn test_extract_column_vector() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .with_row(&[7.0, 8.0, 9.0])
            .build();

        let v0 = m.extract_column_vector(0);
        assert_eq!(v0.num_elements(), 3);
        assert_eq!(v0.get(0), 1.0);
        assert_eq!(v0.get(1), 4.0);
        assert_eq!(v0.get(2), 7.0);

        let v1 = m.extract_column_vector(1);
        assert_eq!(v1.num_elements(), 3);
        assert_eq!(v1.get(0), 2.0);
        assert_eq!(v1.get(1), 5.0);
        assert_eq!(v1.get(2), 8.0);

        let v2 = m.extract_column_vector(2);
        assert_eq!(v2.num_elements(), 3);
        assert_eq!(v2.get(0), 3.0);
        assert_eq!(v2.get(1), 6.0);
        assert_eq!(v2.get(2), 9.0);
    }

    #[test]
    pub fn test_multiply_by_scalar() {
        let m = RowsMatrixBuilder::new()
            .with_row(&[1.0, 2.0, 3.0])
            .with_row(&[4.0, 5.0, 6.0])
            .build();

        let m2 = m.multiply_by_scalar(2.0);
        assert_eq!(m2.num_rows, 2);
        assert_eq!(m2.num_columns, 3);

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
        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 3);

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
        assert_eq!(m2.num_rows, 2);
        assert_eq!(m2.num_columns, 3);

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
        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 3);

        assert_eq!(m.get(0, 0), 0.5);
        assert_eq!(m.get(0, 1), 1.0);
        assert_eq!(m.get(0, 2), 1.5);

        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(1, 1), 2.5);
        assert_eq!(m.get(1, 2), 3.0);
    }
}

#[cfg(test)]
mod column_vector_tests {
    use super::*;

    #[test]
    pub fn test_basics() {
        let mut cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        assert_eq!(cv.num_elements(), 3);
        assert_eq!(cv.get(0), 1.0);
        assert_eq!(cv.get(1), 2.0);
        assert_eq!(cv.get(2), 3.0);

        cv.set(0, 4.0);
        cv.set(1, 5.0);
        cv.set(2, 6.0);
        assert_eq!(cv.get(0), 4.0);
        assert_eq!(cv.get(1), 5.0);
        assert_eq!(cv.get(2), 6.0);
    }

    #[test]
    pub fn new_zero_vector_works() {
        let cv = ColumnVector::new_zero_vector(3);
        assert_eq!(cv.num_elements(), 3);
        assert_eq!(cv.get(0), 0.0);
        assert_eq!(cv.get(1), 0.0);
        assert_eq!(cv.get(2), 0.0);
    }

    #[test]
    pub fn can_create_a_column_vector_and_use_from_and_into() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cvm: Matrix = cv.into();
        assert_eq!(cvm.num_rows, 3);
        assert_eq!(cvm.num_columns, 1);
        assert_eq!(cvm.get(0, 0), 1.0);
        assert_eq!(cvm.get(1, 0), 2.0);
        assert_eq!(cvm.get(2, 0), 3.0);

        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cvm = Matrix::from(cv);
        assert_eq!(cvm.num_rows, 3);
        assert_eq!(cvm.num_columns, 1);
        assert_eq!(cvm.get(0, 0), 1.0);
        assert_eq!(cvm.get(1, 0), 2.0);
        assert_eq!(cvm.get(2, 0), 3.0);

        // check that one can clone a column vector
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = cv.clone();
        let cvm: Matrix = cv.into();
        assert_eq!(cvm.num_rows, 3);
        assert_eq!(cvm.num_columns, 1);
        assert_eq!(cvm.get(0, 0), 1.0);
        assert_eq!(cvm.get(1, 0), 2.0);
        assert_eq!(cvm.get(2, 0), 3.0);
    }

    #[test]
    fn plus_works() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[4.0, 5.0, 6.0]);
        let cv3 = cv.plus(&cv2);
        assert_eq!(cv2.num_elements(), 3);
        assert_eq!(cv3.get(0), 5.0);
        assert_eq!(cv3.get(1), 7.0);
        assert_eq!(cv3.get(2), 9.0);
    }

    #[test]
    fn plus_in_place_works() {
        let mut cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[4.0, 5.0, 6.0]);
        cv.plus_in_place(&cv2);
        assert_eq!(cv.num_elements(), 3);
        assert_eq!(cv.get(0), 5.0);
        assert_eq!(cv.get(1), 7.0);
        assert_eq!(cv.get(2), 9.0);
    }

    #[test]
    fn add_in_place_works() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[4.0, 5.0, 6.0]);
        let res = cv.add(&cv2);
        assert_eq!(res.num_elements(), 3);
        assert_eq!(res.get(0), 5.0);
        assert_eq!(res.get(1), 7.0);
        assert_eq!(res.get(2), 9.0);
    }

    #[test]
    fn minus_works() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[4.0, 5.0, 6.0]);
        let cv3 = cv.minus(&cv2);
        assert_eq!(cv2.num_elements(), 3);
        assert_eq!(cv3.get(0), -3.0);
        assert_eq!(cv3.get(1), -3.0);
        assert_eq!(cv3.get(2), -3.0);
    }

    #[test]
    fn minus_in_place_works() {
        let mut cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[4.0, 5.0, 6.0]);
        cv.minus_in_place(&cv2);
        assert_eq!(cv2.num_elements(), 3);
        assert_eq!(cv.get(0), -3.0);
        assert_eq!(cv.get(1), -3.0);
        assert_eq!(cv.get(2), -3.0);
    }

    #[test]
    fn multiply_by_scalar_works() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = cv.multiply_by_scalar(2.0);
        assert_eq!(cv2.num_elements(), 3);
        assert_eq!(cv2.get(0), 2.0);
        assert_eq!(cv2.get(1), 4.0);
        assert_eq!(cv2.get(2), 6.0);
    }

    #[test]
    fn multiply_by_scalar_in_place_works() {
        let mut cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        cv.multiply_by_scalar_in_place(2.0);
        assert_eq!(cv.num_elements(), 3);
        assert_eq!(cv.get(0), 2.0);
        assert_eq!(cv.get(1), 4.0);
        assert_eq!(cv.get(2), 6.0);
    }

    #[test]
    fn divide_by_scalar_works() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = cv.divide_by_scalar(2.0);
        assert_eq!(cv2.num_elements(), 3);
        assert_eq!(cv2.get(0), 0.5);
        assert_eq!(cv2.get(1), 1.0);
        assert_eq!(cv2.get(2), 1.5);
    }

    #[test]
    fn divide_by_scalar_in_place_works() {
        let mut cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        cv.divide_by_scalar_in_place(2.0);
        assert_eq!(cv.num_elements(), 3);
        assert_eq!(cv.get(0), 0.5);
        assert_eq!(cv.get(1), 1.0);
        assert_eq!(cv.get(2), 1.5);
    }

    #[test]
    fn mult_by_matrix_works() {
        // TODO: add test case for incompatible dimensions

        let v = column_vector![1.0, 2.0];
        let m2 = RowsMatrixBuilder::new()
            .with_row(&[3.0, 4.0])
            .build();

        let m = v.mult_matrix(&m2);
        println!("m: \n{}", m);

        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 2);
        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
    }

    #[test]
    fn dot_product_works() {
        let cv1 = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let dot = cv1.dot_product(&cv2);
        assert_eq!(dot, 14.0);
    }

    #[test]
    fn hadamard_product_works() {
        let cv1 = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv3 = cv1.hadamard_product(&cv2);
        assert_eq!(cv3.num_elements(), 3);
        assert_eq!(cv3.get(0), 1.0);
        assert_eq!(cv3.get(1), 4.0);
        assert_eq!(cv3.get(2), 9.0);
    }

    #[test]
    fn hadamard_product_in_place_works() {
        let mut cv1 = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let cv2 = ColumnVector::new(&[1.0, 2.0, 3.0]);
        cv1.hadamard_product_in_place(&cv2);
        println!("{:?}", cv1);
        assert_eq!(cv1.num_elements(), 3);
        assert_eq!(cv1.get(0), 1.0);
        assert_eq!(cv1.get(1), 4.0);
        assert_eq!(cv1.get(2), 9.0);
    }

    #[test]
    fn test_can_iterate_over_column_vector() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let mut iter = cv.iter();
        assert_eq!(iter.next(), Some(&1.0));
        assert_eq!(iter.next(), Some(&2.0));
        assert_eq!(iter.next(), Some(&3.0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_can_iterate_mutably_over_column_vector() {
        let mut cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        for i in cv.iter_mut() {
            *i += 1.0;
        }
        assert_eq!(cv.get(0), 2.0);
        assert_eq!(cv.get(1), 3.0);
        assert_eq!(cv.get(2), 4.0);
    }

    #[test]
    fn test_can_do_double_iterate_over_column_vectors() {
        // TODO: add test case for incompatible dimensions

        let v0 = column_vector![1.0, 2.0, 3.0];
        let v1 = column_vector![4.0, 5.0, 6.0];
        let mut iter = v0.iter_with(&v1);
        assert_eq!(iter.next(), Some((1.0, 4.0)));
        assert_eq!(iter.next(), Some((2.0, 5.0)));
        assert_eq!(iter.next(), Some((3.0, 6.0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_vec_length() {
        assert_eq!((column_vector![0.0]).vec_length(), 0.0);
        assert_eq!((column_vector![1.0]).vec_length(), 1.0);
        assert_eq!((column_vector![4.0]).vec_length(), 4.0);
        assert_eq!((column_vector![3.0, 4.0]).vec_length(), 5.0);
    }

    #[test]
    fn transpose_works() {
        let cv = ColumnVector::new(&[1.0, 2.0, 3.0]);
        let m = cv.transpose();
        assert_eq!(m.num_rows(), 1);
        assert_eq!(m.num_columns(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
    }

    #[test]
    fn into_value_works() {
        // TODO: add test case for lengths != 1
        let cv = column_vector![42.0];
        let x = cv.into_value();
        assert_eq!(x, 42.0);
    }

    #[test]
    fn fill_new_works() {
        let cv = ColumnVector::fill_new(42.0, 5);
        assert_eq!(cv.num_elements(), 5);
        assert_eq!(cv.get(0), 42.0);
        assert_eq!(cv.get(1), 42.0);
        assert_eq!(cv.get(2), 42.0);
        assert_eq!(cv.get(3), 42.0);
        assert_eq!(cv.get(4), 42.0);
    }

    #[test]
    fn empty_works() {
        let cv = ColumnVector::empty();
        assert_eq!(cv.num_elements(), 0);
        assert_eq!(cv.inner_matrix.num_columns, 1);
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

        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 3);
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

        assert_eq!(m.num_rows, 2);
        assert_eq!(m.num_columns, 3);
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

        assert_eq!(m.num_rows, 3);
        assert_eq!(m.num_columns, 2);
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

        assert_eq!(m.num_rows, 3);
        assert_eq!(m.num_columns, 2);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(2, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(2, 1), 6.0);
    }
}
