use common::linalg::{ColumnVector, Matrix, MatrixShape};
use std::collections::HashMap;

use crate::errors::{IndexOutOfBoundsError, InvalidLayerIndex, NeuralNetworkError};
use crate::LayerIndex;

#[derive(Debug, Clone, PartialEq)]
pub struct BigTheta {
    pub sizes: Vec<usize>,
    pub weights_matricies: HashMap<LayerIndex, Matrix>,
    pub bias_vectors: HashMap<LayerIndex, ColumnVector>,
}

impl BigTheta {
    pub fn zero_from_sizes(sizes: &[usize]) -> Self {
        let mut weights_matricies = HashMap::new();
        let mut bias_vectors = HashMap::new();

        for layer_index in 1..sizes.len() {
            let m_shape = MatrixShape::new(sizes[layer_index], sizes[layer_index - 1]);
            let w_empty = Matrix::new_zero_matrix_with_shape(&m_shape);
            let b_empty = ColumnVector::new_zero_vector(sizes[layer_index]);

            weights_matricies.insert(layer_index, w_empty);
            bias_vectors.insert(layer_index, b_empty);
        }

        BigTheta {
            sizes: sizes.to_vec(),
            weights_matricies,
            bias_vectors,
        }
    }

    /// Returns a `Vec<f64>` containing the unrolled weights and biases of the neural network.
    ///
    /// The unrolled vector is constructed by concatenating the weights and biases of each layer
    /// in the network, starting from layer 1. At each layer, the weights of a layer are included before its biases.
    ///
    pub fn unroll(&self) -> Vec<f64> {
        let mut unrolled_vec = Vec::new();
        for l in 1..self.sizes.len() {
            let w = self.get_weights_matrix(&l);
            let b = self.get_bias_vector(&l);

            unrolled_vec.extend(w.data.as_slice());
            unrolled_vec.extend(b.get_data_as_slice());
        }
        unrolled_vec
    }

    /// Returns a mutable reference to the `Matrix` representing the weights for the specified layer.
    ///
    /// # Arguments
    ///
    /// * `layer_index` - A `LayerIndex` representing the index of the layer whose weights you want to access.
    ///
    /// # Errors
    ///
    /// Returns a `NeuralNetworkError::InvalidLayerIndex` if the given `layer_index` is not valid - i.e. either than layer
    /// doesn't exist, or it doesn't have an associated weights matrix.
    ///
    pub fn weights_at_layer_mut(
        // TODO(dedupe): note that there's get_weights_matrix_mut below which is the same except that it uses unwrap
        &mut self,
        layer_index: LayerIndex,
    ) -> Result<&mut Matrix, NeuralNetworkError> {
        self.weights_matricies
            .get_mut(&layer_index)
            .ok_or(NeuralNetworkError::InvalidLayerIndex(InvalidLayerIndex(
                layer_index,
            )))
    }

    /// Returns a mutable reference to the `ColumnVector` representing the biases for the specified layer.
    ///
    /// # Arguments
    ///
    /// * `layer_index` - A `LayerIndex` representing the index of the layer whose biases you want to access.
    ///
    /// # Errors
    ///
    /// Returns a `NeuralNetworkError::InvalidLayerIndex` if the given `layer_index` is not valid - i.e. either than layer
    /// doesn't exist, or it doesn't have an associated bias vector.
    ///
    pub fn bias_at_layer_mut(
        &mut self,
        layer_index: LayerIndex,
    ) -> Result<&mut ColumnVector, NeuralNetworkError> {
        self.bias_vectors
            .get_mut(&layer_index)
            .ok_or(NeuralNetworkError::IndexOutOfBoundsError(
                IndexOutOfBoundsError(layer_index),
            ))
    }

    pub fn get_weights_matrix(&self, layer_index: &LayerIndex) -> &Matrix {
        self.weights_matricies.get(layer_index).unwrap()
    }

    pub fn get_bias_vector(&self, layer_index: &LayerIndex) -> &ColumnVector {
        self.bias_vectors.get(layer_index).unwrap()
    }

    pub fn get_weights_matrix_mut(&mut self, layer_index: &LayerIndex) -> &mut Matrix {
        self.weights_matricies.get_mut(layer_index).unwrap()
    }

    pub fn get_bias_vector_mut(&mut self, layer_index: &LayerIndex) -> &mut ColumnVector {
        self.bias_vectors.get_mut(layer_index).unwrap()
    }

    pub fn mult_scalar_in_place(&mut self, scalar: f64) {
        for (_, w) in self.weights_matricies.iter_mut() {
            w.mult_scalar_mut(scalar);
        }

        for (_, b) in self.bias_vectors.iter_mut() {
            b.mult_scalar_mut(scalar);
        }
    }

    pub fn mult_scalar_return_new(&self, scalar: f64) -> BigTheta {
        let mut new_big_theta = self.clone();
        new_big_theta.mult_scalar_in_place(scalar);
        new_big_theta
    }

    pub fn divide_scalar_in_place(&mut self, scalar: f64) {
        for (_, w) in self.weights_matricies.iter_mut() {
            w.div_scalar_mut(scalar);
        }

        for (_, b) in self.bias_vectors.iter_mut() {
            b.div_scalar_mut(scalar);
        }
    }

    pub fn divide_scalar_return_new(&self, scalar: f64) -> BigTheta {
        let mut new_big_theta = self.clone();
        new_big_theta.divide_scalar_in_place(scalar);
        new_big_theta
    }

    pub fn subtract_in_place(&mut self, other: &Self) {
        for (layer_index, w) in self.weights_matricies.iter_mut() {
            let other_w = other.weights_matricies.get(layer_index).unwrap();
            w.subtract_mut(other_w);
        }

        for (layer_index, b) in self.bias_vectors.iter_mut() {
            let other_b = other.bias_vectors.get(layer_index).unwrap();
            b.subtract_mut(other_b);
        }
    }

    pub fn add_in_place(&mut self, other: &Self) {
        for (layer_index, w) in self.weights_matricies.iter_mut() {
            let other_w = other.weights_matricies.get(layer_index).unwrap();
            w.add_mut(other_w);
        }

        for (layer_index, b) in self.bias_vectors.iter_mut() {
            let other_b = other.bias_vectors.get(layer_index).unwrap();
            b.add_mut(other_b);
        }
    }

    pub fn elementwise_mult_in_place(&mut self, other: &Self) {
        for (layer_index, w) in self.weights_matricies.iter_mut() {
            let other_w = other.weights_matricies.get(layer_index).unwrap();
            w.hadamard_product_in_place(other_w);
        }

        for (layer_index, b) in self.bias_vectors.iter_mut() {
            let other_b = other.bias_vectors.get(layer_index).unwrap();
            b.hadamard_product_in_place(other_b);
        }
    }

    pub fn elementwise_divide_in_place(&mut self, other: &Self) {
        for (layer_index, w) in self.weights_matricies.iter_mut() {
            let other_w = other.weights_matricies.get(layer_index).unwrap();
            w.elementwise_divide_in_place(other_w);
        }

        for (layer_index, b) in self.bias_vectors.iter_mut() {
            let other_b = other.bias_vectors.get(layer_index).unwrap();
            b.elementwise_divide_in_place(other_b);
        }
    }

    pub fn add_scalar_to_each_element_in_place(&mut self, scalar: f64) {
        for (_, w) in self.weights_matricies.iter_mut() {
            w.add_scalar_to_each_element_in_place(scalar);
        }

        for (_, b) in self.bias_vectors.iter_mut() {
            b.add_scalar_to_each_element_in_place(scalar);
        }
    }

    pub fn elementwise_square_root_in_place(&mut self) {
        for (_, w) in self.weights_matricies.iter_mut() {
            w.elementwise_square_root_in_place();
        }

        for (_, b) in self.bias_vectors.iter_mut() {
            b.elementwise_square_root_in_place();
        }
    }
}

// test module
#[cfg(test)]
mod tests {
    use super::*;
    use common::column_vector;

    fn create_big_theta_for_test(sizes: &[usize]) -> BigTheta {
        let mut big_theta = BigTheta::zero_from_sizes(sizes);
        let mut next_value = 1.0;

        for layer_index in 1..big_theta.sizes.len() {
            let w = big_theta.weights_matricies.get_mut(&layer_index).unwrap();

            for m in 0..w.num_rows() {
                for n in 0..w.num_columns() {
                    w.set(m, n, next_value);
                    next_value += 1.0;
                }
            }

            let b = big_theta.bias_vectors.get_mut(&layer_index).unwrap();
            for i in 0..b.num_elements() {
                b.set(i, next_value);
                next_value += 1.0;
            }
        }

        big_theta
    }

    fn create_big_theta_for_test_with_scale_factor(sizes: &[usize], scale_factor: f64) -> BigTheta {
        let mut big_theta = BigTheta::zero_from_sizes(sizes);
        let mut next_value = 1.0;

        for layer_index in 1..big_theta.sizes.len() {
            let w = big_theta.weights_matricies.get_mut(&layer_index).unwrap();

            for m in 0..w.num_rows() {
                for n in 0..w.num_columns() {
                    w.set(m, n, next_value * scale_factor);
                    next_value += 1.0;
                }
            }

            let b = big_theta.bias_vectors.get_mut(&layer_index).unwrap();
            for i in 0..b.num_elements() {
                b.set(i, next_value * scale_factor);
                next_value += 1.0;
            }
        }

        big_theta
    }

    fn create_big_theta_for_test_with_squaring(sizes: &[usize]) -> BigTheta {
        let mut big_theta = BigTheta::zero_from_sizes(sizes);
        let mut next_value = 1.0;

        for layer_index in 1..big_theta.sizes.len() {
            let w = big_theta.weights_matricies.get_mut(&layer_index).unwrap();

            for m in 0..w.num_rows() {
                for n in 0..w.num_columns() {
                    w.set(m, n, next_value * next_value);
                    next_value += 1.0;
                }
            }

            let b = big_theta.bias_vectors.get_mut(&layer_index).unwrap();
            for i in 0..b.num_elements() {
                b.set(i, next_value * next_value);
                next_value += 1.0;
            }
        }

        big_theta
    }

    #[test]
    fn test_zero_from_sizes() {
        let sizes = vec![2, 3, 1];
        let big_theta = BigTheta::zero_from_sizes(&sizes);

        assert_eq!(big_theta.sizes, sizes);
        assert_eq!(big_theta.weights_matricies.len(), 2);
        assert_eq!(big_theta.bias_vectors.len(), 2);

        let w1 = big_theta.weights_matricies.get(&1).unwrap();
        let b1 = big_theta.bias_vectors.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![0.0; 6]);
        assert_eq!(b1, &column_vector![0.0, 0.0, 0.0]);

        let w2 = big_theta.weights_matricies.get(&2).unwrap();
        let b2 = big_theta.bias_vectors.get(&2).unwrap();
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![0.0; 3]);
        assert_eq!(b2, &column_vector![0.0]);
    }

    #[test]
    fn create_big_theta_for_test_works() {
        let sizes = vec![2, 3, 1];

        let big_theta = create_big_theta_for_test(&sizes);
        assert_eq!(big_theta.sizes, sizes);
        assert_eq!(big_theta.weights_matricies.len(), 2);
        assert_eq!(big_theta.bias_vectors.len(), 2);

        let w1 = big_theta.weights_matricies.get(&1).unwrap();
        let b1 = big_theta.bias_vectors.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(b1, &column_vector![7.0, 8.0, 9.0]);

        let w2 = big_theta.weights_matricies.get(&2).unwrap();
        let b2 = big_theta.bias_vectors.get(&2).unwrap();
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![10.0, 11.0, 12.0]);
        assert_eq!(b2, &column_vector![13.0]);
    }

    #[test]
    fn create_big_theta_for_test_with_scale_factor_works() {
        let sizes = vec![2, 3, 1];

        let big_theta = create_big_theta_for_test_with_scale_factor(&sizes, 2.0);
        assert_eq!(big_theta.sizes, sizes);
        assert_eq!(big_theta.weights_matricies.len(), 2);
        assert_eq!(big_theta.bias_vectors.len(), 2);

        let w1 = big_theta.weights_matricies.get(&1).unwrap();
        let b1 = big_theta.bias_vectors.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(b1, &column_vector![14.0, 16.0, 18.0]);

        let w2 = big_theta.weights_matricies.get(&2).unwrap();
        let b2 = big_theta.bias_vectors.get(&2).unwrap();
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![20.0, 22.0, 24.0]);
        assert_eq!(b2, &column_vector![26.0]);
    }

    #[test]
    fn test_mult_scalar_in_place_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta = create_big_theta_for_test(&sizes);

        big_theta.mult_scalar_in_place(2.0);

        let w1 = big_theta.get_weights_matrix(&1);
        let b1 = big_theta.get_bias_vector(&1);
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(b1, &column_vector![14.0, 16.0, 18.0]);

        let w2 = big_theta.get_weights_matrix(&2);
        let b2 = big_theta.get_bias_vector(&2);
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![20.0, 22.0, 24.0]);
        assert_eq!(b2, &column_vector![26.0]);
    }

    #[test]
    fn test_mult_scalar_return_new_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta = create_big_theta_for_test(&sizes);

        let bt_new = big_theta.mult_scalar_return_new(2.0);

        let w1 = bt_new.get_weights_matrix(&1);
        let b1 = bt_new.get_bias_vector(&1);
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(b1, &column_vector![14.0, 16.0, 18.0]);

        let w2 = bt_new.get_weights_matrix(&2);
        let b2 = bt_new.get_bias_vector(&2);
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![20.0, 22.0, 24.0]);
        assert_eq!(b2, &column_vector![26.0]);
    }

    #[test]
    fn divide_scalar_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta = create_big_theta_for_test(&sizes);

        big_theta.divide_scalar_in_place(2.0);

        let w1 = big_theta.weights_matricies.get(&1).unwrap();
        let b1 = big_theta.bias_vectors.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        assert_eq!(b1, &column_vector![3.5, 4.0, 4.5]);

        let w2 = big_theta.weights_matricies.get(&2).unwrap();
        let b2 = big_theta.bias_vectors.get(&2).unwrap();
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![5.0, 5.5, 6.0]);
        assert_eq!(b2, &column_vector![6.5]);
    }

    #[test]
    fn test_divide_scalar_return_new_works() {
        let sizes = vec![2, 3, 1];
        let big_theta = create_big_theta_for_test_with_scale_factor(&sizes, 2.0);

        let bt_new = big_theta.divide_scalar_return_new(2.0);

        let w1 = bt_new.weights_matricies.get(&1).unwrap();
        let b1 = bt_new.bias_vectors.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(b1, &column_vector![7.0, 8.0, 9.0]);

        let w2 = bt_new.weights_matricies.get(&2).unwrap();
        let b2 = bt_new.bias_vectors.get(&2).unwrap();
        assert_eq!(w2.shape(), MatrixShape::new(1, 3));
        assert_eq!(&w2.data, &vec![10.0, 11.0, 12.0]);
        assert_eq!(b2, &column_vector![13.0]);
    }

    #[test]
    fn test_subtract_in_place_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta_1 = create_big_theta_for_test_with_scale_factor(&sizes, 2.0);
        let big_theta_2 = create_big_theta_for_test(&sizes);

        big_theta_1.subtract_in_place(&big_theta_2);
        assert_eq!(big_theta_1, big_theta_2);
    }

    #[test]
    fn test_add_in_place_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta_1 = create_big_theta_for_test_with_scale_factor(&sizes, 2.0);
        let big_theta_2 = create_big_theta_for_test(&sizes);
        let big_theta_3 = create_big_theta_for_test_with_scale_factor(&sizes, 3.0);

        big_theta_1.add_in_place(&big_theta_2);
        assert_eq!(big_theta_1, big_theta_3);
    }

    #[test]
    fn test_elementwise_mult_in_place_place_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta_1 = create_big_theta_for_test(&sizes);
        let big_theta_2 = create_big_theta_for_test(&sizes);
        let big_theta_3 = create_big_theta_for_test_with_squaring(&sizes);

        big_theta_1.elementwise_mult_in_place(&big_theta_2);
        assert_eq!(big_theta_1, big_theta_3);
    }

    #[test]
    fn test_elementwise_divide_in_place_place_works() {
        let sizes = vec![2, 3, 1];
        let mut big_theta_1 = create_big_theta_for_test_with_squaring(&sizes);
        let big_theta_2 = create_big_theta_for_test(&sizes);
        let big_theta_3 = create_big_theta_for_test(&sizes);

        big_theta_1.elementwise_divide_in_place(&big_theta_2);
        assert_eq!(big_theta_1, big_theta_3);
    }

    #[test]
    fn test_get_weights_matrix_mut() {
        let sizes = vec![2, 3, 1];
        let mut big_theta = BigTheta::zero_from_sizes(&sizes);

        assert_eq!(big_theta.sizes, sizes);
        assert_eq!(big_theta.weights_matricies.len(), 2);
        assert_eq!(big_theta.bias_vectors.len(), 2);

        let w1 = big_theta.weights_matricies.get(&1).unwrap();
        let b1 = big_theta.bias_vectors.get(&1).unwrap();
        assert_eq!(w1.shape(), MatrixShape::new(3, 2));
        assert_eq!(&w1.data, &vec![0.0; 6]);
        assert_eq!(b1, &column_vector![0.0, 0.0, 0.0]);

        let w1 = big_theta.get_weights_matrix_mut(&1);
        w1.data[0] = 10.0;
        assert_eq!(&w1.data, &vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let b1 = big_theta.get_bias_vector_mut(&1);
        b1.set(0, 20.0);
        assert_eq!(b1, &column_vector![20.0, 0.0, 0.0]);
    }
}
