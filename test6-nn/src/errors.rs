use std::fmt;

#[derive(Debug)]
pub enum NeuralNetworkError {
    IndexOutOfBoundsError(IndexOutOfBoundsError),
    InvalidLayerIndex(InvalidLayerIndex),
    // Add more variants for other error cases if needed
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeuralNetworkError::IndexOutOfBoundsError(e) => {
                write!(f, "Neural Network Error: {}", e)
            } // ...
            NeuralNetworkError::InvalidLayerIndex(e) => {
                write!(f, "Neural Network Error: {}", e)
            } // ...
        }
    }
}

impl std::error::Error for NeuralNetworkError {}

#[derive(Debug)]
pub struct IndexOutOfBoundsError(pub usize);

impl fmt::Display for IndexOutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IndexOutOfBoundsError: Index {} is out of bounds",
            self.0
        )
    }
}

impl std::error::Error for IndexOutOfBoundsError {}

#[derive(Debug)]
pub struct InvalidLayerIndex(pub usize);

impl fmt::Display for InvalidLayerIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "InvalidLayerIndex: Index {} is not valid for this network",
            self.0
        )
    }
}

impl std::error::Error for InvalidLayerIndex {}
