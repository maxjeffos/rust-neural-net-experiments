use std::fmt;

#[derive(Debug)]
pub enum NeuralNetworkError {
    IndexOutOfBoundsError(IndexOutOfBoundsError),
    InvalidLayerIndex(InvalidLayerIndex),
    VectorDimensionMismatch(VectorDimensionMismatch),
}

impl fmt::Display for NeuralNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeuralNetworkError::IndexOutOfBoundsError(e) => {
                write!(f, "Neural Network Error: {}", e)
            }
            NeuralNetworkError::InvalidLayerIndex(e) => {
                write!(f, "Neural Network Error: {}", e)
            }
            NeuralNetworkError::VectorDimensionMismatch(e) => {
                write!(f, "Neural Network Error - VectorDimensionMismatch: {}", e)
            }
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

#[derive(Debug, PartialEq)]
pub struct VectorDimensionMismatch {
    pub len1: usize,
    pub len2: usize,
    pub msg: String,
}

impl VectorDimensionMismatch {
    pub fn new(len1: usize, len2: usize) -> Self {
        let msg = format!(
            "VectorDimensionMismatch: vectors of length {} and {} do not match",
            len1, len2
        );
        VectorDimensionMismatch { len1, len2, msg }
    }

    pub fn new_with_msg(len1: usize, len2: usize, msg: &str) -> Self {
        let msg = format!(
            "VectorDimensionMismatch: vectors of length {} and {} do not match. {}",
            len1, len2, msg
        );
        VectorDimensionMismatch { len1, len2, msg }
    }
}

impl fmt::Display for VectorDimensionMismatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for VectorDimensionMismatch {}
