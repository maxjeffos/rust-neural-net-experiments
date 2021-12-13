use crate::linalg::ColumnVector;

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn sigmoid_vector(v: &ColumnVector) -> ColumnVector {
    if v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let mut res = ColumnVector::empty();
    for i in 0..v.num_elements() {
        res.push(sigmoid(v.get(i)))
    }
    res
}

pub fn sigmoid_vector_in_place(v: &mut ColumnVector) {
    if v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }
    for i in 0..v.num_elements() {
        v.set(i, sigmoid(v.get(i)));
    }
}

/// Compute the derivative of the sigmoid function at the given z
pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

pub fn sigmoid_prime_vector(v: &ColumnVector) -> ColumnVector {
    if v.num_elements() == 0 {
        panic!("this function is only valid for column vectors with at least one element (row)")
    }

    let mut res = v.clone();

    for i in 0..res.num_elements() {
        res.set(i, sigmoid_prime(v.get(i)));
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_vector;

    #[test]
    fn sigmoid_works() {
        assert_eq!(sigmoid(-4.0), 0.01798620996209156);
        assert_eq!(sigmoid(-2.0), 0.11920292202211755);
        assert_eq!(sigmoid(-1.0), 0.2689414213699951);
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(1.0), 0.7310585786300049);
        assert_eq!(sigmoid(2.0), 0.8807970779778823);
        assert_eq!(sigmoid(4.0), 0.9820137900379085);
    }

    #[test]
    fn sigmoid_vector_works() {
        let m1 = column_vector![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        let m2 = sigmoid_vector(&m1);
        assert_eq!(m2.num_elements(), 7);
        assert_eq!(m2.get(0), sigmoid(-4.0));
        assert_eq!(m2.get(1), sigmoid(-2.0));
        assert_eq!(m2.get(2), sigmoid(-1.0));
        assert_eq!(m2.get(3), 0.5);
        assert_eq!(m2.get(4), sigmoid(1.0));
        assert_eq!(m2.get(5), sigmoid(2.0));
        assert_eq!(m2.get(6), sigmoid(4.0));
    }

    #[test]
    fn sigmoid_vector_in_place_works() {
        let mut z = column_vector![-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        sigmoid_vector_in_place(&mut z);
        assert_eq!(z.num_elements(), 7);
        assert_eq!(z.get(0), sigmoid(-4.0));
        assert_eq!(z.get(1), sigmoid(-2.0));
        assert_eq!(z.get(2), sigmoid(-1.0));
        assert_eq!(z.get(3), 0.5);
        assert_eq!(z.get(4), sigmoid(1.0));
        assert_eq!(z.get(5), sigmoid(2.0));
        assert_eq!(z.get(6), sigmoid(4.0));
    }

    #[test]
    fn sigmoid_prime_works() {
        assert_eq!(sigmoid_prime(-1.0), 0.19661193324148185);
        assert_eq!(sigmoid_prime(-0.5), 0.2350037122015945);
        assert_eq!(sigmoid_prime(0.0), 0.25);
        assert_eq!(sigmoid_prime(0.5), 0.2350037122015945);
        assert_eq!(sigmoid_prime(1.0), 0.19661193324148185);
    }

    #[test]
    fn sigmoid_prime_vec_works() {
        let v = ColumnVector::new(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
        let sig_vec = sigmoid_prime_vector(&v);
        assert_eq!(sig_vec.num_elements(), 5);
        assert_eq!(sig_vec.get(0), 0.19661193324148185);
        assert_eq!(sig_vec.get(1), 0.2350037122015945);
        assert_eq!(sig_vec.get(2), 0.25);
        assert_eq!(sig_vec.get(3), 0.2350037122015945);
        assert_eq!(sig_vec.get(4), 0.19661193324148185);
    }
}
