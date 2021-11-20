use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
enum Quadrant {
    One,
    Two,
    Three,
    Four,
}

#[derive(Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Point {}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn points_on_line_with_gaussian_noise(
        intercept: f64,
        slope: f64,
        noise_std_dev: f64,
        num_points: usize,
        x_min_inclusive: f64,
        x_max_exclusive: f64,
    ) -> Vec<Point> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, noise_std_dev).unwrap();
        let mut points = Vec::new();
        for _ in 0..num_points {
            let x = rng.gen_range(x_min_inclusive..x_max_exclusive);
            let y = intercept + slope * x + normal.sample(&mut rng);
            points.push(Point { x, y });
        }
        points
    }

    // pub fn random(distribution: &dyn Distribution<T>) -> Point {
    //     match distribution {
    //         Distribution::Uniform(min, max) => {
    //             let mut rng = rand::thread_rng();
    //             let x = rng.gen_range(min, max);
    //             let y = rng.gen_range(min, max);
    //             Point::new(x, y)
    //         },
    //         Distribution::Gaussian(mean, stddev) => {
    //             let mut rng = rand::thread_rng();
    //             let x = rng.gen_range(mean - stddev, mean + stddev);
    // }

    pub fn copy_to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
