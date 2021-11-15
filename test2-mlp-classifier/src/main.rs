use std::fmt;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use ndarray::*;

struct MultilayerPerceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl MultilayerPerceptron {

}

struct MLPArchitecture {
    input_size: usize,
    hidden_layers: Vec<usize>,
    output_size: usize,
}

impl MLPArchitecture {
    fn new(input_size: usize, hidden_layers: Vec<usize>, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_layers,
            output_size,
        }
    }
}

#[derive(Debug)]
struct Point {
    x: f64,
    y: f64,
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

#[derive(Debug)]
enum Color {
    blue,
    orange,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Color::blue => write!(f, "blue"),
            Color::orange => write!(f, "orange"),
        }
    }
}


/// Generates a bunch of (x, y) points which map to a color.
/// Generates n blue points and n orange points.
/// The points are randomly generated but only in the (+, +) and (-, -) quadrants i.e quadrants 1 and 3.
/// The colors are either blue in (+, +) or orange in (-, -).
/// See https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.33779&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
fn generate_fake_training_data(n: usize) -> HashMap<Point, Color> {
    let mut training_data = HashMap::<Point, Color>::new();

    for i in 0..n {
        // generate random point in quadrant 1
        let x = rand::random::<f64>() * 90.0 + 10.0; // + 10 for a bit of buffer
        let y = rand::random::<f64>() * 90.0 + 10.0;
        let p = Point { x, y };

        // add item to hashmap

        training_data.insert(p, Color::blue);

        // generate random point in quadrant 3
        let x = rand::random::<f64>() * -90.0 - 10.0;
        let y = rand::random::<f64>() * -90.0 - 10.0;
        let p = Point { x, y };
        training_data.insert(p, Color::orange);
    }

    training_data 
}


fn main() {
    let fake_training_data = generate_fake_training_data(100);
    
    // print each item in fake_training_data
    for (p, c) in fake_training_data {
        println!("{} -> {}", p, c);
    }
}
