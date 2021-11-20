use common::point::Point;

fn main() {
    let the_data = vec![
        Point::new(0.0, 0.0),
        Point::new(2.0, 2.0),
        Point::new(4.0, 4.0),
        Point::new(6.0, 6.0),
        Point::new(3.0, 3.0),
        Point::new(5.0, 5.0),
    ];

    the_data.iter().for_each(|p| {
        println!("{:?}", p);
    });
    println!();

    let theta_init = (1.0, 0.0); // ideal line for this data is (0.0, 1.0)

    let mut theta0 = theta_init.0;
    let mut theta1 = theta_init.1;

    let learning_rate = 0.1;

    let max_iterations = 200;
    let mut iteration = 0;

    loop {
        let cost_fn_partial_derivatives = compute_partial_derivatives(theta0, theta1, &the_data);
        let cost_fn_pd_0 = cost_fn_partial_derivatives.0;
        let cost_fn_pd_1 = cost_fn_partial_derivatives.1;

        theta0 -= learning_rate * cost_fn_pd_0;
        theta1 -= learning_rate * cost_fn_pd_1;

        iteration += 1;

        if iteration >= max_iterations {
            println!("max iterations reached: {}", iteration);
            break;
        }
    }

    println!("theta0: {}", theta0);
    println!("theta1: {}", theta1);

    println!("final cost: {}", cost(theta0, theta1, &the_data));
}

fn compute_partial_derivatives(theta0: f64, theta1: f64, data: &[Point]) -> (f64, f64) {
    let mut sum0 = 0_f64;
    let mut sum1 = 0_f64;
    let m = data.len();

    data.iter().for_each(|p| {
        let x = p.x; // might get better performance by loading this into a register
        let y = p.y; // might get better performance by loading this into a register

        let hyp = hypothesis(theta0, theta1, x);
        sum0 += hyp - y;
        sum1 += (hyp - y) * x;
    });

    (sum0 / m as f64, sum1 / m as f64)
}

fn hypothesis(theta0: f64, theta1: f64, x: f64) -> f64 {
    theta0 + (theta1 * x)
}

fn cost(theta0: f64, theta1: f64, data: &[Point]) -> f64 {
    let mut cost = 0.0;
    for point in data {
        let hyp = hypothesis(theta0, theta1, point.x);
        let diff = hyp - point.y;
        let diff_squared = diff * diff;
        cost += diff_squared;
    }
    cost / (2.0 * data.len() as f64)
}
