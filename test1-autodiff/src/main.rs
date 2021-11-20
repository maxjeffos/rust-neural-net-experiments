use autodiff::*;

fn e_to_the_x(x: FT<f64>) -> FT<f64> {
    x.exp()
}

fn square(x: u8) -> u8 {
    x * x
}

fn do_op(f: impl Fn(u8) -> u8, x: u8) {
    let y = f(x);
    println!("f({}) = {}", x, y);
}

fn do_op_dual(f: impl Fn(FT<f64>) -> FT<f64>, x: f64) {
    let ft_x = FT::cst(x);
    let y = f(ft_x);
    println!("f({}) = {}", x, y);

    let derivative_at_x = diff(&f, x);
    println!("f'({}) = {}", x, derivative_at_x);
}

fn x_squared(x: FT<f64>) -> FT<f64> {
    x * x
}

fn main() {
    let e = 1_f64.exp();
    println!("{}", e);

    let f = |x: FT<f64>| x.exp();

    let one = F::cst(1.0);
    let f_at_1 = f(one);
    let f_at_2 = f(F::cst(2.0));
    println!("f_at_1: {}", f_at_1);
    println!("f_at_2: {}", f_at_2);

    let d_at_1 = diff(&f, 1.0);
    let d_at_2 = diff(&f, 2.0);
    println!("d_at_1: {}", d_at_1);
    println!("d_at_2: {}", d_at_2);

    for i in 0..4 {
        println!("{} squared is {}", i, square(i));
        do_op(square, i);
    }

    // now with diff
    for i in 0..4 {
        let f = i as f64;
        do_op_dual(x_squared, f);
    }
}
