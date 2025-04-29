use std::f64::consts::PI;

use ndarray::{stack, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_linalg::Solve;
use samples::Samples;

mod samples;

fn main() {
    let N = 100;
    let X = Array1::<f64>::linspace(0.0, 6.0 * PI, N);
    let Y = Array1::<f64>::sin(&X);

    plot_train_vs_test_curves(X, Y, 15, 10);
}

fn make_poly(x: &Array1<f64>, deg: usize) -> Array2<f64> {
    let n = x.len();
    let mut data = vec![Array1::ones(n)];
    
    for d in 0..deg {
        data.push(x.mapv(|v| v.powi((d + 1) as i32)));
    }
    
    stack(Axis(1), &data.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
}

fn get_target(x: &Array1<f64>, xs: &Array1<f64>, ys: &Array1<f64>, deg: usize) -> Array1::<f64> {
    let xspoly = make_poly(xs, deg);
    let w = fit(&xspoly, ys);

    let xpoly = make_poly(x, deg);
    let yhat = xpoly.dot(&w);

    yhat
}

fn get_sample_target(xs: &Array1<f64>, ys: &Array1<f64>, deg: usize) -> Array1::<f64> {
    let xspoly = make_poly(xs, deg);
    let w = fit(&xspoly, ys);

    let yhat = xspoly.dot(&w);

    yhat
}

fn fit(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);

    let a = xtx.solve_into(xty).unwrap();
    a
}

fn get_mse(y: Array1<f64>, yhat: Array1<f64>) -> f64 {
    let d = y - yhat;
    d.dot(&d) / d.len() as f64    
}

fn plot_train_vs_test_curves(x: Array1<f64>, y: Array1<f64>, sample_count: usize, max_deg: usize) -> () {
    let samples = Samples::create_from(&x, y, sample_count);
    let yhat = get_target(&x, &samples.sample_x, &samples.sample_y, max_deg);

    
}

