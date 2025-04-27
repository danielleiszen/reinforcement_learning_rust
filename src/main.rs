use luminal::{nn::{activation::ReLU, linear::Linear}, prelude::*};

const GRID_SIZE: usize = 5;
const NUM_ACTIONS: usize = 4; // Actions: Up, Down, Left, Right
// const ALPHA: f64 = 0.1;  // Learning rate
// const GAMMA: f64 = 0.9; // Discount factor
const INITIAL_EPSILON: f64 = 0.9;
const EPSILON_DECAY: f64 = 0.995;
const MIN_EPSILON: f64 = 0.1;
const EPISODES: usize = 100;

fn main() {
    let mut cx = Graph::new();

    let model = (
        Linear::<2, 16>::initialize(&mut cx),
        ReLU,
        Linear::<16, 16>::initialize(&mut cx),
        ReLU,
        Linear::<16, 4>::initialize(&mut cx)
    );

    let mut input = cx.tensor::<R1<2>>();
    let mut target = cx.tensor::<R1<4>>();

    let mut output = model.forward(input).retrieve();
    let mut loss = mse_loss(output, target).retrieve();
}