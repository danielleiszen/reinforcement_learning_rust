use std::collections::VecDeque;

use burn::{backend::Autodiff, module::Module, nn::{loss::MseLoss, Linear, LinearConfig, Relu}, optim::{AdamConfig, GradientsParams, Optimizer}, prelude::Backend, tensor::{backend::AutodiffBackend, cast::ToElement, Float, Int, Tensor}};
use ndarray::Array1;
use rand::{rng, seq::IteratorRandom, Rng};
use burn::backend::Wgpu;

const GRID_SIZE: usize = 5;
const NUM_ACTIONS: usize = 4; // Actions: Up, Down, Left, Right
// const ALPHA: f64 = 0.1;  // Learning rate
// const GAMMA: f64 = 0.9; // Discount factor
const INITIAL_EPSILON: f64 = 0.9;
const EPSILON_DECAY: f64 = 0.995;
const MIN_EPSILON: f64 = 0.1;
const EPISODES: usize = 100;

// Define the Experience Replay Buffer
struct ReplayBuffer<B: Backend, const D: usize> {
    buffer: VecDeque<(Tensor<B, D>, i64, f64, Tensor<B, D>, bool)>,
}

impl<B: Backend, const D: usize> ReplayBuffer<B, D> {
    fn new() -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(10000),
        }
    }

    fn push(&mut self, experience: (Tensor<B, D>, i64, f64, Tensor<B, D>, bool)) {
        if self.buffer.len() == 10000 {
            self.buffer.pop_front(); // Remove oldest experience
        }
        self.buffer.push_back(experience);
    }

    fn sample(&self) -> Vec<(Tensor<B, D>, i64, f64, Tensor<B, D>, bool)> {
        let mut rng = rng();
        let indices: Vec<usize> = (0..self.buffer.len())
            .choose_multiple(&mut rng, 64);
        indices.into_iter()
            .map(|idx| self.buffer[idx].clone())
            .collect()
    }
}

struct QNetworkConfig {
    layer1: LinearConfig,
    layer2: LinearConfig,
    output: LinearConfig,
}

impl QNetworkConfig {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        QNetworkConfig { 
            layer1: LinearConfig { d_input: input_dim, d_output: hidden_dim, bias: true, initializer: burn::nn::Initializer::Normal { mean: 0.5, std: 0.1 } }, 
            layer2: LinearConfig { d_input: hidden_dim, d_output: hidden_dim, bias: true, initializer: burn::nn::Initializer::Normal { mean: 0.5, std: 0.1 } }, 
            output: LinearConfig { d_input: hidden_dim, d_output: output_dim, bias: true, initializer: burn::nn::Initializer::Normal { mean: 0.5, std: 0.1 } },
        }
    }

    fn build<B: AutodiffBackend>(&self, device: &B::Device) -> QNetwork<B> {
        QNetwork { 
            layer1: self.layer1.init(device), 
            layer2: self.layer2.init(device), 
            output: self.output.init(device),
            activation1: Relu::new(),
            activation2: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
struct QNetwork<B: Backend> {
    layer1: Linear<B>,
    activation1: Relu,
    layer2: Linear<B>,
    activation2: Relu,
    output: Linear<B>,
}

impl<B: AutodiffBackend> QNetwork<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.layer1.forward(input);
        let x = self.activation1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.activation2.forward(x);
        self.output.forward(x)
    }

    fn step(&self, state: Tensor<B, 1>, action: usize, device: &B::Device) -> (Tensor<B, 1>, f64, bool) {
        let x = state.clone().select(0, Tensor::<B, 1, Int>::from_data([0], &device)).into_scalar().to_usize();
        let y = state.select(0, Tensor::<B, 1, Int>::from_data([1], &device)).into_scalar().to_usize();
    
        let (next_x, next_y) = match action {
            0 => (x, y.saturating_sub(1)), // Up
            1 => (x, (y + 1).min(GRID_SIZE - 1)), // Down
            2 => (x.saturating_sub(1), y), // Left
            3 => ((x + 1).min(GRID_SIZE - 1), y), // Right
            _ => (x, y),
        };
    
        let next_state = Tensor::<B, 1>::from_floats([next_x as f64, next_y as f64], &device);
        let reward = if next_y == GRID_SIZE - 1 && next_x == GRID_SIZE - 1 { 10.0 } else { -0.1 };
        let terminal = reward == 10.0;
    
        (next_state, reward, terminal)
    }

    fn compute_loss(&mut self, replay_buffer: &ReplayBuffer<B, 1>, device: &B::Device) -> Tensor<B, 2> {
        let batches = replay_buffer.sample();

        let states = batches.iter().map(|e| e.0.clone()).collect();
        let acts: Vec<usize> = batches.iter().map(|e| e.1 as usize).collect();
        let rewards: Vec<f64> = batches.iter().map(|e| e.2).collect();
        let results = batches.iter().map(|e|e.3.clone()).collect();
        let terminals: Vec<f64> = batches.iter().map(|e|e.4 as u8 as f64).collect();

        let states = Tensor::stack::<2>(states, 0);
        let actions = Tensor::<B, 1, Int>::from_ints::<&[usize]>(acts.as_slice(), device).unsqueeze_dim(1);
        let rewards = Tensor::<B, 1, Float>::from_floats::<&[f64]>(rewards.as_slice(), device).unsqueeze_dim(1);
        let results = Tensor::stack::<2>(results, 0);
        let terminals = Tensor::<B, 1, Float>::from_floats::<&[f64]>(terminals.as_slice(), device).unsqueeze_dim(1);

        let estimation = self.forward(states);
        let x = estimation.gather(1, actions);
        let expectation = self.forward(results).detach();

        let y = expectation.max_dim(1);
        let y = y * 0.99 * terminals + rewards;

        let mse = MseLoss::new();
        let loss = mse.forward_no_reduction(x, y);
    
        loss
    }

    fn dump(&self, device: &B::Device) {
        for y in 0..GRID_SIZE {
            let mut row = Array1::zeros([GRID_SIZE]);
            for x in 0..GRID_SIZE {
                let probe = Tensor::<B, 1>::from_floats([x as f64, y as f64], device);
                let dta = probe.to_data();
                let slc = dta.as_slice::<f32>().unwrap();
                let probe = self.forward(probe);
                let max = probe.argmax(0).into_scalar().to_usize();

                println!("x: {}, y: {}, actions: {:?}, max: {}", x, y, slc, max);
                row[x] = max;
            }
    
            println!("{}:[{}, {}, {}, {}, {}]", y, 
                direction(row[0]), 
                direction(row[1]),
                direction(row[2]), 
                direction(row[3]),
                direction(row[4]), 
            )
        }
    }    
}

fn main_old() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiff = Autodiff<MyBackend>;

    let device = Default::default();

    let mut q_network = QNetworkConfig::new(2, 64, 4).build::<MyAutodiff>(&device);

    let mut rng = rand::rng();
    let mut epsilon = INITIAL_EPSILON;
    let mut buffer = ReplayBuffer::<MyAutodiff, 1>::new();
    let mut optim = AdamConfig::new().init::<MyAutodiff, QNetwork<MyAutodiff>>();

    for _episode in 0..EPISODES {
        let x = rng.random_range(0..GRID_SIZE) as f64;
        let y = rng.random_range(0..GRID_SIZE) as f64;
        let mut state = Tensor::from_floats::<&[f64]>(&[x, y], &device);
        let mut done = false;
        let mut steps = 0;
        let mut aggregated = 0.0;

        while !done {
            // Epsilon-greedy action selection
            let action = if rng.random::<f64>() < epsilon {
                rng.random_range(0..NUM_ACTIONS) // Explore
            } else {
                let fr = q_network.forward(state.clone());
                let a = fr.argmax(0).into_scalar();
            
                a as usize
            };

            steps += 1;

            let (next_state, reward, terminal) = q_network.step(state.clone(), action, &device);
            done = terminal;

            buffer.push((state.clone(), action as i64, reward, next_state.clone(), done));
            state = next_state;

            aggregated += reward;

            if buffer.buffer.len() > 64 {
                let loss = q_network.compute_loss(&buffer, &device);
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &q_network);

                q_network = optim.step(0.01, q_network, grads);
            }
        }

        // Decay epsilon
        epsilon = (epsilon * EPSILON_DECAY).max(MIN_EPSILON);

        println!("EPISODE {} from ({};{}) in {} steps got {}", _episode, x, y, steps, aggregated);
        q_network.dump(&device);
    }    
}

fn direction(index: usize) -> String {
    match index {
        0 => "^".to_string(),
        1 => "D".to_string(),
        2 => "<".to_string(),
        3 => ">".to_string(),
        _ => "?".to_string(),
    }
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiff = Autodiff<MyBackend>;

    let device = Default::default();

    let t1 = Tensor::<MyAutodiff, 1>::from_floats([0.1], &device);
    println!("T1: {}", t1.to_data());
}