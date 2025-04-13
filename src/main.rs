use std::collections::VecDeque;

use ndarray::Array1;
use rand::{rng, seq::IteratorRandom, Rng};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, loss::mse, AdamW, Optimizer, VarBuilder, VarMap};

const GRID_SIZE: usize = 5;
const NUM_ACTIONS: usize = 4; // Actions: Up, Down, Left, Right
// const ALPHA: f64 = 0.1;  // Learning rate
// const GAMMA: f64 = 0.9; // Discount factor
const INITIAL_EPSILON: f64 = 0.9;
const EPSILON_DECAY: f64 = 0.995;
const MIN_EPSILON: f64 = 0.1;
const EPISODES: usize = 500;

// Define the Experience Replay Buffer
struct ReplayBuffer {
    buffer: VecDeque<(Tensor, i64, f64, Tensor, bool)>,
}

impl ReplayBuffer {
    fn new() -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(10000),
        }
    }

    fn push(&mut self, experience: (Tensor, i64, f64, Tensor, bool)) {
        if self.buffer.len() == 10000 {
            self.buffer.pop_front(); // Remove oldest experience
        }
        self.buffer.push_back(experience);
    }

    fn sample(&self) -> Vec<(Tensor, i64, f64, Tensor, bool)> {
        let mut rng = rng();
        let indices: Vec<usize> = (0..self.buffer.len())
            .choose_multiple(&mut rng, 64);
        indices.into_iter()
            .map(|idx| self.buffer[idx].clone())
            .collect()
    }
}

struct QNetwork {
    layer1: linear::Linear,
    layer2: linear::Linear,
    output: linear::Linear,
    optimizer: AdamW,
}

impl QNetwork {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);
        let adam = AdamW::new_lr(varmap.all_vars(), 0.01).unwrap();

        Ok(Self {
            layer1: linear(input_dim, hidden_dim, vb.pp("l1"))?,
            layer2: linear(hidden_dim, hidden_dim, vb.pp("l2"))?,
            output: linear(hidden_dim, output_dim, vb.pp("ou"))?,
            optimizer: adam,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.apply(&self.layer1)?.relu().unwrap();
        let x = x.apply(&self.layer2)?.relu().unwrap();
        x.apply(&self.output)
    }

    fn train(&mut self, replay_buffer: &ReplayBuffer, device: &Device) -> Result<()> {
        let batches = replay_buffer.sample();

        let states: Vec<_> = batches.iter().map(|e| e.0.clone()).collect();
        let actions = batches.iter().map(|e| e.1);
        let rewards = batches.iter().map(|e|e.2);
        let results: Vec<_> = batches.iter().map(|e|e.3.clone()).collect();
        let terminals = batches.iter().map(|e|e.4 as u8 as f64);

        let states = Tensor::stack(&states, 0)?;
        let actions = Tensor::from_iter(actions, device)?.unsqueeze(1)?;
        let rewards = Tensor::from_iter(rewards, device)?.unsqueeze(1)?;
        let results = Tensor::stack(&results, 2)?;
        let terminals = Tensor::from_iter(terminals, device)?.unsqueeze(1)?;

        let estimation = self.forward(&states)?.gather(&actions, 1)?;
        let expectation = self.forward(&results)?.detach();

        let y = expectation.max_keepdim(1)?;
        let y = (y * 0.99 * terminals + rewards)?;

        let loss = mse(&estimation, &y)?;
        self.optimizer.backward_step(&loss)?;

        Ok(())
    }    
}

fn main() {
    const DEVICE: Device = Device::Cpu;

    let mut q_network = QNetwork::new(2, 64, 4, &DEVICE).unwrap();

    let mut rng = rand::rng();
    let mut epsilon = INITIAL_EPSILON;
    let mut buffer = ReplayBuffer::new();

    for _episode in 0..EPISODES {
        let mut state = Tensor::new(&[0.0, 0.0], &DEVICE).unwrap();
        let mut done = false;

        while !done {
            // Epsilon-greedy action selection
            let action = if rng.random::<f64>() < epsilon {
                rng.random_range(0..NUM_ACTIONS) // Explore
            } else {
                let fr = q_network.forward(&state);

                if fr.is_ok() {
                    fr.unwrap().argmax(0).unwrap().to_scalar::<f64>().unwrap() as usize
                } else {
                    panic!("nneee")
                }
            };

            let (next_state, reward, terminal) = step(state.clone(), action, &DEVICE).unwrap();
            done = terminal;

            buffer.push((state.clone(), action as i64, reward, next_state.clone(), done));
            state = next_state;

            if buffer.buffer.len() > 64 {
                _ = q_network.train(&buffer, &DEVICE);
            }
        }

        // Decay epsilon
        epsilon = (epsilon * EPSILON_DECAY).max(MIN_EPSILON);
    }

    for y in 0..GRID_SIZE {
        let mut row = Array1::zeros([GRID_SIZE]);
        for x in 0..GRID_SIZE {
            let probe = Tensor::new(&[x as f64, y as f64], &DEVICE).unwrap();

            if let Ok(max) = q_network.forward(&probe).unwrap().argmax(0) {
                row[x] = max.unsqueeze(1).unwrap().to_scalar::<f64>().unwrap() as usize;
            }
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

fn direction(index: usize) -> String {
    match index {
        0 => "^".to_string(),
        1 => "D".to_string(),
        2 => "<".to_string(),
        3 => ">".to_string(),
        _ => "?".to_string(),
    }
}

fn step(state: Tensor, action: usize, device: &Device) -> Result<(Tensor, f64, bool)> {
    let x: usize = state.get(0)?.to_scalar::<f64>()? as usize;
    let y: usize = state.get(1)?.to_scalar::<f64>()? as usize;

    let (next_x, next_y) = match action {
        0 => (x, y.saturating_sub(1)), // Up
        1 => (x, (y + 1).min(GRID_SIZE - 1)), // Down
        2 => (x.saturating_sub(1), y), // Left
        3 => ((x + 1).min(GRID_SIZE - 1), y), // Right
        _ => (x, y),
    };

    let next_state = Tensor::new(&[next_x as f64, next_y as f64], device)?;
    let reward = if next_y == GRID_SIZE - 1 && next_x == GRID_SIZE - 1 { 10.0 } else { -0.1 };
    let terminal = reward == 10.0;

    Ok((next_state, reward, terminal))
}