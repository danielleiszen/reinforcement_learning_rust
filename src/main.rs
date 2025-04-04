use ndarray::{Array, Array1, Array2};
use rand::Rng;

const GRID_SIZE: usize = 5;
const NUM_ACTIONS: usize = 4; // Actions: Up, Down, Left, Right
const ALPHA: f64 = 0.1;  // Learning rate
const GAMMA: f64 = 0.9; // Discount factor
const INITIAL_EPSILON: f64 = 0.9;
const EPSILON_DECAY: f64 = 0.995;
const MIN_EPSILON: f64 = 0.1;
const EPISODES: usize = 500;

fn main() {
    let mut q_table = Array2::<f64>::zeros((GRID_SIZE * GRID_SIZE, NUM_ACTIONS));
    let mut rng = rand::rng();
    let mut epsilon = INITIAL_EPSILON;

    for _episode in 0..EPISODES {
        let mut state = rng.random_range(0..GRID_SIZE * GRID_SIZE);
        let mut done = false;

        while !done {
            // Epsilon-greedy action selection
            let action = if rng.random::<f64>() < epsilon {
                rng.random_range(0..NUM_ACTIONS) // Explore
            } else {
                q_table.row(state)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0) // Exploit
            };

            let (next_state, reward, terminal) = step(state, action);
            done = terminal;

            // Q-learning update
            let max_q_next = q_table.row(next_state).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            q_table[[state, action]] += ALPHA * (reward + GAMMA * max_q_next - q_table[[state, action]]);

            state = next_state;
        }

        // Decay epsilon
        epsilon = (epsilon * EPSILON_DECAY).max(MIN_EPSILON);
    }

    for y in 0..GRID_SIZE {
        let mut row = Array1::zeros([GRID_SIZE]);
        for x in 0..GRID_SIZE {
            let i = y * GRID_SIZE + x;

            if let Some(max) = q_table.row(i).iter().cloned().reduce(f64::max) {
                if let Some(pos) = q_table.row(i).iter().position(|&p| p == max) {
                    row[x] = pos;
                }
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

fn step(state: usize, action: usize) -> (usize, f64, bool) {
    let x = state % GRID_SIZE;
    let y = state / GRID_SIZE;

    let (next_x, next_y) = match action {
        0 => (x, y.saturating_sub(1)), // Up
        1 => (x, (y + 1).min(GRID_SIZE - 1)), // Down
        2 => (x.saturating_sub(1), y), // Left
        3 => ((x + 1).min(GRID_SIZE - 1), y), // Right
        _ => (x, y),
    };

    let next_state = next_x + next_y * GRID_SIZE;
    let reward = if next_state == GRID_SIZE * GRID_SIZE - 1 { 10.0 } else { -0.1 };
    let terminal = next_state == GRID_SIZE * GRID_SIZE - 1;

    (next_state, reward, terminal)
}