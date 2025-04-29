use ndarray::{Array1, Axis};
use rand::seq::IteratorRandom;

pub struct Samples {
    pub sample_x: Array1<f64>,
    pub sample_y: Array1<f64>,
    pub train_x: Array1<f64>,
    pub train_y: Array1<f64>,
}

impl Samples {
    pub fn create_from(x: &Array1<f64>, y: Array1<f64>, samples: usize) -> Samples {
        let n = x.len();
        let mut rng = rand::rng();
    
        let sample_idx = (0..samples as usize).choose_multiple(&mut rng, samples as usize);
        let remaining_idx: Vec<usize> = (0..n).filter(|i| sample_idx.contains(i) == false).collect();
        
        Samples {
            sample_x: x.select(Axis(0), &sample_idx),
            sample_y: y.select(Axis(0), &sample_idx),
            train_x: x.select(Axis(0), &remaining_idx),
            train_y: y.select(Axis(0), &remaining_idx),
        }
    }
}