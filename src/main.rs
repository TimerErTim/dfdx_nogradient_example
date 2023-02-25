use dfdx::optim::{Adam, AdamConfig};
use dfdx::prelude::*;
use rand::distributions::Uniform;
use rand::{random, Rng, thread_rng};
use rand::prelude::Distribution;

type ModelBuilder = (
    (Linear<10, 64>, Sigmoid),
    (Linear<64, 32>, ReLU),
    (Linear<32, 16>, ReLU),
    (Linear<16, 5>, ReLU),
);
type Model = <ModelBuilder as BuildOnDevice<Cpu, f32>>::Built;

struct ModelTrainer {
    dev: Cpu,
    model: Model,
    opt: Adam<Model>
}

impl ModelTrainer {
    fn new() -> Self {
        let dev = Cpu::default();
        let mut model = dev.build_module::<ModelBuilder, f32>();
        model.reset_params();
        let opt = Adam::new(&model, AdamConfig {
            lr: 1e-3,
            ..Default::default()
        });

        Self {
            dev,
            model,
            opt
        }
    }

    fn step(&mut self, step: usize, input: [f32;10], target: [f32;5]) -> f32 {
        let y = self.dev.tensor(target);
        let x = self.dev.tensor(input).traced();
        let prediction = self.model.forward_mut(x);

        let prediction_data = prediction.array();

        let loss = mse_loss(prediction, y);
        let loss_data = loss.array();
        let gradients = loss.backward();

        let sum_gradients = gradients.get(&self.model.0.0.bias).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.1.0.bias).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.2.0.bias).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.3.0.bias).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.0.0.weight).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.1.0.weight).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.2.0.weight).as_vec().iter().sum::<f32>() +
            gradients.get(&self.model.3.0.weight).as_vec().iter().sum::<f32>();

        if sum_gradients == 0.0 {
            println!("Loss {loss_data} at Step {step}: {input:?} -> {prediction_data:?}; Target: {target:?}");
        }

        self.opt.update(&mut self.model, gradients).unwrap();

        loss_data
    }
}

fn main() {
    let mut trainer = ModelTrainer::new();

    let mut uniform_iter = Uniform::new(-1f32, 1f32).sample_iter(thread_rng());

    for i in 0..2_000_000 {
        let sample = thread_rng().gen_range(0..100);

        let first_target = (sample < 10) as i32 as f32; // You can fiddle with the threshold to produce a "stable" result where all gradients add up to be zero
        // This stable state remains even when the target output is clearly different from the produced output

        trainer.step(i,
                     uniform_iter.by_ref().take(10).collect::<Vec<f32>>().try_into().unwrap(),
                     [first_target, 0., 0., (sample == 3) as i32 as f32, (sample > 95) as i32 as f32]);
    }
}
