use rand::{self, rngs::ThreadRng, Rng};

// The Activation Function enum to avoid string comparison
#[derive(Clone)]
pub enum AF {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    NA
}

// The nodule struct, which is the fundamental structure in VIO
#[derive(Clone)]
pub struct Nodule {
    pub weights: Vec<f32>,
    pub activation: AF
}

// A Noduon struct to contain both neurons and bias nodes
#[derive(Clone)]
pub enum Noduon {
    Neuron(Nodule),
    Bias
}

// Noduon functions
impl Noduon {

    // process function, which takes in the past_layer and performs the weighted sum against the noduon weights
    pub fn process(&self, past_layer: &Vec<f32>) -> f32 {
        match self {
            // neruon processing applies weighted sum and then activation function
            Noduon::Neuron(f) => {

                let mut sum: f32 = f.weights.iter().zip(past_layer.iter()).fold(0.0, |acc, (&w, &p)| acc + w * p);

                // activation function
                sum = match f.activation {
                    AF::Tanh => sum.tanh(),
                    AF::ReLU => sum.max(0.0),
                    AF::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                    _ => 0.0
                 };

                 // return sum
                 sum
                },

            // Bias noduon only returns 1.0
            Noduon::Bias => 1.0
        }
    } // End process

    // Set weights directly sets the weights of this Noduon. Ignores bias.
    pub fn set_weights(&mut self, new_weights: &Vec<f32>) {
        match self {
            // Set neuron weights. This requires a clone with to_vec()
            Noduon::Neuron(f) => {
                f.weights = new_weights.to_vec()
            },
            Noduon::Bias => ()
        }
    }

    // push_weight adds an additional weight between the last two weights, effectively placing it at the last slot before bias
    pub fn push_weight(&mut self, new_weight: f32) {
        match self {
            Noduon::Neuron(f) => {

                // pop last element in the neuron weight
                match f.weights.pop() {

                    // push them back onto the vec, ensuring the old last element is still the last one
                    Some(bias) => {
                        f.weights.push(new_weight);
                        f.weights.push(bias)
                    },
                    None => print!("Fatal error in push_weight: pop failed")
                };
            },
            Noduon::Bias => ()
        }
    } // End push_weight

    // change_weights takes a vector and changes each of the noduon's weights by the respective amount
    pub fn change_weights(&mut self, weight_changes: &Vec<f32>) {
        match self {
            Noduon::Neuron(f) => {
                for i in 0..f.weights.len() {
                    f.weights[i] += weight_changes[i]
                }
            },
            Noduon::Bias => ()
        }
    } // End change_weights
 
    // get_weights returns this Noduon's weights as a vec
    pub fn get_weights(&self) -> Vec<f32> {
        match self {
            Noduon::Neuron(f) => {
                f.weights.clone()
            },
            Noduon::Bias => Vec::new()
        }
    } // End get_weights

    // get_function returns this Noduon's AF
    pub fn get_function(&self) -> AF {
        match self {
            Noduon::Neuron(f) => f.activation.clone(),
            Noduon::Bias => AF::NA
        }
    } // End get_function

}

/// Layers ///

// Layers contain a Vec of noduons
#[derive(Clone)]
pub struct Layer {
    pub noduons: Vec<Noduon>
}

impl Layer {
    // process is the basic function of the layer, performing the weighted sum for each Noduon based on the input
    pub fn process(&self, input: &Vec<f32>) -> Vec<f32>{
        self.noduons.iter().map(|x| x.process(input)).collect()
    } // End process

    // set_weights takes a 2d matrix of weights and sets each row into each of this layers's noduon's weights
    pub fn set_weights(&mut self, new_weights: &Vec<Vec<f32>>) {
        for (nod, weights) in self.noduons.iter_mut().zip(new_weights.iter()) {
            nod.set_weights(weights)
        }
    } // End set_weights

    // push_weights takes a Vec of weights and pushes it to each of this layer's noduons
    pub fn push_weights(&mut self, pushing: Vec<f32>) {
        for (nod, weight) in self.noduons.iter_mut().zip(pushing.iter()) {
            nod.push_weight(*weight)
        }
    } // End push_weights

    // change_weights takes a 2d matrix of weights and changes each of this layer's noduons
    pub fn change_weights(&mut self, weight_changes: &Vec<Vec<f32>>) {
        for (nod, weights) in self.noduons.iter_mut().zip(weight_changes.iter()) {
            nod.change_weights(weights)
        }
    } // End change_weights

    // get_weights returns the 2d matrix of weights
    pub fn get_weights(&self) -> Vec<Vec<f32>> {
        self.noduons.iter().map(|x| x.get_weights()).collect()
    } // End get_weights

    // get_functions returns the vec of functions
    pub fn get_functions(&self) -> Vec<AF> {
        self.noduons.iter().map(|x| x.get_function()).collect()
    } // End get_functions
}

/// Network ///

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>
}

impl Network {
    
    // process is the primary function of the network, taking in the inputs and pushing outputs
    pub fn process(&self, mut input: Vec<f32>) -> Vec<f32> {
        input.push(1.0);
        for layer in self.layers.iter() {
            input = layer.process(&input)
        }
        input
    } // End process

    // set_weights takes a 3d matrix of weights and distributes them over the network
    pub fn set_weights(&mut self, new_weights: Vec<Vec<Vec<f32>>>) {
        for (layer, weights) in self.layers.iter_mut().zip(new_weights.iter()) {
            layer.set_weights(weights)
        }
    } // End set_weights

    // change_weights takes a 3d matrix of weights and applies the changes of the network
    pub fn change_weights(&mut self, weight_changes: Vec<Vec<Vec<f32>>>) {
        for (layer, changes) in self.layers.iter_mut().zip(weight_changes.iter()) {
            layer.change_weights(changes)
        }
    } // End change_weights

    // get_weights returns the 3d matrix of each noduon's weights
    pub fn get_weights(&self) -> Vec<Vec<Vec<f32>>> {
        self.layers.iter().map(|x| x.get_weights()).collect()
    } // End get_weights

    // add_input will push a new random weight to each noduon in the first layer
    pub fn add_input(&mut self) {
        // Start by generating a random Vec
        let mut rng: ThreadRng = rand::thread_rng();
        let new_weights: Vec<f32> = self.layers[0].noduons.iter().map(|_| rng.gen_range(-1.0..=1.0)).collect();
        // Push it out
        self.layers[0].push_weights(new_weights)
    } // End add_input

    // add_output will push a new noduon with random weights to the last layer
    pub fn add_output(&mut self, function: AF) {
        // Start by generating a random set of weights based on the second to last layer
        let mut rng: ThreadRng = rand::thread_rng();
        let num_layers = self.layers.len();
        let new_weights: Vec<f32> = (0..self.layers[num_layers].noduons.len()).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        // Now we append a noduon with these weights to the last layer
        self.layers[num_layers - 1].noduons.push(Noduon::Neuron(Nodule { weights: (new_weights), activation: (function) }));
    } // End add_output


    // forward_pass will perform the process function, except it will return the output of each layer
    pub fn forward_pass(&self, mut input: Vec<f32>) -> Vec<Vec<f32>> {
        // add bias to input
        input.push(1.0);
        // create the outs and add inputs to it
        let mut outs: Vec<Vec<f32>> = Vec::new();
        outs.push(input.clone());
        // for each layer, process and add the result to the output
        for layer in self.layers.iter() {
            input = layer.process(&input);
            outs.push(input.clone())
        }
        outs
    } // End forward_pass

    // backward_pass computes the gradients of each weight and returns them as a 3d matrix
    pub fn backward_pass(&self, forward_pass: Vec<Vec<f32>>) -> Vec<Vec<Vec<f32>>> {
        let mut gradients: Vec<Vec<Vec<f32>>> = Vec::new();

        let weights: Vec<Vec<Vec<f32>>> = self.get_weights();

        gradients
    }
}

/// Builders ///

// build_neuron constructs a random neuron with a set number of weights and a specific function
pub fn build_neuron(num_weights: usize, function: &AF) -> Noduon {
    let mut rng: ThreadRng = rand::thread_rng();
    let weights: Vec<f32> = (0..num_weights).map(|_| rng.gen_range(-1.0..=1.0)).collect();
    Noduon::Neuron(Nodule { weights, activation: function.clone() })
} // End build_neuron

// build_layer constructs a layer with a specific number of noduons and weights, with random weights and a single function
pub fn build_layer(num_noduons: usize, num_weights: usize, function: &AF) -> Layer{
    let nods: Vec<Noduon> = (0..num_noduons).map(|_| build_neuron(num_weights, &function)).collect();
    Layer {noduons: nods}
} // End build_layer

// build_network builds a network based on the network_shape Vec, which will describe the number of noduons in each layer
pub fn build_network(num_inputs: usize, network_shape: Vec<usize>, gen_func: AF, out_func: AF) -> Network {
    // Inititalize the layers Vec
    let mut layers: Vec<Layer> = Vec::new();
    let num_layers: usize = network_shape.len();

    // Build the first layer
    layers.push(build_layer(network_shape[0], num_inputs, &gen_func));

    // Build the hidden layers
    for i in 1..(num_layers - 1) {
        layers.push(build_layer(network_shape[i], network_shape[i-1], &gen_func))
    }

    // Build the output layer
    layers.push(build_layer(network_shape[num_layers - 1], network_shape[num_layers - 2], &out_func));
    
    Network {layers}
} // End build_network

/// Main fn for testing ///
fn main() {
    print!("hi")
}