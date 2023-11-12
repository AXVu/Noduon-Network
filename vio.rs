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

    // get_functions returns the 2d matrix of noduon activation functions
    pub fn get_functions(&self) -> Vec<Vec<AF>> {
        self.layers.iter().map(|x| x.get_functions()).collect()
    } // End get_functions

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
            outs.push(input.clone());
        }
        outs
    } // End forward_pass

    // backward_pass computes the gradients of each weight and returns them as a 3d matrix
    pub fn backward_pass(&self, forward_pass: Vec<Vec<f32>>, actual: Vec<f32>, loss: &str) -> Vec<Vec<Vec<f32>>> {

        let weights: Vec<Vec<Vec<f32>>> = self.get_weights();
        let functions: Vec<Vec<AF>> = self.get_functions();

        let mut gradients: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut derivatives: Vec<Vec<f32>> = Vec::new();

        let outputs = forward_pass.last().unwrap().clone();

        let out_loss_derivative: Vec<f32> = match loss {
            "mse" => {
                let mut derivates: Vec<f32> = vec![];
                for i in 0..outputs.len() {
                    derivates.push(outputs[i] - actual[i])
                }
                derivates
            },
            _ => {
                let derivates: Vec<f32> = outputs.iter().map(|_| 0.0).collect();
                derivates
            }
        };

        derivatives.push(out_loss_derivative);

        // Iterate through layers backwards, using index
        for i in (0..self.layers.len()).rev() {
            let mut layer_gradient: Vec<Vec<f32>> = Vec::new();
            let mut layer_derivatives: Vec<f32> = Vec::new();

            // Iterate through noduons in the layer
            for j in 0..self.layers[i].noduons.len() {
                let mut this_gradients: Vec<f32> = Vec::new();
                let fwd_connections: usize;

                let mut rld: f32 = 0.0;
                // If this layer is not the last, then calculate relative loss derivatives as normal
                if i != self.layers.len() - 1 {
                    // Calculate the rld as the weight of its forward connection times the derivative of that noduon. Summed

                    if i != self.layers.len() - 2 {
                        fwd_connections = weights[i + 1].len() - 1;
                    } else {
                        fwd_connections = weights[i + 1].len();
                    }

                    for k in 0..fwd_connections {
                        rld += derivatives[0][k] * weights[i + 1][k][j];
                    }

                    // Compute the value derivative for this noduon as its rld times the derivative of its activation function
                    let this_derivative: f32 = rld * activation_derivative(forward_pass[i + 1][j], &functions[i][j]);
                    // Push it to the layer derivatives
                    layer_derivatives.push(this_derivative);

                    // Calculate the weight gradients as this noduon's value derivative times the value of the backward connection
                    for k in 0..weights[i][j].len() {
                        this_gradients.push(this_derivative * forward_pass[i][k])
                    }
                } // End default branch
                else {  // This branch takes place on the last layer only
                    for k in 0..weights[i][j].len() {
                        this_gradients.push(derivatives[0][j] * forward_pass[i][k])
                    }
                } // End if branches. At this point, the vec this_gradients will be full of the weight gradients of this noduon
                layer_gradient.push(this_gradients);
            } // End layer. At this point, layer_gradient will have the layer gradients of each noduon in it.
            gradients.insert(0, layer_gradient);
            if i != self.layers.len() - 1 {
                derivatives.insert(0, layer_derivatives)
            }
        }
        // Return the filled gradients matrix
        gradients
    } // End backward_pass

    // train will perform a forward and backward pass, calculate the forward pass, perform the backward pass, and perform gradient desc
    pub fn train(&mut self, input: Vec<f32>, expected: Vec<f32>, loss: &str, learn_rate: f32) {
        let fwd_pass: Vec<Vec<f32>> = self.forward_pass(input);
        let mut gradients: Vec<Vec<Vec<f32>>> = self.backward_pass(fwd_pass, expected, loss);

        for layer in gradients.iter_mut() {
            for noduon in layer.iter_mut() {
                for weight in noduon.iter_mut() {
                    *weight *= -learn_rate
                }
            }
        }

        self.change_weights(gradients)
    } // End train

}

/// Builders ///

// build_neuron constructs a random neuron with a set number of weights and a specific function
pub fn build_neuron(num_weights: usize, function: &AF, scale: f32) -> Noduon {
    let mut rng: ThreadRng = rand::thread_rng();
    let weights: Vec<f32> = (0..num_weights).map(|_| rng.gen_range(-1.0..=1.0) / scale).collect();
    Noduon::Neuron(Nodule { weights, activation: function.clone() })
} // End build_neuron

// build_layer constructs a layer with a specific number of noduons and weights, with random weights and a single function
pub fn build_layer(num_noduons: usize, num_weights: usize, function: &AF, hidden: bool) -> Layer{
    let mut nods: Vec<Noduon> = (0..num_noduons).map(|_| build_neuron(num_weights, &function, num_weights as f32)).collect();
    if hidden {nods.push(Noduon::Bias)}
    Layer {noduons: nods}
} // End build_layer

// build_network builds a network based on the network_shape Vec, which will describe the number of noduons in each layer
pub fn build_network(num_inputs: usize, network_shape: Vec<usize>, gen_func: AF, out_func: AF) -> Network {
    // Inititalize the layers Vec
    let mut layers: Vec<Layer> = Vec::new();
    let num_layers: usize = network_shape.len();

    // Build the first layer
    layers.push(build_layer(network_shape[0], num_inputs + 1, &gen_func, true));

    // Build the hidden layers
    for i in 1..(num_layers - 1) {
        layers.push(build_layer(network_shape[i], network_shape[i-1] + 1, &gen_func, true))
    }

    // Build the output layer
    layers.push(build_layer(network_shape[num_layers - 1], network_shape[num_layers - 2] + 1, &out_func, false));
    
    Network {layers}
} // End build_network

/// Helpers ///

// activation_derivative takes in an activation function and returns its derivative
pub fn activation_derivative(num: f32, function: &AF) -> f32 {
    match function {
        AF::ReLU => {
            if num > 0.0 {
                return 1.0
            } else {
                return 0.0
            }
        },
        AF::Sigmoid => {
            (1.0 / (1.0 + (-num).exp())) * (1.0 - (1.0 / (1.0 + (-num).exp())))
        },
        AF::Tanh => {
            1.0 - num.tanh().powi(2)
        },
        _ => 1.0
    }
}