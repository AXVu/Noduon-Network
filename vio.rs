use rand::thread_rng;
use rand::{self, rngs::ThreadRng, Rng};
use rand_distr::{Distribution, Normal};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::{fs, io::Write, vec};

// The Activation Function enum to avoid string comparison
#[derive(Clone)]
pub enum AF {
    Linear,
    ReLU,
    LReLU(f32),
    Sigmoid,
    Tanh,
    NA,
    Softmax,
}

// The nodule struct, which is the fundamental structure in VIO
#[derive(Clone)]
pub struct Nodule {
    pub weights: Vec<f32>,
    pub activation: AF,
}

// A Noduon struct to contain both neurons and bias nodes
#[derive(Clone)]
pub enum Noduon {
    Neuron(Nodule),
    Embedding(Vec<f32>),
    Bias,
}

// Noduon functions
impl Noduon {
    // process function, which takes in the past_layer and performs the weighted sum against the noduon weights
    pub fn process(&self, past_layer: &Vec<f32>) -> f32 {
        match self {
            // neruon processing applies weighted sum and then activation function
            Noduon::Neuron(f) => {
                let mut sum: f32 = f
                    .weights
                    .iter()
                    .zip(past_layer.iter())
                    .fold(0.0, |acc, (&w, &p)| acc + w * p);
                // activation function
                sum = match f.activation {
                    AF::Tanh => sum.tanh(),
                    AF::ReLU => sum.max(0.0),
                    AF::LReLU(e) => sum.max(sum * e),
                    AF::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                    _ => 0.0,
                };

                sum
            }
            Noduon::Embedding(f) => f[past_layer[0] as usize],
            // Bias noduon only returns 1.0
            Noduon::Bias => 1.0,
        }
    } // End process

    // Set weights directly sets the weights of this Noduon. Ignores bias.
    pub fn set_weights(&mut self, new_weights: &Vec<f32>) {
        match self {
            // Set neuron weights. This requires a clone with to_vec()
            Noduon::Neuron(f) => f.weights = new_weights.to_vec(),
            _ => (),
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
                    }
                    None => print!("Fatal error in push_weight: pop failed"),
                };
            }
            // Bias has no weights an embedding neurons should never have more than 1
            _ => (),
        }
    } // End push_weight

    // change_weights takes a vector and changes each of the noduon's weights by the respective amount
    pub fn change_weights(&mut self, weight_changes: &Vec<f32>) {
        match self {
            Noduon::Neuron(f) => {
                for i in 0..f.weights.len() {
                    f.weights[i] += weight_changes[i]
                }
            }
            _ => (),
        }
    } // End change_weights

    pub fn change_weight_emb(&mut self, weight_change: &f32, inp: f32) {
        match self {
            Noduon::Embedding(e) => e[inp as usize] += weight_change,
            _ => (),
        }
    }

    // get_weights returns this Noduon's weights as a vec
    pub fn get_weights(&self) -> Vec<f32> {
        match self {
            Noduon::Neuron(f) => f.weights.clone(),
            _ => Vec::new(),
        }
    } // End get_weights

    // get_function returns this Noduon's AF
    pub fn get_function(&self) -> AF {
        match self {
            Noduon::Neuron(f) => f.activation.clone(),
            _ => AF::NA,
        }
    } // End get_function

    // randomize randomizes each of the weights in this noduon
    pub fn randomize(&mut self) {
        match self {
            Noduon::Neuron(f) => {
                let mut rng: ThreadRng = rand::thread_rng();
                f.weights = f
                    .weights
                    .iter()
                    .map(|_| rng.gen_range(-1.0..=1.0))
                    .collect();
            }
            _ => (),
        }
    }
}

/// Layers ///
// Layers contain a Vec of noduons
#[derive(Clone)]
pub struct Layer {
    pub noduons: Vec<Noduon>,
}

impl Layer {
    // process is the basic function of the layer, performing the weighted sum for each Noduon based on the input
    pub fn process(&self, input: &Vec<f32>) -> Vec<f32> {
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

    // change_weights_emb is specifically for networks with embedding layers
    pub fn change_weights_emb(&mut self, weight_changes: &Vec<Vec<f32>>, inp: f32) {
        for (n, w) in self.noduons.iter_mut().zip(weight_changes.iter()) {
            n.change_weight_emb(&w[0], inp)
        }
    }

    // get_weights returns the 2d matrix of weights
    pub fn get_weights(&self) -> Vec<Vec<f32>> {
        self.noduons.iter().map(|x| x.get_weights()).collect()
    } // End get_weights

    // get_functions returns the vec of functions
    pub fn get_functions(&self) -> Vec<AF> {
        self.noduons.iter().map(|x| x.get_function()).collect()
    } // End get_functions

    // get_size returns the number of noduons
    pub fn get_size(&self) -> usize {
        self.noduons.len()
    }

    pub fn get_num_inp(&self) -> usize {
        self.noduons[0].get_weights().len()
    }

    // randomize randomizes the weights of each Noduon in the layer
    pub fn randomize(&mut self) {
        self.noduons.iter_mut().for_each(|x| x.randomize())
    }
}

/// Network ///

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    // get_depth returns the number of layers
    pub fn get_depth(&self) -> usize {
        self.layers.len()
    }

    // process is the primary function of the network, taking in the inputs and pushing outputs
    pub fn process(&self, input: &Vec<f32>) -> Vec<f32> {
        if self.layers[0].get_num_inp() - 1 != input.len() {
            println!("Incorrect input shape");
            return Vec::new();
        }
        let mut inp = Vec::with_capacity(input.len() + 1);
        inp.extend(input);
        inp.push(1.0);
        for layer in self.layers.iter() {
            inp = layer.process(&inp)
        }
        inp
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

    // change_weights_emb is specifically for networks with embedding layers
    pub fn change_weights_emb(&mut self, weight_changes: Vec<Vec<Vec<f32>>>, inp: f32) {
        let mut first = true;
        for (layer, changes) in self.layers.iter_mut().zip(weight_changes.iter()) {
            if first {
                layer.change_weights_emb(changes, inp);
                first = false;
            } else {
                layer.change_weights(changes)
            }
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
        let new_weights: Vec<f32> = self.layers[0]
            .noduons
            .iter()
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        // Push it out
        self.layers[0].push_weights(new_weights)
    } // End add_input

    // add_output will push a new noduon with random weights to the last layer
    pub fn add_output(&mut self, function: AF) {
        // Start by generating a random set of weights based on the second to last layer
        let mut rng: ThreadRng = rand::thread_rng();
        let num_layers = self.layers.len();
        let new_weights: Vec<f32> = (0..self.layers[num_layers - 2].noduons.len())
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        // Now we append a noduon with these weights to the last layer
        self.layers[num_layers - 1]
            .noduons
            .push(Noduon::Neuron(Nodule {
                weights: (new_weights),
                activation: (function),
            }));
    } // End add_output

    // randomize will randomize each of the network's weights between -1.0..1.0
    pub fn randomize(&mut self) {
        self.layers.iter_mut().for_each(|x| x.randomize());
    } // End randomize

    // forward_pass will perform the process function, except it will return the output of each layer
    pub fn forward_pass(&self, input: &Vec<f32>) -> Vec<Vec<f32>> {
        if self.layers[0].get_num_inp() - 1 != input.len() {
            println!("Incorrect input shape");
            return Vec::new();
        }
        // add bias to input
        let mut inp = Vec::with_capacity(input.len() + 1);
        inp.extend(input);
        inp.push(1.0);
        // create the outs and add inputs to it
        let mut outs: Vec<Vec<f32>> = Vec::new();
        outs.push(inp.clone());
        // for each layer, process and add the result to the output
        for layer in self.layers.iter() {
            inp = layer.process(&inp);
            outs.push(inp.clone());
        }
        outs
    } // End forward_pass

    // backward_pass computes the gradients of each weight and returns them as a 3d matrix
    pub fn backward_pass(
        &self,
        forward_pass: Vec<Vec<f32>>,
        actual: &Vec<f32>,
        loss: &str,
    ) -> Vec<Vec<Vec<f32>>> {
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
            }
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
                    let this_derivative: f32 =
                        rld * activation_derivative(forward_pass[i + 1][j], &functions[i][j]);
                    // Push it to the layer derivatives
                    layer_derivatives.push(this_derivative);

                    // Calculate the weight gradients as this noduon's value derivative times the value of the backward connection
                    for k in 0..weights[i][j].len() {
                        this_gradients.push(this_derivative * forward_pass[i][k])
                    }
                }
                // End default branch
                else {
                    // This branch takes place on the last layer only
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
    pub fn train(&mut self, input: &Vec<f32>, expected: &Vec<f32>, loss: &str, learn_rate: f32) {
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

    // train_emb is specifically for networks with embedding layers
    pub fn train_emb(
        &mut self,
        input: &Vec<f32>,
        expected: &Vec<f32>,
        loss: &str,
        learn_rate: f32,
    ) {
        let fwd_pass: Vec<Vec<f32>> = self.forward_pass(input);
        let mut gradients: Vec<Vec<Vec<f32>>> = self.backward_pass(fwd_pass, expected, loss);

        for layer in gradients.iter_mut() {
            for noduon in layer.iter_mut() {
                for weight in noduon.iter_mut() {
                    *weight *= -learn_rate
                }
            }
        }

        self.change_weights_emb(gradients, input[0])
    } // End train

    // split_layers is used to separate a trained encoder into encode/decode layers or to cut off the head of a model.
    pub fn split_layers(&self, num_layers: usize) -> (Network, Network) {
        let first = Network {
            layers: self.layers.iter().take(num_layers).cloned().collect(),
        };
        let second = Network {
            layers: self.layers.iter().skip(num_layers).cloned().collect(),
        };
        (first, second)
    }

    pub fn batch_process(&self, inputs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.process(x)).collect()
    }

    pub fn batch_mse(
        &mut self,
        inputs: Vec<Vec<f32>>,
        expected: Vec<Vec<f32>>,
        loss: &str,
        learn_rate: &f32,
    ) {
        let _ = inputs
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| self.train(x, y, loss, *learn_rate));
    }
}

/// Builders ///

// build_neuron constructs a random neuron with a set number of weights and a specific function
pub fn build_neuron(num_weights: usize, function: &AF, scale: f32) -> Noduon {
    let mut rng: ThreadRng = rand::thread_rng();
    let normal = Normal::new(0.0, scale.powi(2)).unwrap();
    let weights: Vec<f32> = (0..num_weights).map(|_| normal.sample(&mut rng)).collect();
    Noduon::Neuron(Nodule {
        weights,
        activation: function.clone(),
    })
} // End build_neuron

// build_layer constructs a layer with a specific number of noduons and weights, with random weights and a single function
pub fn build_layer(num_noduons: usize, num_weights: usize, function: &AF, hidden: bool) -> Layer {
    let scale = match function {
        AF::ReLU => 2.0 / num_weights as f32,
        AF::Sigmoid => 1.0 / ((num_noduons + num_weights) as f32 / 2.0),
        AF::Tanh => 1.0 / ((num_noduons + num_weights) as f32 / 2.0),
        _ => 1.0 / ((num_noduons + num_weights) as f32 / 2.0),
    };
    let mut nods: Vec<Noduon> = (0..num_noduons)
        .map(|_| build_neuron(num_weights, &function, scale))
        .collect();
    if hidden {
        nods.push(Noduon::Bias)
    }
    Layer { noduons: nods }
} // End build_layer

// build_network builds a network based on the network_shape Vec, which will describe the number of noduons in each layer
pub fn build_network(
    num_inputs: usize,
    network_shape: Vec<usize>,
    gen_func: AF,
    out_func: AF,
) -> Network {
    // Inititalize the layers Vec
    let mut layers: Vec<Layer> = Vec::new();
    let num_layers: usize = network_shape.len();

    // Build the first layer
    layers.push(build_layer(
        network_shape[0],
        num_inputs + 1,
        &gen_func,
        true,
    ));

    // Build the hidden layers
    for i in 1..(num_layers - 1) {
        layers.push(build_layer(
            network_shape[i],
            network_shape[i - 1] + 1,
            &gen_func,
            true,
        ))
    }

    // Build the output layer
    layers.push(build_layer(
        network_shape[num_layers - 1],
        network_shape[num_layers - 2] + 1,
        &out_func,
        false,
    ));

    Network { layers }
} // End build_network

// network_from_weights builds a network from pre-generated weights
pub fn network_from_weights(
    weights: Vec<Vec<Vec<f32>>>,
    activation_internal: AF,
    activation_end: AF,
) -> Network {
    let mut layers: Vec<Layer> = Vec::new();
    let num_layers: usize = weights.len();

    for i in 0..num_layers {
        let layer_len: usize = weights[i].len();
        let mut layer: Vec<Noduon> = Vec::new();
        for j in 0..layer_len {
            if weights[i][j] != vec![] {
                if i != num_layers - 1 {
                    layer.push(Noduon::Neuron(Nodule {
                        weights: weights[i][j].clone(),
                        activation: activation_internal.clone(),
                    }))
                } else {
                    layer.push(Noduon::Neuron(Nodule {
                        weights: weights[i][j].clone(),
                        activation: activation_end.clone(),
                    }))
                }
            } else {
                layer.push(Noduon::Bias)
            }
        }
        layers.push(Layer { noduons: layer })
    }
    Network { layers }
}

pub fn build_embedding_layer(dict_size: usize, embedding_size: usize) -> Layer {
    let mut noduons = Vec::with_capacity(embedding_size);
    let mut rng = thread_rng();
    for _ in 0..embedding_size {
        noduons.push(Noduon::Embedding(
            (0..dict_size).map(|_| rng.gen_range(-1.0..=1.0)).collect(),
        ))
    }
    noduons.push(Noduon::Bias);
    Layer { noduons }
}

/// Helpers ///

pub fn zip_network(networks: Vec<Network>) -> Network {
    let mut layers = Vec::new();
    for network in networks.iter() {
        layers.extend(network.layers.clone())
    }
    Network { layers }
}

// activation_derivative takes in an activation function and returns its derivative
pub fn activation_derivative(num: f32, function: &AF) -> f32 {
    match function {
        AF::ReLU => {
            if num > 0.0 {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        AF::LReLU(e) => {
            if num > 0.0 {
                return 1.0;
            } else {
                return -e;
            }
        }
        AF::Sigmoid => (1.0 / (1.0 + (-num).exp())) * (1.0 - (1.0 / (1.0 + (-num).exp()))),
        AF::Tanh => 1.0 - num.tanh().powi(2),
        _ => 1.0,
    }
}

///// Mutation and children /////

// generate_children generates x new networks with mutations
pub fn generate_children(
    network: Network,
    num_children: usize,
    mut_rate: f64,
    mut_degree: f32,
) -> Vec<Network> {
    let mut rng: ThreadRng = rand::thread_rng();
    let weights: Vec<Vec<Vec<f32>>> = network.get_weights();
    let mut new_networks: Vec<Network> = Vec::new();

    for _ in 0..num_children {
        let muts: Vec<Vec<Vec<f32>>> = weights
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|noduon| {
                        noduon
                            .iter()
                            .map(|_| {
                                if rng.gen_bool(mut_rate) {
                                    rng.gen_range(-mut_degree..=mut_degree)
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let mut new: Network = network.clone();
        new.change_weights(muts);
        new_networks.push(new);
    }

    new_networks
} // End generate_children

// generate_child generates a new network with mutations
pub fn generate_child(parent: &Network, mut_rate: f64, mut_degree: f32) -> Network {
    let mut rng: ThreadRng = rand::thread_rng();
    let weights: Vec<Vec<Vec<f32>>> = parent.get_weights();

    let muts: Vec<Vec<Vec<f32>>> = weights
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|noduon| {
                    noduon
                        .iter()
                        .map(|_| {
                            if rng.gen_bool(mut_rate) {
                                rng.gen_range(-mut_degree..=mut_degree)
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut new: Network = parent.clone();
    new.change_weights(muts);

    new
} // End generate_children

// generation_turnover takes in an old generation along with fitness scores and turns it over to a new one of the same size
pub fn generation_turnover(
    networks: Vec<Network>,
    scores: Vec<f32>,
    cut: usize,
    cut_below: f32,
    elites: usize,
    direct_children: usize,
    mut_rate: f64,
    mut_degree: f32,
) -> Vec<Network> {
    let shaped = networks[0].clone();
    let size: usize = networks.len();

    let mut new_networks: Vec<Network> = Vec::new();

    let mut ordered: Vec<(&Network, &f32)> = networks.iter().zip(scores.iter()).collect();
    ordered.sort_by(|a: &(&Network, &f32), b: &(&Network, &f32)| {
        b.1.partial_cmp(&a.1).expect("failed to unwrap partialcmp")
    });

    ordered = ordered[0..(size - cut)].to_vec();

    let parents: Vec<Network> = ordered
        .iter()
        .filter(|x| x.1 >= &cut_below)
        .map(|x| x.0.clone())
        .collect();

    let num_parents = parents.len();

    // Begin building new generation by preserving the elites
    for i in 0..elites.min(num_parents) {
        new_networks.push(parents[i].clone())
    }

    let mut muta_children: usize = 0;

    // Add children to the new generation, iterating
    if num_parents != 0 {
        while muta_children < direct_children {
            let target: usize = muta_children % num_parents;
            new_networks.push(generate_child(&parents[target], mut_rate, mut_degree));
            muta_children += 1
        }
    }

    let det_children: usize = new_networks.len();

    // Fill with randomized networks
    for _ in det_children..size {
        let mut child = shaped.clone();
        child.randomize();
        new_networks.push(child)
    }

    new_networks
}

///// TO AND FROM CSV /////

// Converts the weights of a network into a txt with a name specified by the file_name parameter
pub fn network_to_txt(network: Network, file_name: String) -> Result<(), Box<dyn Error>> {
    // Check if there is a file, if there is remove it before creating a new one
    if Path::new(&(file_name.clone() + &String::from(".txt"))).exists() {
        fs::remove_file(file_name.clone() + &String::from(".txt"))?;
    }

    // Create new file on the given path, and initialize the weights and shape of the network
    let mut file = File::create(file_name + &String::from(".txt"))?;
    let weights = network.get_weights();
    let shape: Vec<String> = network
        .layers
        .iter()
        .map(|x| x.get_size().to_string())
        .collect();

    // Write in data
    writeln!(file, "{}", shape.join(","))?;

    for layer in weights {
        for noduon in layer {
            writeln!(
                file,
                "{}",
                noduon
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            )?;
        }
    }

    Ok(())
}

pub fn network_from_txt(
    file_name: String,
    activation_internal: AF,
    activation_end: AF,
) -> Result<Network, Box<dyn Error>> {
    let path = file_name + &String::from(".txt");
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut i: usize = 0;
    let mut j: usize = 0;
    let mut layer_num: usize = 0;
    let mut new_weights: Vec<Vec<Vec<f32>>> = vec![vec![]];
    let mut num_shape: Vec<usize> = Vec::new();

    for line in reader.lines() {
        if i == 0 {
            num_shape = line?
                .split(',')
                .map(|x| x.parse::<usize>().unwrap())
                .collect();
        } else {
            let set = line?;
            if j == num_shape[layer_num] {
                new_weights.push(vec![]);
                j = 0;
                layer_num += 1;
            }
            if set.clone().len() != 0 {
                let nod_weights: Vec<f32> =
                    set.split(',').map(|x| x.parse::<f32>().unwrap()).collect();
                new_weights[layer_num].push(nod_weights);
            } else {
                new_weights[layer_num].push(vec![]);
            }
            j += 1;
        }
        i += 1;
    }
    Ok(network_from_weights(
        new_weights,
        activation_internal,
        activation_end,
    ))
}

//// Optimizers ////
pub struct Adam {
    pub m1: Vec<Vec<Vec<f32>>>,
    pub m2: Vec<Vec<Vec<f32>>>,
    pub b1: f32,
    pub b2: f32,
    pub step: i32,
    pub learning_rate: f32,
}

impl Adam {
    pub fn get_weight_changes(&mut self, backward_pass: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let mut changes = backward_pass.clone();
        // Update estimates
        for l in 0..self.m1.len() {
            for n in 0..self.m1[l].len() {
                for w in 0..self.m1[l][n].len() {
                    // Moment update
                    self.m1[l][n][w] = (self.b1 * self.m1[l][n][w]
                        + (1.0 - self.b1) * backward_pass[l][n][w])
                        / (1.0 - self.b1.powi(self.step));
                    self.m2[l][n][w] = (self.b2 * self.m2[l][n][w]
                        + (1.0 - self.b2) * backward_pass[l][n][w].powi(2))
                        / (1.0 - self.b2.powi(self.step));
                    // Calculate weight change
                    changes[l][n][w] = -self.learning_rate * self.m1[l][n][w]
                        / (self.m2[l][n][w].sqrt() + 0.0000001)
                }
            }
        }
        changes
    }

    pub fn train_network(
        &mut self,
        network: &mut Network,
        inputs: &Vec<Vec<f32>>,
        expected: &Vec<Vec<f32>>,
        loss: &str,
    ) {
        if network.layers[0].get_num_inp() - 1 != inputs[0].len() {
            println!(
                "Adam gets wrong input size, expected {} got {}",
                network.layers[0].get_num_inp(),
                inputs[0].len()
            )
        }
        let mut g = network.get_weights();
        let backwards: Vec<Vec<Vec<Vec<f32>>>> = inputs
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| network.backward_pass(network.forward_pass(x), y, loss))
            .collect();
        let num_samples = backwards.len();
        for l in 0..g.len() {
            for n in 0..g[l].len() {
                for w in 0..g[l][n].len() {
                    g[l][n][w] =
                        backwards.iter().map(|b| b[l][n][w]).sum::<f32>() / num_samples as f32;
                }
            }
        }
        network.change_weights(self.get_weight_changes(&g));
        self.step += 1
    }

    // add_input will push a new random weight to each noduon in the first layer
    pub fn add_input(&mut self) {
        for n in 0..self.m1[0].len() {
            match self.m1[0][n].pop() {
                Some(b) => {
                    self.m1[0][n].push(0.0);
                    self.m1[0][n].push(b)
                }
                _ => {
                    println!("Adam add_input failed, no bias term")
                }
            }
            match self.m2[0][n].pop() {
                Some(b) => {
                    self.m2[0][n].push(0.0);
                    self.m2[0][n].push(b)
                }
                _ => {
                    println!("Adam add_input failed, no bias term")
                }
            }
        }
    } // End add_input

    // expand_emb adds an additional weight to each embedding noduon in the first section
    pub fn expand_emb(&mut self) {
        let mut rng = thread_rng();
        for i in 0..self.m1[0].len() {
            self.m1[0][i].push(rng.gen_range(-1.0..=1.0))
        }
    }

    pub fn train_network_emb(
        &mut self,
        network: &mut Network,
        inputs: &Vec<Vec<f32>>,
        expected: &Vec<Vec<f32>>,
        loss: &str,
    ) {
        if inputs[0].len() != 1 {
            println!(
                "Adam gets wrong input size, expected 1 got {}",
                inputs[0].len()
            )
        }
        for (i, e) in inputs.iter().zip(expected.iter()) {
            let mut forward = network.forward_pass(i);
            forward[0][0] = 1.0;
            let changes = self.get_weight_changes_emb(&network.backward_pass(forward, e, loss), i[0]);
            network.change_weights_emb(changes, i[0])
        }
    }
    
    pub fn get_weight_changes_emb(&mut self, backward_pass: &Vec<Vec<Vec<f32>>>, inp: f32) -> Vec<Vec<Vec<f32>>> {
        let mut changes = backward_pass.clone();
        // Embedding layer
        for n in 0..self.m1[0].len() {
            // Moment update
            self.m1[0][n][inp as usize] = (self.b1 * self.m1[0][n][inp as usize]
                + (1.0 - self.b1) * backward_pass[0][n][inp as usize])
                / (1.0 - self.b1.powi(self.step));
            self.m2[0][n][inp as usize] = (self.b2 * self.m2[0][n][inp as usize]
                + (1.0 - self.b2) * backward_pass[0][n][inp as usize].powi(2))
                / (1.0 - self.b2.powi(self.step));
            // Calculate weight change
            changes[0][n][inp as usize] = -self.learning_rate * self.m1[0][n][inp as usize]
                / (self.m2[0][n][inp as usize].sqrt() + 0.0000001)
        }
        // Update estimates for other layers
        for l in 1..self.m1.len() {
            for n in 0..self.m1[l].len() {
                for w in 0..self.m1[l][n].len() {
                    // Moment update
                    self.m1[l][n][w] = (self.b1 * self.m1[l][n][w]
                        + (1.0 - self.b1) * backward_pass[l][n][w])
                        / (1.0 - self.b1.powi(self.step));
                    self.m2[l][n][w] = (self.b2 * self.m2[l][n][w]
                        + (1.0 - self.b2) * backward_pass[l][n][w].powi(2))
                        / (1.0 - self.b2.powi(self.step));
                    // Calculate weight change
                    changes[l][n][w] = -self.learning_rate * self.m1[l][n][w]
                        / (self.m2[l][n][w].sqrt() + 0.0000001)
                }
            }
        }
        changes
    }
}

pub fn construct_adam(weights: &Vec<Vec<Vec<f32>>>, b1: f32, b2: f32, learning_rate: f32) -> Adam {
    let mut m = weights.clone();
    for layer in 0..m.len() {
        for noduon in 0..m[layer].len() {
            for weight in 0..m[layer][noduon].len() {
                m[layer][noduon][weight] = 0.0
            }
        }
    }
    Adam {
        m1: m.clone(),
        m2: m.clone(),
        b1,
        b2,
        step: 1,
        learning_rate,
    }
}
