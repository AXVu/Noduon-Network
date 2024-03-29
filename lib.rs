/*
Documentation:
The basic structure is the Noduon, which has the weighted sum method.
There are three basic Noduon types.
Input Noduons, which have a value set externally. Its result method returns this value. These are named.
Inner Noduons, of which there are a couple types, connect to any number of other inner noduons or input noduons.
Its result method returns the values of those noduons multiplied by its respective weight, along with an activation function.
Output Noduons, which are exactly the same as Inner noduons except they are named.

Detailed and Methods:

Noduons methods are implemented in the Noduon enum
set_weights(&mut self, new_weights, mutate, mut_rate, mut_degree) -> Checks if new_weights is the same size as the old weights and replaces them.
Input/Bias->N/a. If mutate is true, has a chance to mutate each weight by a number in +-mut_degree
set_connections(&mut self, new_connections, mutate, mut_rate) -> checks if new_connections is the same size as old connections and replaces them. Input/Bias->N/A.
If mutate is true, has a chance to mutate each connection to either true or false with equal chance regardless of original status.
set_value(&mut self, new_value) -> Sets a new value for the input node. Does nothing for other types.
change_weights(&mut self, weight_changes) -> Changes the weights of this noduon by the values in a given vector
get_weights(&self) -> Returns the current weights of the given Noduon. Input/Bias -> empty vector
get_connections(&self) -> Returns the current connections of the given Noduon. Input/Bias -> empty vector
get_type(&self) -> Returns the noduon type as a string ("input", "inner1d",etc)
result(&self, past_layer) -> Returns the sum of each value of the input vector times the connected weights, passed through an activation
function. Input -> returns value. Bias -> returns 1
randomize(&mut self) -> re-randomizes the weights of this noduon

Helper:
activation_derivative(num, function) -> outputs the derivative of the specified activation function


**********

The next structure in the hierarchy is the layer.
Layers contain any number of noduons of various types. (saved in self.noduons)
Layers have functions that will take an input and generate the output vector resulting from each of its noduon's calculations.
Layers will have "presets" that can be used to generate different layer types.
As of now, there is only 1 kind of layer, but this may increase in the future.
ALWAYS put bias nodes LAST in input layers, builder functions do this by default. Otherwise setter function will not work

Layer methods are implemented in the Layer enum
get_size(&self) -> Returns the number of Noduons in the layer
get_types(&self) -> Returns the types of the noduons in this layer as a vector
get_weights(&self) -> Returns the weights of the noduons in this layer as a matrix of weight values
get_connections(&self) -> Returns the weights of the noduons in this layer as a matrix of boolean values
set_values(&mut self, values) -> Sets the values of inputs, starting at the top
set_connections(&mut self, new_connections, mutate, mut_rate) -> Calls set_connections for each noduon in the layer, using the input matrix
set_weights(&mut self, new_weights, mutate, mut_rate) -> Calls set_weights for each noduon in the layer, using the input matrix
change_weights(&mut self, weight_changes) -> Calls the change_weights function for each noduon in the layer
add_noduon(&mut self, noduon) -> Adds an already-built Noduon of any type to the layer
add_dense_noduon(&mut self, previous_layer_noduons, function) -> Builds a noduon fully connected to the previous layer, with the specified activation function
add_dense_output_noduon(&mut self, previous_layer_noduons, fucntion) -> Builds a noduon fully connected to the previous layer, with the specified activation function and name
process(&self, past_layer) -> Takes the result of a previous layer as a vector, outputting the results of this layer as a vector.

***********
The next structure in the hierarchy is the Network.
Networks contain any number of layers of various (only one right now) types. (saved in self.layers)
Networks have functions that will take inputs and generate the expected output
Networks will have "presets" that can be used to generate Networks of specific structure
Networks have a "target" trait that will focus on a certain layer, and only move forwards
As of now, there is only 1 kind of network (feed-forward), but this may change in the future.
As of now, Inputs will only work in the first layer, and outputs will only work in the last layer

Network methods are implemented directly in the struct
add_empty_layer(&mut self) -> Adds an empty layer with no noduons to the network
add_noduon(&mut self, noduon) -> Adds the specified noduon to the target layer
add_dense_noduon(&mut self, function) -> Adds a fully connected noduon to the target layer, inferring the number of connections by the size of that layer
lock_layer(&mut self) -> Moves target forward 1, stopping modification of the previous layer and allowing it for the next.
add_layer(&mut self, layer) -> Adds the specified layer to the network
add_input_layer(&mut self, num_inputs) -> Adds a layer with the specified number of input noduons
add_dense_layer(&mut self, num_noduons, function) -> Adds a dense layer to the network with num_noduons Noduons and the specified function. Moves the target to this layer.
add_dense_output_layer(&mut self, num_outputs, output_names, function) -> Adds a dense layer of output noduons with the specified names and function. Moves the target to this layer.
update_network(&mut self) -> Updates the num_inputs and num_outputs parameters to reflect the first and last layers of the network.
get_weights(&self) -> Returns the weights of noduons as a 3d matrix. Rows are layers, columns are noduons, elements are weights.
get_types(&self) -> Returns the types of noduons as a matrix. Rows are layers, columns are noduon types as strings.
get_connections(&self) -> Returns the truth value of noduon connections as a 3d matrix. rows are layers, columns are noduons, elements are weights.
get_functions(&self) -> Returns the functions of each noduon in 2d matrix form.
set_weights(&mut self, new_weights, mutate, mut_rate, mut_degree) -> Changes the weights to the specifed ones. If mutate mutate by mut_degee with freq mut_rate
change_weights(&mut self, weight_changes) -> Changes the weights of the matrix with the corresponding 3d matrix
set_connections(&mut self, mutate, mut_rate) -> Sets the connections of the noduons with the corresponding 3d matrix. Has a chance equal to half the mut_rate to change each connection.
process(&mut self, inputs) -> With the given inputs, do a full process through each layer and return the output of the network.
forward_pass(&mut self, inputs) -> Returns the full matrix of the output of each layer in sequence
loss(&mut self, predicted, actual, function) -> Returns the loss between the predicted and actual value
backward_pass(&mut self, forward_pass, actual, loss_function) -> Returns the gradient of each connection based on the forward_pass.
model_to_txt(&self, model_name) -> Produces three files. One with the weights, connections, and types.
randomize(&mut self) -> Randomize all of the weights in the network.
*/

use std::{vec, fs, io};
use std::error::Error;
use rand::Rng;
use std::io::BufRead;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use std::arch::x86_64::*;
use std::thread;

// Activation function enum
#[derive(Clone)]
pub enum AF {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    NA
}

// Noduons that perform no calculations. Instead, they simply store the input value
#[derive(Clone)]
pub struct InputNoduon {
    value: f32
}

#[derive(Clone)]
// A 1D Inner Noduon only connects to noduons on one dimension
pub struct InnerNoduon1D {
    connections: Vec<bool>,
    weights: Vec<f32>,
    function: AF
}

// Output Noduons perform exactly the same as Inner Noduons
#[derive(Clone)]
pub struct OutputNoduon1D {
    connections: Vec<bool>,
    weights: Vec<f32>,
    function: AF
}

// Define the Noduon enum with Input, Inner1D, Output1D, and Bias noduons
#[derive(Clone)]
pub enum Noduon {

    Input(InputNoduon),
    Inner1D(InnerNoduon1D),
    Output1D(OutputNoduon1D),
    Bias

}

impl Noduon {
    // Set weights for non-inputs, will do nothing if applied to input
    pub fn set_weights(&mut self, new_weights: &Vec<f32>, mutate: bool, mut_rate: f32, mut_degree: f32) {
        // Check whether mutation should occur
        if mutate {
            // Use a match statement in order to determine what should be done with the weights
            match self {
                // Input and Bias noduons have no weights, so attempting to set weights on them will fail
                Noduon::Input(_) | Noduon::Bias => {},
                // Inner noduons will have their weights set as long as the vector is the proper length
                Noduon::Inner1D(f) => {
                    let mut rng = rand::thread_rng();
                    // Check for proper vector length
                    if new_weights.len() == f.weights.len() {
                        // Iterate through this Noduon's weights
                        for i in 0..new_weights.len() {
                            // Set its weights to the new value
                            f.weights[i] = new_weights[i];
                            // Pick a random number compared against mut rate to decide whether to mutate
                            if rng.gen_range(0.0..1.0) < mut_rate{
                                // Change the weight by a random float between - and + mut_degree
                                f.weights[i] = f.weights[i] + rng.gen_range(-mut_degree..=mut_degree);
                            }
                        }
                    }
                },
                // Output Noduons will have their weights set as long as the vector is the proper length
                Noduon::Output1D(f) => {
                    let mut rng = rand::thread_rng();
                    // Check for proper vector length
                    if new_weights.len() == f.weights.len() {
                        // Iterate through this Noduon's weights
                        for i in 0..new_weights.len() {
                            // Set its weights to the new value
                            f.weights[i] = new_weights[i];
                            // Pick a random number compared against mut rate to decide whether to mutate
                            if rng.gen_range(0.0..1.0) < mut_rate{
                                // Change the weight by a random float between - and + mut_degree
                                f.weights[i] = f.weights[i] + rng.gen_range(-mut_degree..=mut_degree);
                            }
                        }
                    }
                }
            }
        } else {
            // Use a match statement in order to determine what should be done with the weights
            match self {
                // Input and Bias noduons have no weights, so attempting to set weights on them will fail
                Noduon::Input(_) | Noduon::Bias => {},
                // Inner noduons will have their weights set as long as the vector is the proper length
                Noduon::Inner1D(f) => {
                    // Check for proper length
                    if new_weights.len() == f.weights.len() {
                        f.weights = new_weights.to_vec();
                    }
                },
                // Output noduons will have their weights set as long as the vector is the proper length
                Noduon::Output1D(f) => {
                    if new_weights.len() == f.weights.len() {
                        f.weights = new_weights.to_vec();
                    }
                }
            }
        }


    }

    // Accepts a vector of weight changes and applies them to this Noduon's weights
    pub fn change_weights(&mut self, weight_changes: &Vec<f32>) {
        // Use match statement to determine what should be done
        match self {
            // Ignore Bias and input noduons
            Noduon::Input(_) | Noduon::Bias => {},
            // Change the weights for inner noduon by the specified amount
            Noduon::Inner1D(f) => {
                if weight_changes.len() == f.weights.len() {
                    // Iterate through weights
                    for i in 0..weight_changes.len() {
                        f.weights[i] += weight_changes[i]
                    }
                }
            },
            // Change the weights for output noduon by the specified amount
            Noduon::Output1D(f) => {
                if weight_changes.len() == f.weights.len() {
                    // Iterate through weights
                    for i in 0..weight_changes.len() {
                        f.weights[i] += weight_changes[i]
                    }
                }
            }
        }
    }

    // Sets connections for non-inputs, will do nothing if applied to input
    pub fn set_connections(&mut self, new_connections: &Vec<bool>, mutate: bool, mut_rate: f32) {
        // Check for mutation
        if mutate {
            // Match statement to determine what happens
            match self {
                // Ignore Input and Bias noduons
                Noduon::Input(_) | Noduon::Bias => {},
                // Connections will set to the new ones
                Noduon::Inner1D(f) => {
                    let mut rng = rand::thread_rng();
                    if new_connections.len() == f.connections.len() {
                        // Iterate through connections
                        for i in 0..new_connections.len() {
                            f.connections[i] = new_connections[i];
                            // If mutate is determined, will set to random connection (not guaranteed to change)
                            if rng.gen_range(0.0..1.0) < mut_rate{
                                f.connections[i] = rng.gen_bool(0.5);
                            }
                        }
                    }
                },
                // Connections will set to the new ones
                Noduon::Output1D(f) => {
                    let mut rng = rand::thread_rng();
                    if new_connections.len() == f.connections.len() {
                        // Iterate through connections
                        for i in 0..new_connections.len() {
                            f.connections[i] = new_connections[i];
                            // If mutate is determined, will set to random connection (not guaranteed to change)
                            if rng.gen_range(0.0..1.0) < mut_rate{
                                f.connections[i] = rng.gen_bool(0.5);
                            }
                        }
                    }
                }
            }
        } else {
            // Match statement to determine what happens if no mutation
            match self {
                // Ignore Input and Bias noduons
                Noduon::Input(_) | Noduon::Bias => {},
                // Connections are set to the new ones if vector length is correct
                Noduon::Inner1D(f) => {
                    if new_connections.len() == f.connections.len() {
                        f.connections = new_connections.to_vec();
                    }
                },
                // Connections are set to the new ones if vector length is correct
                Noduon::Output1D(f) => {
                    if new_connections.len() == f.connections.len() {
                        f.connections = new_connections.to_vec();
                    }
                }
            }
        }
    }

    // Set the input value. Doesn't do anything for other types
    pub fn set_value(&mut self, new_value: f32) {
        match self {
            Noduon::Input(f) => f.value = new_value,
            _ => ()
        }
    }

    // Return the noduon type as a string
    pub fn get_type(&self) -> String{
        match self {
            Noduon::Input(_) => String::from("input"),
            Noduon::Inner1D(_) => String::from("inner1d"),
            Noduon::Output1D(_) => String::from("output1d"),
            Noduon::Bias => String::from("bias")
        }
    }

    // Gets weights for non-inputs, will return an empty vector if applied to input
    pub fn get_weights(&self) -> Vec<f32> {
        match self {
            Noduon::Input(_) | Noduon::Bias => vec![],
            Noduon::Inner1D(f) => f.weights.clone(),
            Noduon::Output1D(f) => f.weights.clone(),
        }
    }

    // Gets connections for non-inputs, will return an empty vector if allplied to input
    pub fn get_connections(&self) -> Vec<bool> {
        match self {
            Noduon::Input(_) | Noduon::Bias => vec![],
            Noduon::Inner1D(f) => f.connections.clone(),
            Noduon::Output1D(f) => f.connections.clone()
        }
    }

    // Returns the activation function of this Noduon
    pub fn get_function(&self) -> AF {
        match self {
            Noduon::Input(_) => AF::NA,
            Noduon::Inner1D(f) => f.function.clone(),
            Noduon::Output1D(f) => f.function.clone(),
            Noduon::Bias => AF::NA
        }
    }
    
    // Randomizes the weights of this Noduon
    pub fn randomize(&mut self) {
        match self {
            // Does nothing on Bias or Input noduons
            Noduon::Bias | Noduon::Input(_) => (),
            // Randomizes weights between -1 and +1
            Noduon::Inner1D(f) => {
                let mut rng = rand::thread_rng();
                for i in 0..f.weights.len() {
                    f.weights[i] = rng.gen_range(-1.0..=1.0);
                }
            },
            // Randomizes weights between -1 and +1
            Noduon::Output1D(f) => {
                let mut rng = rand::thread_rng();
                for i in 0..f.weights.len() {
                    f.weights[i] = rng.gen_range(-1.0..=1.0)
                }
            }
        }
    }

    // Passing in a previous layer's output vector, returns the output of a noduon. Returns the value of an input noduon
    pub fn result(&self, past_layer: &Vec<f32>) -> f32 {
        match self {
            // Input noduons return their value
            Noduon::Input(f) => f.value,
            // Bias noduons return 1
            Noduon::Bias => 1.0,
            // Inner noduons return the respective weighted sum
            Noduon::Inner1D(f) => {
                let mut total: f32 = 0.0;
                // Iterate through
                for connect in 0..f.connections.len() {
                    if f.connections[connect] {
                        total += past_layer[connect] * f.weights[connect];
                    }
                }

                // Match activation function and apply to total
                total = match f.function {
                    AF::Tanh => total.tanh(),
                    AF::ReLU => {
                        if total > 0.0 {
                            total
                        } else {
                            0.0
                        }},
                    AF::Sigmoid => 1.0 / (1.0 + (-total).exp()),
                    _ => 0.0
                 };
                 total
            },
            // Output noduons return their weighted sum
            Noduon::Output1D(f) => {
                let mut total: f32 = 0.0;
                // Iterate through
                for connect in 0..f.connections.len() {
                    if f.connections[connect] {
                        total += past_layer[connect] * f.weights[connect];
                    }
                }

                // Match activation function and apply to total
                total = match f.function {
                    AF::Tanh => total.tanh(),
                    AF::ReLU => {
                        if total > 0.0 {
                            total
                        } else {
                            0.0
                        }},
                    AF::Sigmoid => 1.0 / (1.0 + (-total).exp()),
                    _ => total
                 };
                total
            }
        }
    }


}


// Helper function for gradient
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

// End of Noduon section
////////////////////////////////////
// Start of Layer section

// Layers are collections of noduons. 1D layers are 1 dimensional arrays of noduons
#[derive(Clone)]
pub struct Layer1D {
    noduons: Vec<Noduon>
}

// Layers are different ways to collect noduons
#[derive(Clone)]
pub enum Layer {
    Standard(Layer1D)
}

impl Layer {

    // An advanced processing to optimize time taken, using vector-matrix math and multithreading
    pub fn simd_process(&self, past_layer: &Vec<f32>) -> Vec<f32> {
        let weights = self.get_weights();
        let funcs = self.get_functions();
        let mut result: Vec<f32> = vec![];
        // Iterate through rows, doing weighted sums
        for row in 0..weights.len() {
            if weights[row].len() == 0 {
                result.push(1.0);
            } else {
                let mut total = simd_weighted_sum(&weights[row], past_layer);
                total = match funcs[row] {
                    AF::Tanh => total.tanh(),
                    AF::ReLU => {
                        if total > 0.0 {
                            total
                        } else {
                            0.0
                        }},
                    AF::Sigmoid => 1.0 / (1.0 + (-total).exp()),
                    _ => total};
                result.push(total)
            }
        }
        return result;
    }

    // The standard process of a layer, taking the previous layer's output and outputting the new layer
    pub fn process(&self, past_layer: &Vec<f32>) -> Vec<f32> {
        match self {
            Layer::Standard(f) => {

                let mut result: Vec<f32> = vec![];

                // Iterate through the noduons in the layer, passing the past_layer
                for noduon in &f.noduons {
                    result.push(noduon.result(&past_layer))
                }

                result

            }
        }
    }

    // Returns Noduon types as a vector
    pub fn get_types(&self) -> Vec<String> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_type()).collect()
        }
    }

    // Returns the weights as a matrix. Rows are noduons, columns are the previous layer's noduons
    pub fn get_weights(&self) -> Vec<Vec<f32>> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_weights()).collect()
        }
    }

    // Returns the connections of each Noduon as a matrix. Empty lists are either Input or Bias noduons.
    pub fn get_connections(&self) -> Vec<Vec<bool>> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_connections()).collect()
        }
    }

    // Returns the activation functions of each Nouon in layer as a vector
    pub fn get_functions(&self) -> Vec<AF> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_function()).collect()
        }
    }

    // Sets the value of each input noduon
    pub fn set_values(&mut self, values: &Vec<f32>) {
        match self {
            Layer::Standard(f) => {
                for i in 0..values.len() {
                    f.noduons[i].set_value(values[i]);
                }
            }
        }
    }

    // Sets the weights of each noduon in a given layer
    pub fn set_weights(&mut self, new_weights: &Vec<Vec<f32>>, mutate: bool, mut_rate: f32, mut_degree: f32) {
        match self {
            Layer::Standard(f) => {
                for i in 0..f.noduons.len() {
                    f.noduons[i].set_weights(&new_weights[i], mutate, mut_rate, mut_degree);
                }
            }
        }
    }

    // Changes the weights of each noduon in this layer
    pub fn change_weights(&mut self, weight_changes: &Vec<Vec<f32>>) {
        match self {
            Layer::Standard(f) => {
                for i in 0..f.noduons.len() {
                    f.noduons[i].change_weights(&weight_changes[i])
                }
            }
        }
    }

    // Sets the connections of each noduon in a given layer
    pub fn set_connections(&mut self, new_connections: &Vec<Vec<bool>>, mutate: bool, mut_rate: f32) {
        match self {
            Layer::Standard(f) => {
                for i in 0..f.noduons.len() {
                    f.noduons[i].set_connections(&new_connections[i], mutate, mut_rate);
                }
            }
        }
    }

    // Adds the given noduon to the layer
    pub fn add_noduon(&mut self, noduon: Noduon) {
        match self {
            Layer::Standard(f) => {
                f.noduons.push(noduon)
            }
        }
    }

    // Adds a fully connected noduon to the layer
    pub fn add_dense_noduon(&mut self, previous_layer_noduons: usize, function: AF) {
        match self {
            Layer::Standard(f) => {

                let mut rng = rand::thread_rng();
                let weight: Vec<f32> = (0..previous_layer_noduons).map(|_| rng.gen_range(-1.0..=1.0)).collect();
                let connects: Vec<bool> = (0..previous_layer_noduons).map(|_| true).collect();
                let new_noduon: Noduon = Noduon::Inner1D(InnerNoduon1D { connections: connects, weights: weight, function: function });
                f.noduons.push(new_noduon);
            }
        }
    }

    // Adds a fully connected output noduon to the layer
    pub fn add_dense_output_noduon(&mut self, previous_layer_noduons: usize, function: AF) {
        match self {
            Layer::Standard(f) => {

                let mut rng = rand::thread_rng();
                let weight: Vec<f32> = (0..previous_layer_noduons).map(|_| rng.gen_range(-1.0..=1.0)).collect();
                let connects: Vec<bool> = (0..previous_layer_noduons).map(|_| true).collect();
                let new_noduon: Noduon = Noduon::Output1D(OutputNoduon1D{ connections: connects, weights: weight, function: function });
                f.noduons.push(new_noduon);
            }
        }
    }

    // Randomize the weights of each noduon in the layer
    pub fn randomize(&mut self) {
        match self {
            Layer::Standard(f) => {
                for noduon in 0..f.noduons.len() {
                    f.noduons[noduon].randomize();
                }
            }
        }
    }


    // Return the number of noduons in the layer
    pub fn get_size(&self) -> usize{
        match self {
            Layer::Standard(f) => f.noduons.len()
        }
    }

}

// A helper function to do each vector's multiplication
pub fn simd_weighted_sum(weights: &Vec<f32>, vector: &Vec<f32>) -> f32{
    let vals = weights.len();
    let chunks = vals / 8;
    let mut total_sum: f32 = 0.0;
    let w_ptr = weights.as_ptr();
    let v_ptr = vector.as_ptr();
    // Unsafe block for SIMD operations
    unsafe {
        for chunk in 0..chunks {
            // This line of code performs the multiplication
            let sum = _mm256_mul_ps(_mm256_loadu_ps(w_ptr.add(chunk * 8)), _mm256_loadu_ps(v_ptr.add(chunk * 8)));
            // This line starts the sum
            let sum = _mm256_hadd_ps(sum, sum);
            // Take first 4 elements
            let low = _mm256_extractf128_ps(sum, 0);

            let mut res = [0.0_f32; 4];
            _mm_storeu_ps(res.as_mut_ptr(), low);

            // Add the result into the total sum
            total_sum += res.iter().sum::<f32>();
        }
        // Handle the extra values
        for i in (chunks * 8)..vals {
            total_sum += weights[i] * vector[i]
        }
        total_sum
    }
}

// End Layer
/////////////////////////////////////////////////
// Start Network

#[derive(Clone)]
pub struct Network {
    layers: Vec<Layer>,
    num_inputs: usize,
    num_outputs: usize,
    target: usize
}

impl Network {
    
    // Add layer to network
    pub fn add_empty_layer(&mut self) {
        self.layers.push(Layer::Standard(Layer1D { noduons: vec![] }));
    }

    // Add noduon to currently targeted layer
    pub fn add_noduon(&mut self, noduon: Noduon) {
        self.layers[self.target].add_noduon(noduon)
    }

    // Add dense noduon to currently targeted layer
    pub fn add_dense_noduon(&mut self, function: AF) {
        if self.target != 0 {
            let previous_size = self.layers[self.target - 1].get_size();
            self.layers[self.target].add_dense_noduon(previous_size, function)
        }
    }

    // Add dense output noduon to currently targeted layer
    pub fn add_dense_output_noduon(&mut self, function: AF) {
        if self.target != 0 {
            let previous_size = self.layers[self.target - 1].get_size();
            self.layers[self.target].add_dense_output_noduon(previous_size, function);
        }
    }

    // Add built layer to network
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer)
    }

    // Add an input layer to network
    pub fn add_input_layer(&mut self, num_inputs: usize) {
        let mut noduons: Vec<Noduon> = vec![];
        for _i in 0..num_inputs {
                noduons.push(Noduon::Input(InputNoduon {value: 0.0 }))
        }
        noduons.push(Noduon::Bias);

        self.layers.push(Layer::Standard(Layer1D { noduons: noduons }))
    }

    // Add a dense layer to network
    pub fn add_dense_layer(&mut self, num_noduons: usize, function: AF) {
        let previous_size: usize = self.layers[self.layers.len() - 1].get_size();

        let mut new_layer: Layer = Layer::Standard(Layer1D { noduons: vec![] });

        for _i in 0..num_noduons {
            new_layer.add_dense_noduon(previous_size, function.clone());
        }
        new_layer.add_noduon(Noduon::Bias);
        self.add_layer(new_layer);
        self.target = self.layers.len()
    }

    // Add a fully connected layer of outputs
    pub fn add_dense_output_layer(&mut self, num_outputs: usize, function: AF) {
        let previous_size: usize = self.layers[self.layers.len() - 1].get_size();

        let mut new_layer: Layer = Layer::Standard(Layer1D { noduons: vec![] });

        for _i in 0..num_outputs {
            new_layer.add_dense_output_noduon(previous_size, function.clone());
        }

        self.add_layer(new_layer);
        self.target = self.layers.len() - 1;
    }

    // Moves target forward 1, allowing for editing of the next layer
    pub fn lock_layer(&mut self) {
        self.target += 1
    }

    // Updates input and output num values, assuming the last layer is an output layer and the first layer has a bias node
    pub fn update_network(&mut self) {
        self.num_inputs = self.layers[0].get_size() - 1;
        self.num_outputs = self.layers[self.layers.len() - 1].get_size();
    }

    // Returns weights as a 3d matrix. Rows are layers, columns are noduons, elements are weights. Empty columns are either input or bias
    pub fn get_weights(&self) -> Vec<Vec<Vec<f32>>> {
        self.layers.iter().map(|x| x.get_weights()).collect()
    }

    // Returns Noduon types as a matrix. Rows are Layers, Columns are types
    pub fn get_types(&self) -> Vec<Vec<String>> {
        self.layers.iter().map(|x| x.get_types()).collect()
    }

    // Returns Noduon connections as a 3d matrix
    pub fn get_connections(&self) -> Vec<Vec<Vec<bool>>> {
        self.layers.iter().map(|x| x.get_connections()).collect()
    }

    pub fn get_functions(&self) -> Vec<Vec<AF>> {
        self.layers.iter().map(|x| x.get_functions()).collect()
    }

    // Sets weights of each noduon in the network. You should include empty values for bias or input values.
    pub fn set_weights(&mut self, new_weights: Vec<Vec<Vec<f32>>>, mutate: bool, mut_rate: f32, mut_degree: f32) {
        for i in 0..self.layers.len() {
            self.layers[i].set_weights(&new_weights[i], mutate, mut_rate, mut_degree);
        }
    }

    // Changes weights of each noduon in the network. Include empty values for bias or input noduons.
    pub fn change_weights(&mut self, weight_changes: Vec<Vec<Vec<f32>>>) {
        for i in 0..self.layers.len() {
            self.layers[i].change_weights(&weight_changes[i])
        }
    }

    // Sets connections of each noduon in the network. You should invlude empty values for bias or input values
    pub fn set_connections(&mut self, new_connections: Vec<Vec<Vec<bool>>>, mutate: bool, mut_rate: f32) {
        for i in 0..self.layers.len() {
            self.layers[i].set_connections(&new_connections[i], mutate, mut_rate)
        }
    }

    // Takes inputs, passing each layer's result to each other. 
    pub fn process(&mut self, inputs: &Vec<f32>) -> Vec<f32> {

        if inputs.len() == self.num_inputs {
            self.layers[0].set_values(inputs);
        }

        let mut results: Vec<Vec<f32>> = vec![vec![]];
        for i in 0..self.layers.len() {
            results.push(self.layers[i].process(&results[i]));
        }

        return results[self.layers.len()].clone();
    }

    // Alternate process that takes use of SIMD operations
    pub fn simd_process(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        // Check if inputs are proper
        if inputs.len() == self.num_inputs {
            self.layers[0].set_values(inputs);
        }

        let mut results: Vec<Vec<f32>> = vec![self.layers[0].process(&vec![])];

        for i in 1..self.layers.len() {
            results.push(self.layers[i].simd_process(&results[i - 1]))
        }
        results[self.layers.len() - 1].clone()
    }

    // Does a full process but also returns all intermediate layer results
    pub fn forward_pass(&mut self, inputs: &Vec<f32>) -> Vec<Vec<f32>> {

        let mut results: Vec<Vec<f32>> = vec![];

        if inputs.len() == self.num_inputs {
            self.layers[0].set_values(inputs);
        } else {
            return vec![];
        }

        results.push(self.layers[0].process(&vec![]));

        for i in 1..self.layers.len() {
            results.push(self.layers[i].process(&results[i-1]));
        }

        return results;

    }

    // Calculates the loss function
    pub fn loss(&mut self, predicted: Vec<f32>, actual: Vec<f32>, function: String) -> f32 {

        match function.as_str() {
            "mse" => {
                let mut total = 0.0;
                for i in 0..predicted.len() {
                    total += (predicted[i] - actual[i]).powi(2);
                }
                total / predicted.len() as f32
            },
            "mae" => {
                let mut total: f32 = 0.0;
                for i in 0..predicted.len() {
                    total += (predicted[i] - actual[i]).abs();
                }
                total / predicted.len() as f32
            },
            _ => 1.0

        }

    }

    // Computes the gradients of all weights based on the output and loss function
    pub fn backwards_pass(&mut self, forward_pass: Vec<Vec<f32>>, actual: Vec<f32>, loss_function: String) -> Vec<Vec<Vec<f32>>> {

        // Declare the necessary variables
        let mut derivatives: Vec<Vec<f32>> = vec![];
        let mut gradients: Vec<Vec<Vec<f32>>> = vec![];
        let weights: Vec<Vec<Vec<f32>>> = self.get_weights();
        let connections: Vec<Vec<Vec<bool>>> = self.get_connections();
        let functions: Vec<Vec<AF>> = self.get_functions();

        // Extract outputs for simplicity
        let outputs = forward_pass.last().unwrap().clone();

        // Compute the output loss derivative
        let out_loss_derivative: Vec<f32> = match loss_function.as_str() {
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

        // Iterate over each layer
        for i in (0..self.layers.len()).rev() {
            let mut layer_gradients: Vec<Vec<f32>> = vec![];
            let mut layer_derivatives: Vec<f32> = vec![];
            // Iterate over each noduon
            for j in 0..self.layers[i].get_size() {
                let mut noduon_gradients: Vec<f32> = vec![];
                let mut rld: f32 = 0.0;

                // Calculate the rld (relative loss derivative) for the noduon based on weights of its forwards connections
                let cap: usize;
                if i != self.layers.len() - 1 {
                    cap = weights[i + 1].len();
                    for k in 0..cap {
                        if connections[i + 1][k].contains(&true) && connections[i + 1][k][j] {
                            rld += (derivatives[0][k] * weights[i + 1][k][j]) as f32
                        }
                    }
                    
                    // Compute the value derivative for this noduon

                    let my_derivative: f32 = rld * activation_derivative(forward_pass[i][j], &functions[i][j]);
                    layer_derivatives.push(my_derivative);

                    // Calculate the gradient from each backwards result and my_derivative
                    for k in 0..weights[i][j].len() {
                        noduon_gradients.push(my_derivative * forward_pass[i - 1][k]);
                    }
                } else {
                    for k in 0..weights[i][j].len() {
                        noduon_gradients.push(derivatives[0][j] * forward_pass[i - 1][k]);
                    }
                }
                layer_gradients.push(noduon_gradients);
            }
            gradients.insert(0, layer_gradients);
            if i != self.layers.len() - 1 {
                derivatives.insert(0, layer_derivatives);
            }
        }

        gradients

    }



    // Converts the weights of a network into a txt with a name specified by the file_name parameter
    fn weights_to_txt(&self, file_name: String) -> Result<(), Box<dyn Error>> {
        
        if Path::new(&(file_name.clone()+&String::from(".txt"))).exists() {
            fs::remove_file(file_name.clone()+&String::from(".txt"))?;
        }

        let mut file = File::create(file_name+&String::from(".txt"))?;
        let weights = self.get_weights();
        let shape: Vec<String> = self.layers.iter().map(|x| x.get_size().to_string()).collect();

        writeln!(file, "{}", shape.join(","))?;

        for layer in weights {
            for noduon in layer {
                writeln!(file, "{}", noduon.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","))?;
            }
        }
        
        Ok(())
    }

    // Converts the connections of a network into a txt with a name
    fn connections_to_txt(&self, file_name: String) -> Result<(), Box<dyn Error>> {

        if Path::new(&(file_name.clone()+&String::from(".txt"))).exists() {
            fs::remove_file(file_name.clone()+&String::from(".txt"))?;
        }

        let mut file = File::create(file_name+&String::from(".txt"))?;
        let connections = self.get_connections();
        let shape: Vec<String> = self.layers.iter().map(|x| x.get_size().to_string()).collect();

        writeln!(file, "{}", shape.join(","))?;

        for layer in connections {
            for noduon in layer {
                writeln!(file, "{}", noduon.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","))?;
            }
        }
        
        Ok(())
    }

    // Converts the types of noduons of a network into a txt with a name
    fn types_to_txt(&self, file_name: String) -> Result<(), Box<dyn Error>> {

        if Path::new(&(file_name.clone()+&String::from(".txt"))).exists() {
            fs::remove_file(file_name.clone()+&String::from(".txt"))?;
        }

        let mut file = File::create(file_name+&String::from(".txt"))?;
        let connections = self.get_types();

        for layer in connections {
                writeln!(file, "{}", layer.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","))?;
        }
        
        Ok(())
    }

    // Generates files for the types, connections, and weights of the model
    pub fn model_to_txt(&self, model_name: String) -> Result<(), Box<dyn Error>>{
        self.types_to_txt(model_name.clone()+&String::from("_types"))?;
        self.connections_to_txt(model_name.clone()+&String::from("_connections"))?;
        self.weights_to_txt(model_name.clone()+&String::from("_weights"))?;
        Ok(())
    }

    // Take the weights from a text file
    fn weights_from_txt(&mut self, file_name: String) -> Result<(), Box<dyn Error>> {
        let path = file_name + &String::from(".txt");
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut i: usize = 0;
        let mut j: usize = 0;
        let mut layer_num: usize = 0;
        let mut new_weights: Vec<Vec<Vec<f32>>> = vec![vec![]];
        let num_shape: Vec<usize> = self.layers.iter().map(|x| x.get_size()).collect();
        for line in reader.lines() {
            if i == 0{
                let my_shape: Vec<String> = self.layers.iter().map(|x| x.get_size().to_string()).collect();
                let shape: Vec<String> = line?.split(',').map(|x| x.to_string()).collect();
                if my_shape != shape {
                    println!("Incorrect Shape");
                    return Ok(());
                }
            } else {
                let set = line?;
                if j == num_shape[layer_num] {
                    new_weights.push(vec![]);
                    j = 0;
                    layer_num += 1;
                }
                if set.clone().len() != 0 {
                let nod_weights: Vec<f32> = set.split(',').map(|x| x.parse::<f32>().unwrap()).collect();
                new_weights[layer_num].push(nod_weights);
                } else {
                    new_weights[layer_num].push(vec![]);
                }
                j += 1;
            }
            i += 1;
        }
        self.set_weights(new_weights, false, 0.0, 0.0);
        Ok(())
    }

    // Take the connections from a text file
    fn connections_from_txt(&mut self, file_name: String) -> Result<(), Box<dyn Error>> {
        let path = file_name + &String::from(".txt");
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut i: usize = 0;
        let mut j: usize = 0;
        let mut layer_num: usize = 0;
        let mut new_connections: Vec<Vec<Vec<bool>>> = vec![vec![]];
        let num_shape: Vec<usize> = self.layers.iter().map(|x| x.get_size()).collect();
        for line in reader.lines() {
            if i == 0{
                let my_shape: Vec<String> = self.layers.iter().map(|x| x.get_size().to_string()).collect();
                let shape: Vec<String> = line?.split(',').map(|x| x.to_string()).collect();
                if my_shape != shape {
                    println!("Incorrect Shape");
                    return Ok(());
                }
            } else {
                let set = line?;
                if j == num_shape[layer_num] {
                    new_connections.push(vec![]);
                    j = 0;
                    layer_num += 1;
                }
                if set.clone().len() != 0 {
                    let nod_connections: Vec<bool> = set.split(',').map(|x| x.parse::<bool>().unwrap()).collect();
                    new_connections[layer_num].push(nod_connections);
                } else {
                    new_connections[layer_num].push(vec![]);
                }
                j += 1;
            }
            i += 1;
        }
        self.set_connections(new_connections, false, 0.0);
        Ok(())
    }


    // randomize the weights of the network
    pub fn randomize(&mut self) {
        for i in 0..self.layers.len() {
            self.layers[i].randomize();
        }
    }
}

// Take the connections from a text file
fn model_types_from_txt(file_name: String, inner_function: AF, output_function: AF) -> Result<Network, Box<dyn Error>> {
    let path = file_name + &String::from(".txt");
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut new_network: Network = Network {layers: vec![], num_inputs: 0, num_outputs: 0, target: 0};
    
    let mut num_outputs = 0;
    
    for line in reader.lines() {
        new_network.add_empty_layer();
        let type_strings: Vec<String> = line?.split(',').map(|x| x.to_string()).collect();
        for string in type_strings {
            match string.as_str() {
                "input" => &new_network.add_noduon(Noduon::Input(InputNoduon { value: 0.0 })),
                "inner1d" => &new_network.add_dense_noduon(inner_function.clone()),
                "bias" => &new_network.add_noduon(Noduon::Bias),
                "output1d" => {
                    new_network.add_dense_output_noduon(output_function.clone());
                    num_outputs = num_outputs + 1;
                    &()
                },
                _default => &()
            };
        }
        new_network.lock_layer();
    }
    new_network.update_network();
    Ok(new_network)
}

// Takes 3 input files and builds a model from them
pub fn model_from_txt(type_file: String, weight_file: String, connection_file: String, inner_function: AF, output_function: AF) -> Network {
    let mut model: Network = model_types_from_txt(type_file, inner_function, output_function).unwrap();
    let _ = model.weights_from_txt(weight_file);
    let _ = model.connections_from_txt(connection_file);
    return model;
}

// Builds a sequential dense model from the given vector. The 0th element is the num inputs the last element is the num outputs
pub fn build_typical_model(shape: Vec<usize>, inner_function: AF, output_function: AF) -> Network {
    let mut model: Network = Network { layers: vec![], num_inputs: 0, num_outputs: 0, target: 0 };
    model.add_input_layer(shape[0]);
    for num_noduons in &shape[1..shape.len()-1] {
        model.add_dense_layer(*num_noduons, inner_function.clone())
    }
    model.add_dense_output_layer(*shape.last().unwrap(), output_function);
    model.update_network();
    model
}

//// End Network
/////////////////
//// Start Agencies

// An agent is a Network with a fitness score
#[derive(Clone)]
pub struct Agent {
    pub network: Network,
    pub score: f32
}


// An agency is a collection of agents (Networks) designed to help with reinforcement learning tasks, also referred to as generation
#[derive(Clone)]
pub struct Agency {
    pub agents: Vec<Agent>,
    pub agency_max_size: usize,
    pub root: Network
}

impl Agency {
    // Reorders the agents based on their performance, highest scoring being put closer to the beginning
    pub fn reorder(&mut self, scores: Vec<f32>) -> f32 {
        for agent in 0..self.agents.len() {
            self.agents[agent].score = scores[agent];
        }
        self.agents.sort_by(|x,y| y.score.partial_cmp(&x.score).unwrap());
        self.agents[0].score
    }

    // Removes N number of Agents from the bottom of the ranking
    fn cut_bottom_agents(&mut self, num_cut: usize) {
        self.agents = self.agents[0..(self.agency_max_size - num_cut)].to_vec();
    }

    fn cut_top_agents(&mut self, num_cut: usize, exclude_top: usize) {
        self.agents.splice(exclude_top..num_cut, std::iter::empty());
    }

    // Adds copies from root until full
    pub fn fill_from_root(&mut self, randomize_weights: bool) {
        for _ in 0..(self.agency_max_size - self.agents.len()) {
            let mut agent = Agent {network: self.root.clone(), score: 0.0};
            if randomize_weights {
                agent.network.randomize();
            }
            self.agents.push(agent);
        }
    }

    // Clones with a chance to mutate
    fn direct_descendant(&mut self, parent: usize, mut_rate: f32, mut_degree: f32) -> Agent {
        let mut model = self.root.clone();
        model.set_weights(self.agents[parent].network.get_weights(), true, mut_rate, mut_degree);
        model.set_connections(self.agents[parent].network.get_connections(), true, mut_rate);
        Agent { network: model, score: 0.0 }
    }

    // Takes two agents and crosses their weights, randomly swapping
    fn crossover(&mut self, parent1: usize, parent2: usize) -> Agent {
        // Save each weight/connection set for future use
        let mut p1_weights = self.agents[parent1].network.get_weights();
        let p2_weights = self.agents[parent2].network.get_weights();
        let mut p1_connections: Vec<Vec<Vec<bool>>> = self.agents[parent1].network.get_connections();
        let p2_connections: Vec<Vec<Vec<bool>>> = self.agents[parent2].network.get_connections();

        let mut rng = rand::thread_rng();

        // For each weight and connection either take it from parent 1 or parent 2
        for layer in 0..p1_weights.len() {
            for noduon in 0..p1_weights[layer].len() {
                for weightcon in 0..p1_weights[layer][noduon].len() {
                    if rng.gen_bool(0.5) {
                        p1_weights[layer][noduon][weightcon] = p2_weights[layer][noduon][weightcon];
                    }
                    
                    if rng.gen_bool(0.5) {
                        p1_connections[layer][noduon][weightcon] = p2_connections[layer][noduon][weightcon];
                    }
                }
            }
        }
        // Take a model from root and replace with the crossed weights and connections
        let mut model = self.root.clone();
        model.set_connections(p1_connections, false, 0.0);
        model.set_weights(p1_weights, false, 0.0, 0.0);
        Agent { network: model, score: 0.0 }
    }

    // A full generation change. There are parameters for different elements of the transition
    pub fn genetic_generation(&mut self, cut_bottom: usize, top_crosses: usize, top_rand_crosses: usize, direct_descendants: usize, elites: usize, mut_rate: f32, mut_degree: f32) {
        self.cut_bottom_agents(cut_bottom);
        let num_parents = self.agents.len();
        let mut cross: usize = 0;
        // Generate new children as direct descendants from parents with mutation
        while cross < direct_descendants {
            for i in 0..num_parents {
                let child = self.direct_descendant(i, mut_rate, mut_degree);
                self.agents.push(child);
                cross += 1;
                if cross >= direct_descendants {
                    break;
                }
            }
        }
        cross = 0;
        // Generate new children from crossovers with the top performing parents
        while cross < top_crosses {
            for i in 0..num_parents {
                for j in 1..num_parents {
                    if i != j {
                        let child = self.crossover(i, j);
                        self.agents.push(child);
                        cross += 1;
                        if cross >= top_crosses {
                            break;
                        }
                    }
                }
                if cross >= top_crosses {
                    break;
                }
            }
        }
        cross = 0;
        // Generate new children from crossovers with random parents
        while cross < top_rand_crosses {
            let mut rng = rand::thread_rng();
            for i in 0..num_parents {
                let j = rng.gen_range(0..num_parents);
                if i != j {
                    let child = self.crossover(i, j);
                    self.agents.push(child);
                    cross += 1;
                    if cross >= top_rand_crosses {
                        break;
                    }
                }
                if cross >= top_rand_crosses {
                    break;
                }
            }
        }
        // Cut non-elite parents
        self.cut_top_agents(num_parents, elites);
        // Fill extra slots with random children from root
        self.fill_from_root(true);
        // Cut overflow children
        let extras: i32 = self.agents.len() as i32 - self.agency_max_size as i32;
        if extras > 0 {
            self.cut_bottom_agents(extras as usize);
        }
    }

    pub fn networks_iter(&self) -> impl Iterator<Item = Network> {
        let nets: Vec<Network> = self.agents.iter().map(|x| x.network.clone()).collect();
        nets.into_iter()
    }


}

// Build an Agency of size N by generating random weights from the root Network
pub fn build_agency_from_root(root: Network, max_size: usize, randomize_weights: bool) -> Agency {
    let mut seeds: Vec<Agent> = (0..max_size).map(|_| Agent {network: root.clone(), score: 0.0}).collect();
    if randomize_weights {
        for i in 0..seeds.len() {
            seeds[i].network.randomize();
        }
    }
    Agency { agents: seeds, agency_max_size: max_size, root }    
}

/// END AGENCIES ///////
///////////////////////
/// START TUTORS /////

pub struct Tutor {
    pub gradients: Vec<Vec<Vec<f32>>>,
    pub learning_coefficient: f32,
    pub other_parameters: [f32; 4],
    pub pupil: Network
}

pub fn build_default_tutor(network: Network) -> Tutor {
    Tutor { gradients: vec![], learning_coefficient: 0.001, other_parameters: [0.0, 0.0, 0.0, 0.0], pupil: network }
}

impl Tutor {

    // Apply basic gradient descent to the tutor's pupil
    pub fn gradient_descent(&mut self, mut gradients: Vec<Vec<Vec<f32>>>) {

        // Update gradients to hold the new gradients
        self.gradients = gradients.clone();

        // Iterate through the gradients, mutliplying by the negative learning coefficient
        for i in 0..gradients.len() {
            for j in 0..gradients[i].len() {
                for k in 0..gradients[i][j].len() {
                    gradients[i][j][k] *= self.learning_coefficient * -1.0
                }
            }
        }

        // Change each weight by the determined amount
        self.pupil.change_weights(gradients)

    }

    // Preform a complete training step on a batch of data
    pub fn batch_train_gd(&mut self, train_features: Vec<Vec<f32>>, train_labels: Vec<Vec<f32>>, loss_function:String) {

        let mut gradients:Vec<Vec<Vec<f32>>> = self.pupil.get_weights().iter().map(|x| x.iter().map(|x| x.iter().map(|_| 0.0).collect()).collect()).collect();
        let mut gradient_sets: Vec<Vec<Vec<Vec<f32>>>> = vec![];

        for i in 0..train_features.len() {
            let forward = self.pupil.forward_pass(&train_features[i]);
            let backward = self.pupil.backwards_pass(forward, train_labels[i].clone(), loss_function.clone());
            gradient_sets.push(backward)
        }

        for i in 0..gradients.len() {
            for j in 0..gradients[i].len() {
                for k in 0..gradients[i][j].len() {
                    for l in 0..gradient_sets.len() {
                        gradients[i][j][k] += gradient_sets[l][i][j][k]
                    }
                    gradients[i][j][k] = gradients[i][j][k] / gradient_sets.len() as f32;
                }
            }
         }

        self.gradient_descent(gradients);

    }

}

/// END TUTORS /////////
///////////////////////
/// 

pub fn main() {
    println!("Wheee");
}