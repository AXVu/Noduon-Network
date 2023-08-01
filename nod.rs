
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
set_weights(&mut self, new_weights) -> Checks if new_weights is the same size as the old weights and replaces them. Input/Bias->N/a
set_connections(&mut self, new_connections) -> checks if new_connections is the same size as old connections and replaces them. Input/Bias->N/A
set_value(&mut self, new_value) -> Sets a new value for the input node. Does nothing for other types.
get_weights(&self) -> Returns the current weights of the given Noduon. Input/Bias -> empty vector
get_connections(&self) -> Returns the current connections of the given Noduon. Input/Bias -> empty vector
get_type(&self) -> Returns the noduon type as a string ("input", "inner1d",etc)
result(&self, past_layer) -> Returns the sum of each value of the input vector times the connected weights, passed through an activation
function. Input -> returns value. Bias -> returns 1

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
set_values(&mut self) -> Sets the values of inputs, starting at the top
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
process(&mut self, inputs) -> With the given inputs, do a full process through each layer and return the output of the network.
 */

use std::{vec, fs, io};
use std::error::Error;
use rand::Rng;
use std::time;
use std::io::{Read, BufRead};
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

#[derive(Clone)]
struct InputNoduon {
    value: f64
}

#[derive(Clone)]
// A 1D Inner Noduon only connects to noduons on one dimension
struct InnerNoduon1D {
    connections: Vec<bool>,
    weights: Vec<f64>,
    function: String
}

#[derive(Clone)]
struct OutputNoduon1D {
    name: String,
    connections: Vec<bool>,
    weights: Vec<f64>,
    function: String
}


#[derive(Clone)]
enum Noduon {

    Input(InputNoduon),
    Inner1D(InnerNoduon1D),
    Output1D(OutputNoduon1D),
    Bias

}

impl Noduon {
    // Set weights for non-inputs, will do nothing if applied to input
    fn set_weights(&mut self, new_weights: &Vec<f64>) {
        match self {
            Noduon::Input(_) | Noduon::Bias => {},
            Noduon::Inner1D(f) => {
                if new_weights.len() == f.weights.len() {
                    f.weights = new_weights.to_vec();
                }
            },
            Noduon::Output1D(f) => {
                if new_weights.len() == f.weights.len() {
                    f.weights = new_weights.to_vec();
                }
            }
        }
    }

    // Sets connections for non-inputs, will do nothing if applied to input
    fn set_connections(&mut self, new_connections: &Vec<bool>) {
        match self {
            Noduon::Input(_) | Noduon::Bias => {},
            Noduon::Inner1D(f) => {
                if new_connections.len() == f.connections.len() {
                    f.connections = new_connections.to_vec();
                }
            },
            Noduon::Output1D(f) => {
                if new_connections.len() == f.connections.len() {
                    f.connections = new_connections.to_vec();
                }
            }
        }
    }

    // Set the input value. Doesn't do anything for other types
    fn set_value(&mut self, new_value: f64) {
        match self {
            Noduon::Input(f) => f.value = new_value,
            default => ()
        }
    }

    // Return the noduon type as a string
    fn get_type(&self) -> String{
        match self {
            Noduon::Input(_) => String::from("input"),
            Noduon::Inner1D(_) => String::from("inner1d"),
            Noduon::Output1D(_) => String::from("output1d"),
            Noduon::Bias => String::from("bias")
        }
    }

    // Gets weights for non-inputs, will return an empty vector if applied to input
    fn get_weights(&self) -> Vec<f64> {
        match self {
            Noduon::Input(_) | Noduon::Bias => vec![],
            Noduon::Inner1D(f) => f.weights.clone(),
            Noduon::Output1D(f) => f.weights.clone(),
        }
    }

    // Gets connections for non-inputs, will return an empty vector if allplied to input
    fn get_connections(&self) -> Vec<bool> {
        match self {
            Noduon::Input(_) | Noduon::Bias => vec![],
            Noduon::Inner1D(f) => f.connections.clone(),
            Noduon::Output1D(f) => f.connections.clone()
        }
    }

    fn randomize(&mut self) {
        match self {
            Noduon::Bias | Noduon::Input(_) => (),
            Noduon::Inner1D(f) => {
                let mut rng = rand::thread_rng();
                for i in 0..f.weights.len() {
                    f.weights[i] = rng.gen_range(-1.0..=1.0);
                }
            },
            Noduon::Output1D(f) => {
                let mut rng = rand::thread_rng();
                for i in 0..f.weights.len() {
                    f.weights[i] = rng.gen_range(-1.0..=1.0)
                }
            }
        }
    }

    // Passing in a previous layer's output vector, returns the output of a noduon. Returns the value of an input noduon
    fn result(&self, past_layer: &Vec<f64>) -> f64 {
        match self {
            Noduon::Input(f) => f.value,
            Noduon::Bias => 1.0,
            Noduon::Inner1D(f) => {
                let mut total: f64 = 0.0;
                for connect in 0..f.connections.len() {
                    if f.connections[connect] {
                        total += past_layer[connect] * f.weights[connect];
                    }
                }

                total = match f.function.as_str() {
                    "tanh" => total.tanh(),
                    "relu" => {
                        if total > 0.0 {
                            total
                        } else {
                            0.0
                        }},
                    "sigmod" => 1.0 / (1.0 + (-total).exp()),
                    _default => 0.0
                 };
                 total
            },
            Noduon::Output1D(f) => {
                let mut total: f64 = 0.0;
                for connect in 0..f.connections.len() {
                    if f.connections[connect] {
                        total += past_layer[connect] * f.weights[connect];
                    }
                }

                total = match f.function.as_str() {
                    "tanh" => total.tanh(),
                    "relu" => {
                        if total > 0.0 {
                            total
                        } else {
                            0.0
                        }},
                    "sigmoid" => 1.0 / (1.0 + (-total).exp()),
                    _default => total
                 };

                total
            }
        }
    }

}

// End of Noduon section
////////////////////////////////////
// Start of Layer section

#[derive(Clone)]
struct Layer1D {
    noduons: Vec<Noduon>
}

#[derive(Clone)]
enum Layer {
    Standard(Layer1D)
}

impl Layer {
    // The standard process of a layer, taking the previous layer's output and outputting the new layer
    fn process(&self, past_layer: Vec<f64>) -> Vec<f64> {
        match self {
            Layer::Standard(f) => {

                let mut result: Vec<f64> = vec![];

                for noduon in &f.noduons {
                    result.push(noduon.result(&past_layer))
                }

                result

            }
        }
    }

    // Returns Noduon types as a vector
    fn get_types(&self) -> Vec<String> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_type()).collect()
        }
    }

    // Returns the weights as a matrix. Rows are noduons, columns are the previous layer's noduons
    fn get_weights(&self) -> Vec<Vec<f64>> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_weights()).collect()
        }
    }

    // Returns the connections of each Noduon as a matrix. Empty lists are either Input or Bias noduons.
    fn get_connections(&self) -> Vec<Vec<bool>> {
        match self {
            Layer::Standard(f) => f.noduons.iter().map(|x| x.get_connections()).collect()
        }
    }

    // Sets the value of each input noduon
    fn set_values(&mut self, values: Vec<f64>) {
        match self {
            Layer::Standard(f) => {
                for i in 0..values.len() {
                    f.noduons[i].set_value(values[i]);
                }
            }
        }
    }

    // Sets the weights of each noduon in a given layer
    fn set_weights(&mut self, new_weights: &Vec<Vec<f64>>) {
        match self {
            Layer::Standard(f) => {
                for i in 0..f.noduons.len() {
                    f.noduons[i].set_weights(&new_weights[i]);
                }
            }
        }
    }

    // Sets the connections of each noduon in a given layer
    fn set_connections(&mut self, new_connections: &Vec<Vec<bool>>) {
        match self {
            Layer::Standard(f) => {
                for i in 0..f.noduons.len() {
                    f.noduons[i].set_connections(&new_connections[i]);
                }
            }
        }
    }

    // Adds the given noduon to the layer
    fn add_noduon(&mut self, noduon: Noduon) {
        match self {
            Layer::Standard(f) => {
                f.noduons.push(noduon)
            }
        }
    }

    // Adds a fully connected noduon to the layer
    fn add_dense_noduon(&mut self, previous_layer_noduons: usize, function: String) {
        match self {
            Layer::Standard(f) => {

                let mut rng = rand::thread_rng();
                let weight: Vec<f64> = (0..previous_layer_noduons).map(|_| rng.gen_range(-1.0..=1.0)).collect();
                let connects: Vec<bool> = (0..previous_layer_noduons).map(|_| true).collect();
                let new_noduon: Noduon = Noduon::Inner1D(InnerNoduon1D { connections: connects, weights: weight, function: function });
                f.noduons.push(new_noduon);
            }
        }
    }

    // Adds a fully connected output noduon to the layer
    fn add_dense_output_noduon(&mut self, previous_layer_noduons: usize, function: String, name: String) {
        match self {
            Layer::Standard(f) => {

                let mut rng = rand::thread_rng();
                let weight: Vec<f64> = (0..previous_layer_noduons).map(|_| rng.gen_range(-1.0..=1.0)).collect();
                let connects: Vec<bool> = (0..previous_layer_noduons).map(|_| true).collect();
                let new_noduon: Noduon = Noduon::Output1D(OutputNoduon1D{ connections: connects, weights: weight, name: name, function: function });
                f.noduons.push(new_noduon);
            }
        }
    }

    // Randomize the weights of each noduon in the layer
    fn randomize(&mut self) {
        match self {
            Layer::Standard(f) => {
                for noduon in 0..f.noduons.len() {
                    f.noduons[noduon].randomize();
                }
            }
        }
    }


    // Return the number of noduons in the layer
    fn get_size(&self) -> usize{
        match self {
            Layer::Standard(f) => f.noduons.len()
        }
    }

}

// End Layer
/////////////////////////////////////////////////
// Start Network

#[derive(Clone)]
struct Network {
    layers: Vec<Layer>,
    num_inputs: usize,
    num_outputs: usize,
    target: usize
}

impl Network {
    
    // Add layer to network
    fn add_empty_layer(&mut self) {
        self.layers.push(Layer::Standard(Layer1D { noduons: vec![] }));
    }

    // Add noduon to currently targeted layer
    fn add_noduon(&mut self, noduon: Noduon) {
        self.layers[self.target].add_noduon(noduon)
    }

    // Add dense noduon to currently targeted layer
    fn add_dense_noduon(&mut self, function: String) {
        if self.target != 0 {
            let previous_size = self.layers[self.target - 1].get_size();
            self.layers[self.target].add_dense_noduon(previous_size, function)
        }
    }

    // Add dense output noduon to currently targeted layer
    fn add_dense_output_noduon(&mut self, function: String, name: String) {
        if self.target != 0 {
            let previous_size = self.layers[self.target - 1].get_size();
            self.layers[self.target].add_dense_output_noduon(previous_size, function, name);
        }
    }

    // Add built layer to network
    fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer)
    }

    // Add an input layer to network
    fn add_input_layer(&mut self, num_inputs: usize) {
        let mut noduons: Vec<Noduon> = vec![];
        for i in 0..num_inputs {
                noduons.push(Noduon::Input(InputNoduon {value: 0.0 }))
        }
        noduons.push(Noduon::Bias);

        self.layers.push(Layer::Standard(Layer1D { noduons: noduons }))
    }

    // Add a dense layer to network
    fn add_dense_layer(&mut self, num_noduons: usize, function: String) {
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
    fn add_dense_output_layer(&mut self, num_outputs: usize, output_names: Vec<String>, function: String) {
        let previous_size: usize = self.layers[self.layers.len() - 1].get_size();

        let mut new_layer: Layer = Layer::Standard(Layer1D { noduons: vec![] });

        for i in 0..num_outputs {
            new_layer.add_dense_output_noduon(previous_size, function.clone(), output_names[i].clone());
        }

        self.add_layer(new_layer);
        self.target = self.layers.len() - 1;
    }

    // Moves target forward 1, allowing for editing of the next layer
    fn lock_layer(&mut self) {
        self.target += 1
    }

    // Updates input and output num values, assuming the last layer is an output layer
    fn update_network(&mut self) {
        self.num_inputs = self.layers[0].get_size();
        self.num_outputs = self.layers[self.layers.len() - 1].get_size();
    }

    // Returns weights as a 3d matrix. Rows are layers, columns are noduons, elements are weights. Empty columns are either input or bias
    fn get_weights(&self) -> Vec<Vec<Vec<f64>>> {
        self.layers.iter().map(|x| x.get_weights()).collect()
    }

    // Returns Noduon types as a matrix. Rows are Layers, Columns are types
    fn get_types(&self) -> Vec<Vec<String>> {
        self.layers.iter().map(|x| x.get_types()).collect()
    }

    // Returns Noduon connections as a 3d matrix
    fn get_connections(&self) -> Vec<Vec<Vec<bool>>> {
        self.layers.iter().map(|x| x.get_connections()).collect()
    }

    // Sets weights of each noduon in the network. You should include empty values for bias or input values.
    fn set_weights(&mut self, new_weights: Vec<Vec<Vec<f64>>>) {
        for i in 0..self.layers.len() {
            self.layers[i].set_weights(&new_weights[i]);
        }
    }

    // Sets connections of each noduon in the network. You should invlude empty values for bias or input values
    fn set_connections(&mut self, new_connections: Vec<Vec<Vec<bool>>>) {
        for i in 0..self.layers.len() {
            self.layers[i].set_connections(&new_connections[i])
        }
    }

    // Takes inputs, passing each layer's result to each other. 
    fn process(&mut self, inputs: Vec<f64>) -> Vec<f64> {

        if inputs.len() == self.num_inputs {
            self.layers[0].set_values(inputs);
        }

        let mut results: Vec<Vec<f64>> = vec![vec![]];
        for i in 0..self.layers.len() {
            results.push(self.layers[i].process(results[i].clone()));
        }

        return results[self.layers.len()].clone();
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
    fn model_to_txt(&self, model_name: String) -> Result<(), Box<dyn Error>>{
        self.types_to_txt(model_name.clone()+&String::from("_types"))?;
        self.connections_to_txt(model_name.clone()+&String::from("_connections"))?;
        self.weights_to_txt(model_name.clone()+&String::from("_weights"))?;
        Ok(())
    }

    // Take the weights from a text file
    fn weights_from_txt(&mut self, file_name: String) -> Result<(), Box<dyn Error>> {
        let path = file_name + &String::from(".txt");
        let mut file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut i: usize = 0;
        let mut j: usize = 0;
        let mut layer_num: usize = 0;
        let mut new_weights: Vec<Vec<Vec<f64>>> = vec![vec![]];
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
                let nod_weights: Vec<f64> = set.split(',').map(|x| x.parse::<f64>().unwrap()).collect();
                new_weights[layer_num].push(nod_weights);
                } else {
                    new_weights[layer_num].push(vec![]);
                }
                j += 1;
            }
            i += 1;
        }
        self.set_weights(new_weights);
        Ok(())
    }

    // Take the connections from a text file
    fn connections_from_txt(&mut self, file_name: String) -> Result<(), Box<dyn Error>> {
        let path = file_name + &String::from(".txt");
        let mut file = File::open(path)?;
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
        self.set_connections(new_connections);
        Ok(())
    }


    // randomize the weights of the network
    fn randomize(&mut self) {
        let mut rng = rand::thread_rng();

    }
}

// Take the connections from a text file
fn model_types_from_txt(file_name: String) -> Result<Network, Box<dyn Error>> {
    let path = file_name + &String::from(".txt");
    let mut file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut new_network: Network = Network {layers: vec![], num_inputs: 0, num_outputs: 0, target: 0};
    
    let mut num_outputs = 0;
    
    for line in reader.lines() {
        new_network.add_empty_layer();
        let type_strings: Vec<String> = line?.split(',').map(|x| x.to_string()).collect();
        for string in type_strings {
            match string.as_str() {
                "input" => &new_network.add_noduon(Noduon::Input(InputNoduon { value: 0.0 })),
                "inner1d" => &new_network.add_dense_noduon(String::from("relu")),
                "bias" => &new_network.add_noduon(Noduon::Bias),
                "output1d" => {
                    new_network.add_dense_output_noduon(String::from("relu"), num_outputs.to_string());
                    num_outputs = num_outputs + 1;
                    &()
                },
                default => &()
            };
        }
        new_network.lock_layer();
    }
    new_network.update_network();
    Ok(new_network)
}

// Takes 3 input files and builds a model from them
fn model_from_txt(type_file: String, weight_file: String, connection_file: String) -> Network {
    let mut model: Network = model_types_from_txt(type_file).unwrap();
    model.weights_from_txt(weight_file);
    model.connections_from_txt(connection_file);
    return model;
}

// Builds a sequential dense model from the given vector. The 0th element is the num inputs the last element is the num outputs
fn build_typical_model(shape: Vec<usize>, inner_function: String, output_function: String) {
    let mut model: Network = Network { layers: vec![], num_inputs: 0, num_outputs: 0, target: 0 };
    model.add_input_layer(shape[0]);
    for num_noduons in &shape[1..shape.len()-1] {
        model.add_dense_layer(*num_noduons, inner_function.clone())
    }
    model.add_dense_output_layer(*shape.last().unwrap(), vec![String::from(""); *shape.last().unwrap()], output_function)
}

fn main() {
    let model: Network = model_from_txt(String::from("with_the_fur_types"), String::from("with_the_fur_weights"), String::from("with_the_fur_connections"));
    let mut m2 = model.clone();
    m2.target = m2.target - 1;
    m2.add_dense_output_noduon(String::from("relu"), String::from("apple"));
    m2.weights_to_txt(String::from("in_the_club"));
}