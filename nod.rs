
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
set_weights(&self, new_weights) -> Checks if new_weights is the same size as the old weights and replaces them. Input/Bias->N/a
set_connections(&self, new_connections) -> checks if new_connections is the same size as old connections and replaces them. Input/Bias->N/A
get_weights(&self) -> Returns the current weights of the given Noduon. Input/Bias -> empty vector
get_connections(&self) -> Returns the current connections of the given Noduon. Input/Bias -> empty vector
result(&self, past_layer) -> Returns the sum of each value of the input vector times the connected weights, passed through an activation
function. Input -> returns value. Bias -> returns 1


The next up structure in the hierarchy is the layer.
Layers contain any number of noduons of various types.
Layers have functions that will take an input and generate the output vector resulting from each of its noduon's calculations.
Layers will have "presets" that can be used to generate different layer types.
As of now, there is only 1 kind of layer, but this may increase in the future.


 */

use std::vec;
use rand::Rng;

struct InputNoduon {
    value: f64
}

// A 1D Inner Noduon only connects to noduons on one dimension
struct InnerNoduon1D {
    connections: Vec<bool>,
    weights: Vec<f64>,
    function: String
}

struct OutputNoduon1D {
    name: String,
    connections: Vec<bool>,
    weights: Vec<f64>,
    function: String
}



enum Noduon {

    Input(InputNoduon),
    Inner1D(InnerNoduon1D),
    Output1D(OutputNoduon1D),
    Bias

}

impl Noduon {
    // Set weights for non-inputs, will do nothing if applied to input
    fn set_weights(&mut self, new_weights: Vec<f64>) {
        match self {
            Noduon::Input(_) | Noduon::Bias => {},
            Noduon::Inner1D(f) => {
                if new_weights.len() == f.weights.len() {
                    f.weights = new_weights;
                }
            },
            Noduon::Output1D(f) => {
                if new_weights.len() == f.weights.len() {
                    f.weights = new_weights;
                }
            }
        }
    }

    // Sets connections for non-inputs, will do nothing if applied to input
    fn set_connections(&mut self, new_connections: Vec<bool>) {
        match self {
            Noduon::Input(_) | Noduon::Bias => {},
            Noduon::Inner1D(f) => {
                if new_connections.len() == f.connections.len() {
                    f.connections = new_connections;
                }
            },
            Noduon::Output1D(f) => {
                if new_connections.len() == f.connections.len() {
                    f.connections = new_connections;
                }
            }
        }
    }

    fn set_value(&mut self, new_value: f64) {
        match self {
            Noduon::Input(f) => f.value = new_value,
            default => ()
        }
    }

    // Gets weights for non-inputs, will return an empty vector if applied to input
    fn get_weights(&self) -> Vec<f64> {
        match self {
            Noduon::Input(_) | Noduon::Bias => vec![],
            Noduon::Inner1D(f) => f.weights.clone(),
            Noduon::Output1D(f) => f.weights.clone()
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

struct Layer1D {
    noduons: Vec<Noduon>
}

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

    // Returns the weights as a matrix. Rows are noduons, columns are the previous layer's noduons
    fn weights_as_matrix(&self) -> Vec<Vec<f64>> {
        match self {
            Layer::Standard(f) => {
                let mut layer_matrix: Vec<Vec<f64>> = vec![];
                for noduon in &f.noduons {
                    layer_matrix.push(noduon.get_weights())
                }
                layer_matrix
            }
        }
    }

    fn set_values(&mut self, values: Vec<f64>) {
        match self {
            Layer::Standard(f) => {
                for i in 0..values.len() {
                    f.noduons[i].set_value(values[i]);
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

    fn get_size(&self) -> usize{
        match self {
            Layer::Standard(f) => f.noduons.len()
        }
    }

}

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

    // Add built layer to network
    fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer)
    }

    // Add an input layer to network
    fn add_input_layer(&mut self, num_inputs: usize, input_names: Vec<String>) {
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
        self.add_layer(new_layer)
    }

    fn add_dense_output_layer(&mut self, num_outputs: usize, output_names: Vec<String>, function: String) {
        let previous_size: usize = self.layers[self.layers.len() - 1].get_size();

        let mut new_layer: Layer = Layer::Standard(Layer1D { noduons: vec![] });

        for i in 0..num_outputs {
            new_layer.add_dense_output_noduon(previous_size, function.clone(), output_names[i].clone());
        }

        self.add_layer(new_layer)
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

}

fn main() {
    let mut model: Network = Network { layers: vec![], num_inputs: 0, num_outputs: 0, target: 0 };
    model.add_input_layer(3, vec!["A", "B", "C"].iter().map(|&x| String::from(x)).collect());
    model.add_dense_layer(8, String::from("relu"));
    model.add_dense_output_layer(4, vec!["W","X","Y","Z"].iter().map(|&x| String::from(x)).collect(), String::from("sigmoid"));
    model.update_network();
    let output: Vec<f64> = model.process(vec![0.0, 0.0, 0.0]);
    println!("{:?}",output)
}