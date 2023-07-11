
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

struct InputNoduon {
    name: String,
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
                    "sigmod" => 1.0 / (1.0 + (-total).exp()),
                    _default => 0.0
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
}

fn main() {
    let a: Noduon = Noduon::Input(InputNoduon { name: String::from("Barry"), value: 5.0 });
    let b: Noduon = Noduon::Inner1D(InnerNoduon1D { connections: vec![true], weights: vec![0.5], function: String::from("relu") });
    let c: Noduon = Noduon::Output1D(OutputNoduon1D { name: String::from("Allen"), connections: vec![false,true], weights: vec![0.0, 0.5], function: String::from("relu") });

    let layers: Vec<Noduon> = vec![a,b,c];
    let mut previous: Vec<f64> = vec![];
    for layer in 0..layers.len() {
        previous.push(layers[layer].result(&previous))
    }
    print!("{:?}",previous)
}