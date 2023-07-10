
/*
Documentation:
The basic structure is the Noduon, which has the weighted sum method.
There are three basic Noduon types.
Input Noduons, which have a value set externally. Its weighted sum method returns this value. These are named.
Inner Noduons, of which there are a couple types, connect to any number of other inner noduons or input noduons.
Its weighted sum method returns the values of those noduons multiplied by its respective weight.
Output Noduons, which are exactly the same as Inner noduons except they are named.

Detailed and Methods:


 */

use std::default;

#[derive(Default)]
struct InputNoduon {
    name: String,
    value: f64
}

// InputNoduons can be named, and have only one value
impl InputNoduon {
    fn get_value(&self) -> f64{
        return self.value;
    }

    fn set_value(&mut self, value: f64) {
        self.value = value;
    }
}

// A 1D Inner Noduon only connects to noduons on one dimension
struct InnerNoduon1D {
    connections: Vec<bool>,
    weights: Vec<f64>,
    function: String
}

impl InnerNoduon1D {
    // Takes a vector of booleans the same length as the old one and replaces the original
    fn set_connections(&mut self, new_connections: Vec<bool>) {
        if new_connections.len() == self.connections.len() {
            self.connections = new_connections;
        }
    }

    // Returns weights
    fn get_weights(&self) -> &Vec<f64> {
        return &self.weights;
    }

    // Returns connection truth values
    fn get_connections(&self) -> &Vec<bool> {
        return &self.connections;
    }

    // Takes a vector, multiplying it by each weight if connected then applying the flattening function
    fn result(&self, layer_results: Vec<f64>) -> f64{
        let mut total: f64 = 0.0;
        for connect in 0..self.connections.len() {
            if self.connections[connect] {
                total += layer_results[connect] * self.weights[connect];
            }
        }

        total = match self.function.as_str() {
            "tanh" => total.tanh(),
            "relu" => {
                if total > 0.0 {
                    total
                } else {
                    0.0
                }
            },
            "sigmod" => 1.0 / (1.0 + (-total).exp()),
            default => 0.0
        };

        return total;
    }
}

struct OutputNoduon1D {
    name: String,
    connections: Vec<bool>,
    weights: Vec<f64>
}



enum Noduon {

    Input(InputNoduon),
    Inner1D(InnerNoduon1D),
    Output1D(OutputNoduon1D)

}

impl Noduon {
    // Set weights for non-inputs, will do nothing if applied to input
    fn set_weights(&mut self, new_weights: Vec<f64>) {
        match self {
            Noduon::Input(f) => {},
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
            Noduon::Input(f) => {},
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
            Noduon::Input(f) => vec![],
            Noduon::Inner1D(f) => f.weights.clone(),
            Noduon::Output1D(f) => f.weights.clone()
        }
    }

    // Gets connections for non-inputs, will return an empty vector if allplied to input
    fn get_connections(&self) -> Vec<bool> {
        match self {
            Noduon::Input(f) => vec![],
            Noduon::Inner1D(f) => f.connections.clone(),
            Noduon::Output1D(f) => f.connections.clone()
        }
    }

}

fn main() {
    println!("Hi")
}