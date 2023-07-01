
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

#[derive(Default)]
struct InputNoduon {
    name: String,
    value: f64
}

impl InputNoduon {
    fn get_value(&self) -> f64{
        return self.value;
    }

    fn set_value(&mut self, value: f64) {
        self.value = value;
    }
}

struct InnerNoduon {
    connections: Vec<Noduon>,
    weights: Vec<f64>
}

impl InnerNoduon {
    fn get_sum(&self) -> f64 {
        let mut sum: f64 = 0.0;
        for connection in 0..self.connections.len() {
            sum += self.connections[connection].get_val() * self.weights[connection]
        }
        return sum;
    }
}

struct OutputNoduon {
    name: String,
    connections: Vec<Noduon>,
    weights: Vec<f64>
}

enum Noduon {
    
    Input(InputNoduon),
    Inner(InnerNoduon),
    Output(OutputNoduon)
     
}

impl Noduon {
    fn get_val(&self) -> f64 {
        match self {
            Noduon::Input(f) => f.get_value(),
            Noduon::Inner(f) => f.get_sum(),
            Noduon::Output(f) => f
        }
    }
}

fn main() {
    println!("Hi")
}