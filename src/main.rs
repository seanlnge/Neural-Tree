enum Value {
    Constant(f64),
    Mutable(f64)
}

impl Value {
    fn value(&self) -> f64 {
        match self {
            Self::Constant(x) => *x,
            Self::Mutable(x) => *x
        }
    }
}

enum Transform {
    Identity,

    Bias(Value),
    Negate,
    Scalar(Value),
    Invert,

    Sigmoid,
    Tanh,
    ReLU,

    Composition(Box<Transform>, Box<Transform>)
}
impl Transform {
    fn new(mut transforms: Vec<Transform>) -> Self {
        if transforms.len() == 1 { return transforms.pop().unwrap() }

        transforms.reverse();
        let mut previous_composition = Self::Composition(
            Box::new(transforms.pop().unwrap()),
            Box::new(transforms.pop().unwrap())
        );
        while transforms.len() > 0 {
            previous_composition = Self::Composition(
                Box::new(previous_composition),
                Box::new(transforms.pop().unwrap())
            );
        }

        previous_composition
    }

    fn apply(&self, mut value: f64) -> f64 {
        match self {
            Self::Identity => value,

            Self::Bias(x) => x.value() + value,
            Self::Negate => -value,
            Self::Scalar(x) => x.value() * value,
            Self::Invert => 1.0 / value,
            
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Tanh => (value.exp() + (-value).exp()) / (value.exp() - (-value).exp()),
            Self::ReLU => value.max(0.0),
            
            Self::Composition(bt1, bt2) => {
                let t1 = bt1.as_ref();
                let t2 = bt2.as_ref();
                t2.apply(t1.apply(value))
            }
        }
    }

    fn backpropagate(&mut self, expected: f64) -> f64 {
        // dT/dz * expected
        match self {
            Self::Identity => expected,              // dT/dz = 1

            Self::Bias(x) => {                       // dT/dz = 1
                x.backpropagate(expected);
                expected
            },
            Self::Negate => -expected,               // dT/dz = -1
            Self::Scalar(x) => {                     // dT/dz = x
                x.value() * expected
                , // dT/dz = x
            Self::Invert => -1.0 / expected.powi(1), // dT/dz = -1/z^2

        }
    }
}

enum NodeType {
    Constant(f64),
    Sum,
    Product
}

struct Node {
    inputs: Vec<Node>,
    node_type: NodeType,
    transform: Transform,
    activation: Option<f64>
}
impl Node {
    fn new(inputs: Vec<Node>, node_type: NodeType, transform: Transform) -> Self {
        Self { inputs, node_type, transform, activation: None }
    }

    fn evaluate(&mut self) -> f64 {
        let mut inputs = vec![];
        for input in self.inputs.iter_mut() {
            inputs.push(input.evaluate());
        }
        
        let untransformed = match self.node_type {
            NodeType::Constant(x) => x,
            NodeType::Sum => {
                let mut total = 0.0_f64;
                for input in inputs.iter() { total += input }
                total
            },
            NodeType::Product => {
                let mut total = 0.0_f64;
                for input in inputs.iter() { total *= input }
                total
            }
        };

        self.activation = Some(self.transform.apply(untransformed));
        self.activation.unwrap()
    }
}

fn main() {
    let m = Node::new(vec![], NodeType::Constant(1.0), Transform::new(vec![
        Transform::Negate,
        Transform::Bias(Value::Constant(0.5))
    ]));
    let n = Node::new(vec![], NodeType::Constant(-0.3), Transform::new(vec![
        Transform::Bias(Value::Mutable(-0.2)),
        Transform::Invert,
        Transform::Scalar(Value::Mutable(1.6))
    ]));
    
    let mut o = Node::new(vec![m, n], NodeType::Sum, Transform::Identity);
    println!("{}", o.evaluate())
}