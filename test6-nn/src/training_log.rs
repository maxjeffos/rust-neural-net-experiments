use std::env;
use std::fs;
use std::path;

use crate::SimpleNeuralNetwork;
use metrics::epoch_timestamp;
use serde_derive::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct TrainingSession {
    training_session_id: u128,
    start_time_epoch: u128,
    initial_cost: f64,
    network_config: NetworkConfig,
    optimizer: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct TrainingUpdate {
    epoch: usize,
    epochs_completed: usize,
    timestamp_epoch: u128,
    training_set_cost: f64,
    test_set_cost: f64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct NetworkConfig {
    layers: Vec<LoggerLayerInfo>,
}

impl NetworkConfig {
    pub fn from_neural_network(nn: &SimpleNeuralNetwork) -> Self {
        let mut layers = Vec::new();

        for l in 0..nn.sizes.len() {
            let li = nn.layer_infos.get(&l).unwrap();
            let activation_function = &li.activation_function;
            let activation_function_str = format!("{:?}", activation_function);
            let initializer = li.initializer.clone();

            let layer = LoggerLayerInfo {
                size: nn.sizes[l],
                activation_function: activation_function_str,
                initializer,
            };

            layers.push(layer);
        }

        Self { layers }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct LoggerLayerInfo {
    size: usize,
    activation_function: String,
    initializer: Option<String>,
}

pub struct TrainingSessionLogger {
    pub training_session_id: u128,
    pub full_session_output_directory: Option<path::PathBuf>,
}

impl TrainingSessionLogger {
    pub fn new() -> Self {
        let training_session_id = epoch_timestamp();
        Self {
            training_session_id,
            full_session_output_directory: None,
        }
    }

    pub fn create_training_log_directory(&mut self) -> std::io::Result<()> {
        let maybe_training_log_home = std::env::var("TRAINING_LOG_HOME");

        let mut training_sessions_path = if let Ok(training_log_home) = maybe_training_log_home {
            let training_home = path::PathBuf::from(training_log_home);
            training_home
        } else {
            let mut training_sessions_path = env::current_dir()?;
            println!(
                "The current directory is {}",
                training_sessions_path.display()
            );
            training_sessions_path.push(path::Path::new("training-sessions"));
            println!(
                "The training sessions directory is: {}",
                training_sessions_path.display()
            );
            training_sessions_path
        };

        // I think this code is screwy
        // make sure the directory exists and fail if it does not
        let p_exists = training_sessions_path.exists();
        if p_exists {
            println!("dir exists: {:?}", training_sessions_path);
            // create sub dir for this sesh
            let this_tr_sesh_segment = format!("{}", self.training_session_id);
            training_sessions_path.push(path::Path::new(&this_tr_sesh_segment));
            fs::create_dir(&training_sessions_path)?;
            self.full_session_output_directory = Some(training_sessions_path);
        } else {
            println!("dir does not exist: {:?}", training_sessions_path);
            let error_message = format!("directory does not exist: {:?}", training_sessions_path);
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                error_message,
            ));
        }

        Ok(())
    }

    pub fn write_training_session_file(
        &self,
        initial_cost: f64,
        network_config: NetworkConfig,
        optimizer: String,
    ) -> Result<(), std::io::Error> {
        let training_session = TrainingSession {
            training_session_id: self.training_session_id,
            start_time_epoch: self.training_session_id,
            initial_cost,
            network_config,
            optimizer,
        };

        if let Some(ref output_dir) = self.full_session_output_directory {
            let mut full_output_path = output_dir.clone();
            full_output_path.push(path::Path::new("session-info.json"));
            let serialized_graph_json_string =
                serde_json::to_string_pretty(&training_session).unwrap();
            fs::write(&full_output_path, serialized_graph_json_string)?;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                String::from("you need to call create_training_log_directory() to complete the setup of the training session logger"),
            ));
        }

        Ok(())
    }

    pub fn write_update(
        &self,
        epoch: usize,
        epochs_completed: usize,
        training_set_cost: f64,
        test_set_cost: f64,
    ) -> Result<(), std::io::Error> {
        let training_update = TrainingUpdate {
            epoch,
            epochs_completed,
            timestamp_epoch: epoch_timestamp(),
            training_set_cost,
            test_set_cost,
        };

        if let Some(ref output_dir) = self.full_session_output_directory {
            let mut full_output_path = output_dir.clone();
            full_output_path.push(path::Path::new(&format!("epoch-{}.json", epoch)));
            let serialized_graph_json_string =
                serde_json::to_string_pretty(&training_update).unwrap();
            fs::write(&full_output_path, serialized_graph_json_string)?;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                String::from("you need to call create_training_log_directory() to complete the setup of the training session logger"),
            ));
        }

        Ok(())
    }
}
