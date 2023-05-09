#![allow(dead_code)]

use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Counter {
    count: u32,
}

impl Counter {
    pub fn count(&mut self) {
        self.count += 1;
    }

    pub fn get_total(&self) -> u32 {
        self.count
    }
}

pub fn epoch_timestamp() -> u128 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_nanos()
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SimpleTimer {
    pub name: String,
    state: SimpleTimerState,
    start_time_ms: u128,
    stop_time_ms: u128,
}

#[derive(Debug, PartialEq)]
pub enum TimerError {
    AlreadyStarted,
    NotStarted,
    AlreadyStopped,
    NotStopped,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum SimpleTimerState {
    NotStarted,
    Started,
    Stopped,
}

impl SimpleTimer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            state: SimpleTimerState::NotStarted,
            start_time_ms: 0,
            stop_time_ms: 0,
        }
    }

    pub fn start_new(name: &str) -> Self {
        let mut timer = Self::new(name);
        timer.start();
        timer
    }

    pub fn start_or_err(&mut self) -> Result<(), TimerError> {
        match &self.state {
            SimpleTimerState::NotStarted => {
                self.start_time_ms = epoch_timestamp();
                self.state = SimpleTimerState::Started;
                Ok(())
            }
            SimpleTimerState::Started => Err(TimerError::AlreadyStarted),
            SimpleTimerState::Stopped => Err(TimerError::AlreadyStopped),
        }
    }

    pub fn start(&mut self) {
        match &self.state {
            SimpleTimerState::NotStarted => {
                self.start_time_ms = epoch_timestamp();
                self.state = SimpleTimerState::Started;
            }
            SimpleTimerState::Started => panic!("Invalid timer use - timer already started"),
            SimpleTimerState::Stopped => panic!("Invalid timer use - timer already stopped"),
        }
    }

    pub fn stop_or_err(&mut self) -> Result<(), TimerError> {
        match &self.state {
            SimpleTimerState::NotStarted => Err(TimerError::NotStarted),
            SimpleTimerState::Started => {
                self.stop_time_ms = epoch_timestamp();
                self.state = SimpleTimerState::Stopped;
                Ok(())
            }
            SimpleTimerState::Stopped => Err(TimerError::AlreadyStopped),
        }
    }

    pub fn stop(&mut self) {
        match &self.state {
            SimpleTimerState::NotStarted => panic!("Invalid timer use - timer not started"),
            SimpleTimerState::Started => {
                self.stop_time_ms = epoch_timestamp();
                self.state = SimpleTimerState::Stopped;
            }
            SimpleTimerState::Stopped => panic!("Invalid timer use - timer already stopped"),
        }
    }

    pub fn get_total_nanoseconds(&self) -> Result<u128, TimerError> {
        match &self.state {
            SimpleTimerState::NotStarted => Err(TimerError::NotStarted),
            SimpleTimerState::Started => Err(TimerError::NotStopped),
            SimpleTimerState::Stopped => Ok(self.stop_time_ms - self.start_time_ms),
        }
    }

    pub fn get_total_milliseconds(&self) -> Result<u128, TimerError> {
        let total_ms = self.get_total_nanoseconds()?;
        let total_nanos = total_ms / 1_000_000;
        Ok(total_nanos)
    }
}

impl fmt::Display for SimpleTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let maybe_total_time = self.get_total_milliseconds();
        match maybe_total_time {
            Ok(total_time) => write!(f, "{}: {} ms", self.name, total_time),
            Err(_) => write!(f, "{}: not in stopped state.", self.name),
        }
    }
}

pub struct MultiPointTimerCollection {
    mpts: HashMap<String, MultiPointTimer>,
}

impl MultiPointTimerCollection {
    pub fn new() -> Self {
        Self {
            mpts: HashMap::new(),
        }
    }

    pub fn get_multi_point_timer(&mut self, name: &str) -> &mut MultiPointTimer {
        if self.mpts.contains_key(name) {
            self.mpts.get_mut(name).unwrap()
        } else {
            let mtp = MultiPointTimer::new(name);
            self.mpts.insert(name.to_owned(), mtp);
            let mtp = self.mpts.get_mut(name).unwrap();
            mtp
        }
    }
}

pub struct MultiPointTimer {
    name: String,
    instances: Vec<SimpleTimer>,
}

impl MultiPointTimer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            instances: Vec::new(),
        }
    }

    pub fn get_instance(&mut self) -> &mut SimpleTimer {
        let inst = SimpleTimer::new(&self.name);
        self.instances.push(inst);
        self.instances.last_mut().unwrap()
    }

    pub fn start_instance(&mut self) -> &mut SimpleTimer {
        let inst = SimpleTimer::new(&self.name);
        self.instances.push(inst);
        let the_instance = self.instances.last_mut().unwrap();
        the_instance.start();
        the_instance
    }

    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    pub fn get_total_nanoseconds(&self) -> u128 {
        let total = self
            .instances
            .iter()
            .filter(|i| i.state == SimpleTimerState::Stopped)
            .fold(0, |acc, i| acc + (*i).stop_time_ms - (*i).start_time_ms);
        total
    }

    pub fn get_total_milliseconds(&self) -> u128 {
        let total_nanos = self.get_total_nanoseconds();
        total_nanos / 1_000_000
    }
}
