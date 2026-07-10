use pyo3::prelude::*;
use serde::Serialize;

#[pyclass]
#[derive(Clone, Default, Serialize)]
pub struct Metrics {
    #[pyo3(get)]
    pub tokens: u64,

    #[pyo3(get)]
    pub vram_peak: usize,

    #[pyo3(get)]
    pub ram_peak: usize,

    #[pyo3(get)]
    pub swap_vram_ram: u64,

    #[pyo3(get)]
    pub swap_ram_ssd: u64,

    #[pyo3(get)]
    pub attention_mass_total: f64,

    #[pyo3(get)]
    pub attention_mass_vram: f64,
}

impl Metrics {
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    pub fn inc_tokens(&mut self) {
        self.tokens += 1;
    }

    pub fn add_attention_mass(&mut self, total: f64, vram: f64) {
        self.attention_mass_total += total;
        self.attention_mass_vram += vram;
    }
}
