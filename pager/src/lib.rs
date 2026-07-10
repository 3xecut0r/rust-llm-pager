use pyo3::prelude::*;

mod core;
mod metrics;

use core::Pager as InnerPager;
use metrics::Metrics;

#[pyclass]
pub struct PyPager {
    inner: InnerPager,
}

#[pymethods]
impl PyPager {
    #[new]
    #[pyo3(signature = (
        vram,
        ram,
        recent,
        rebalance_interval = 32,
        promote_margin = 0.10,
        ram_promote_margin = 0.20,
        policy = "heavy_hitter"
    ))]
    fn new(
        vram: usize,
        ram: usize,
        recent: usize,
        rebalance_interval: u64,
        promote_margin: f32,
        ram_promote_margin: f32,
        policy: &str,
    ) -> Self {
        Self {
            inner: InnerPager::new(
                vram,
                ram,
                recent,
                rebalance_interval,
                promote_margin,
                ram_promote_margin,
                policy.to_string(),
            ),
        }
    }

    fn on_step(&mut self, token_idx: u64, layer: u32, attention: Vec<f32>) {
        self.inner.on_step(token_idx, layer, attention);
    }

    fn metrics(&self) -> Metrics {
        self.inner.metrics()
    }

    fn tiers(&self) -> Vec<u8> {
        self.inner.tiers()
    }

    fn vram_block_ids(&self) -> Vec<u64> {
        self.inner.vram_block_ids()
    }
}

#[pymodule]
fn pager(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyPager>()?;
    module.add_class::<Metrics>()?;
    Ok(())
}
