use std::cmp::Ordering;
use std::collections::HashSet;

use crate::metrics::Metrics;

const BLOCK_SIZE_BYTES: usize = 16 * 1024 * 1024;
const SINK_BLOCKS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Vram,
    Ram,
    Ssd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    RecentOnly,
    SinksRecent,
    HeavyHitter,
    SinksHeavyHitter,
}

impl Policy {
    pub fn from_name(name: &str) -> Self {
        match name {
            "recent_only" => Self::RecentOnly,
            "sinks_recent" => Self::SinksRecent,
            "heavy_hitter" => Self::HeavyHitter,
            "sinks_heavy_hitter" => Self::SinksHeavyHitter,
            _ => Self::HeavyHitter,
        }
    }
}

#[derive(Clone, Debug)]
pub struct KvBlock {
    pub id: u64,
    pub size: usize,
    pub score: f32,
    pub tier: Tier,
    pub pinned: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PagerConfig {
    pub vram: usize,
    pub ram: usize,
    pub recent_window: usize,
    pub rebalance_interval: u64,
    pub promote_margin: f32,
    pub ram_promote_margin: f32,
    pub policy: Policy,
}

pub struct Pager {
    cfg: PagerConfig,
    metrics: Metrics,
    blocks: Vec<KvBlock>,
}

impl Pager {
    pub fn new(
        vram: usize,
        ram: usize,
        recent_window: usize,
        rebalance_interval: u64,
        promote_margin: f32,
        ram_promote_margin: f32,
        policy: String,
    ) -> Self {
        Self {
            cfg: PagerConfig {
                vram,
                ram,
                recent_window,
                rebalance_interval,
                promote_margin,
                ram_promote_margin,
                policy: Policy::from_name(&policy),
            },
            metrics: Metrics::default(),
            blocks: Vec::new(),
        }
    }

    pub fn on_step(&mut self, token_idx: u64, _layer: u32, attention: Vec<f32>) {
        self.metrics.inc_tokens();

        self.ensure_blocks(attention.len());
        self.update_scores(&attention);
        self.update_attention_quality(&attention);
        self.pin_blocks(token_idx);

        if self.metrics.tokens == 1 || self.metrics.tokens % self.cfg.rebalance_interval == 0 {
            self.place_blocks();
            self.update_memory_peaks();
        }
    }

    pub fn metrics(&self) -> Metrics {
        self.metrics.snapshot()
    }

    pub fn tiers(&self) -> Vec<u8> {
        self.blocks
            .iter()
            .map(|block| match block.tier {
                Tier::Vram => 0,
                Tier::Ram => 1,
                Tier::Ssd => 2,
            })
            .collect()
    }

    pub fn vram_block_ids(&self) -> Vec<u64> {
        self.blocks
            .iter()
            .filter(|block| block.tier == Tier::Vram)
            .map(|block| block.id)
            .collect()
    }

    fn ensure_blocks(&mut self, count: usize) {
        while self.blocks.len() < count {
            let id = self.blocks.len() as u64;
            self.blocks.push(KvBlock {
                id,
                size: BLOCK_SIZE_BYTES,
                score: 0.0,
                tier: Tier::Ram,
                pinned: false,
            });
        }
    }

    fn update_scores(&mut self, attention: &[f32]) {
        for (block, score) in self.blocks.iter_mut().zip(attention.iter()) {
            block.score = block.score * 0.90 + *score * 0.10;
        }
    }

    fn update_attention_quality(&mut self, attention: &[f32]) {
        let mut total = 0.0_f64;
        let mut vram = 0.0_f64;

        for (block, score) in self.blocks.iter().zip(attention.iter()) {
            let score = *score as f64;
            total += score;

            if block.tier == Tier::Vram {
                vram += score;
            }
        }

        self.metrics.add_attention_mass(total, vram);
    }

    fn pin_blocks(&mut self, token_idx: u64) {
        let recent_start = token_idx.saturating_sub(self.cfg.recent_window as u64);

        for block in &mut self.blocks {
            let is_recent = block.id >= recent_start;
            let is_sink = block.id < SINK_BLOCKS as u64;

            block.pinned = match self.cfg.policy {
                Policy::RecentOnly => is_recent,
                Policy::SinksRecent | Policy::HeavyHitter | Policy::SinksHeavyHitter => {
                    is_recent || is_sink
                }
            };
        }
    }

    fn place_blocks(&mut self) {
        let old_tiers: Vec<Tier> = self.blocks.iter().map(|block| block.tier).collect();
        let max_vram_blocks = self.cfg.vram / BLOCK_SIZE_BYTES;
        let max_ram_blocks = self.cfg.ram / BLOCK_SIZE_BYTES;

        match self.cfg.policy {
            Policy::RecentOnly | Policy::SinksRecent => {
                self.place_baseline_blocks(&old_tiers, max_vram_blocks, max_ram_blocks);
            }
            Policy::HeavyHitter => {
                self.place_heavy_hitter(&old_tiers, max_vram_blocks, max_ram_blocks);
            }
            Policy::SinksHeavyHitter => {
                self.place_sinks_heavy_hitter(&old_tiers, max_vram_blocks, max_ram_blocks);
            }
        }
    }

    fn place_baseline_blocks(
        &mut self,
        old_tiers: &[Tier],
        max_vram_blocks: usize,
        max_ram_blocks: usize,
    ) {
        let mut block_ids: Vec<usize> = (0..self.blocks.len()).collect();

        block_ids.sort_by(|&a, &b| {
            self.blocks[b]
                .pinned
                .cmp(&self.blocks[a].pinned)
                .then_with(|| self.blocks[a].id.cmp(&self.blocks[b].id))
        });

        self.reset_tiers(Tier::Ssd);
        self.assign_ranked_blocks(block_ids, max_vram_blocks, max_ram_blocks);
        self.count_tier_changes(old_tiers);
    }

    fn place_heavy_hitter(
        &mut self,
        old_tiers: &[Tier],
        max_vram_blocks: usize,
        max_ram_blocks: usize,
    ) {
        let has_vram_blocks = self.blocks.iter().any(|block| block.tier == Tier::Vram);

        if !has_vram_blocks {
            let mut block_ids: Vec<usize> = (0..self.blocks.len()).collect();

            block_ids.sort_by(|&a, &b| {
                self.blocks[b]
                    .pinned
                    .cmp(&self.blocks[a].pinned)
                    .then_with(|| compare_scores(self.blocks[b].score, self.blocks[a].score))
            });

            self.assign_ranked_blocks(block_ids, max_vram_blocks, max_ram_blocks);
            self.count_tier_changes(old_tiers);
            return;
        }

        for block in &mut self.blocks {
            if block.pinned {
                block.tier = Tier::Vram;
            }
        }

        self.enforce_vram_budget_keep_best(max_vram_blocks);
        self.promote_vram_with_hysteresis(max_vram_blocks);
        self.fill_ram(max_ram_blocks);
        self.promote_ram_with_hysteresis(max_ram_blocks);
        self.count_tier_changes(old_tiers);
    }

    fn place_sinks_heavy_hitter(
        &mut self,
        old_tiers: &[Tier],
        max_vram_blocks: usize,
        max_ram_blocks: usize,
    ) {
        self.reset_tiers(Tier::Ssd);

        if max_vram_blocks == 0 {
            self.count_tier_changes(old_tiers);
            return;
        }

        let mut selected_vram = HashSet::new();

        for block in self.blocks.iter().take(SINK_BLOCKS) {
            if selected_vram.len() >= max_vram_blocks {
                break;
            }
            selected_vram.insert(block.id);
        }

        let remaining_after_sinks = max_vram_blocks.saturating_sub(selected_vram.len());
        let recent_budget = self
            .cfg
            .recent_window
            .min((max_vram_blocks / 2).max(1))
            .min(remaining_after_sinks);
        let recent_start = self.blocks.len().saturating_sub(recent_budget);

        for block in self.blocks.iter().skip(recent_start) {
            if selected_vram.len() >= max_vram_blocks {
                break;
            }
            selected_vram.insert(block.id);
        }

        let mut scored_blocks: Vec<(u64, f32)> = self
            .blocks
            .iter()
            .filter(|block| !selected_vram.contains(&block.id))
            .map(|block| (block.id, block.score))
            .collect();

        scored_blocks.sort_by(|a, b| compare_scores(b.1, a.1));

        for (block_id, _) in scored_blocks {
            if selected_vram.len() >= max_vram_blocks {
                break;
            }
            selected_vram.insert(block_id);
        }

        for block in &mut self.blocks {
            if selected_vram.contains(&block.id) {
                block.tier = Tier::Vram;
            }
        }

        self.fill_ram(max_ram_blocks);
        self.count_tier_changes(old_tiers);
    }

    fn assign_ranked_blocks(
        &mut self,
        block_ids: Vec<usize>,
        max_vram_blocks: usize,
        max_ram_blocks: usize,
    ) {
        for (rank, idx) in block_ids.into_iter().enumerate() {
            self.blocks[idx].tier = if rank < max_vram_blocks {
                Tier::Vram
            } else if rank < max_vram_blocks + max_ram_blocks {
                Tier::Ram
            } else {
                Tier::Ssd
            };
        }
    }

    fn reset_tiers(&mut self, tier: Tier) {
        for block in &mut self.blocks {
            block.tier = tier;
        }
    }

    fn enforce_vram_budget_keep_best(&mut self, max_vram_blocks: usize) {
        let mut vram_ids: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier == Tier::Vram)
            .map(|(idx, _)| idx)
            .collect();

        if vram_ids.len() <= max_vram_blocks {
            return;
        }

        vram_ids.sort_by(|&a, &b| {
            self.blocks[b]
                .pinned
                .cmp(&self.blocks[a].pinned)
                .then_with(|| compare_scores(self.blocks[b].score, self.blocks[a].score))
        });

        for idx in vram_ids.into_iter().skip(max_vram_blocks) {
            self.blocks[idx].tier = Tier::Ram;
        }
    }

    fn promote_vram_with_hysteresis(&mut self, max_vram_blocks: usize) {
        loop {
            let vram_count = self
                .blocks
                .iter()
                .filter(|block| block.tier == Tier::Vram)
                .count();

            if vram_count < max_vram_blocks {
                if let Some(best) = self.best_non_vram_block() {
                    self.blocks[best].tier = Tier::Vram;
                    continue;
                }
                break;
            }

            let Some(worst) = self.worst_unpinned_vram_block() else {
                break;
            };
            let Some(best) = self.best_non_vram_block() else {
                break;
            };

            let improvement = self.blocks[best].score - self.blocks[worst].score;
            if improvement <= self.cfg.promote_margin {
                break;
            }

            let previous_best_tier = self.blocks[best].tier;
            self.blocks[best].tier = Tier::Vram;
            self.blocks[worst].tier = previous_best_tier;
        }
    }

    fn fill_ram(&mut self, max_ram_blocks: usize) {
        let current_ram_count = self
            .blocks
            .iter()
            .filter(|block| block.tier == Tier::Ram)
            .count();

        if current_ram_count >= max_ram_blocks {
            self.trim_ram(max_ram_blocks);
            return;
        }

        let mut candidates: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier == Tier::Ssd)
            .map(|(idx, _)| idx)
            .collect();

        candidates.sort_by(|&a, &b| compare_scores(self.blocks[b].score, self.blocks[a].score));

        for idx in candidates.into_iter().take(max_ram_blocks - current_ram_count) {
            self.blocks[idx].tier = Tier::Ram;
        }
    }

    fn trim_ram(&mut self, max_ram_blocks: usize) {
        let mut ram_ids: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier == Tier::Ram)
            .map(|(idx, _)| idx)
            .collect();

        if ram_ids.len() <= max_ram_blocks {
            return;
        }

        ram_ids.sort_by(|&a, &b| compare_scores(self.blocks[a].score, self.blocks[b].score));

        let overflow = ram_ids.len() - max_ram_blocks;
        for idx in ram_ids.into_iter().take(overflow) {
            self.blocks[idx].tier = Tier::Ssd;
        }
    }

    fn promote_ram_with_hysteresis(&mut self, max_ram_blocks: usize) {
        loop {
            let ram_count = self
                .blocks
                .iter()
                .filter(|block| block.tier == Tier::Ram)
                .count();

            if ram_count < max_ram_blocks {
                break;
            }

            let Some(worst) = self.worst_ram_block() else {
                break;
            };
            let Some(best) = self.best_ssd_block() else {
                break;
            };

            let improvement = self.blocks[best].score - self.blocks[worst].score;
            if improvement <= self.cfg.ram_promote_margin {
                break;
            }

            self.blocks[best].tier = Tier::Ram;
            self.blocks[worst].tier = Tier::Ssd;
        }
    }

    fn worst_unpinned_vram_block(&self) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier == Tier::Vram && !block.pinned)
            .min_by(|(_, a), (_, b)| compare_scores(a.score, b.score))
            .map(|(idx, _)| idx)
    }

    fn best_non_vram_block(&self) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier != Tier::Vram)
            .max_by(|(_, a), (_, b)| compare_scores(a.score, b.score))
            .map(|(idx, _)| idx)
    }

    fn worst_ram_block(&self) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier == Tier::Ram)
            .min_by(|(_, a), (_, b)| compare_scores(a.score, b.score))
            .map(|(idx, _)| idx)
    }

    fn best_ssd_block(&self) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.tier == Tier::Ssd)
            .max_by(|(_, a), (_, b)| compare_scores(a.score, b.score))
            .map(|(idx, _)| idx)
    }

    fn count_tier_changes(&mut self, old_tiers: &[Tier]) {
        for (block, old_tier) in self.blocks.iter().zip(old_tiers.iter()) {
            match (*old_tier, block.tier) {
                (Tier::Vram, Tier::Ram) | (Tier::Ram, Tier::Vram) => {
                    self.metrics.swap_vram_ram += block.size as u64;
                }
                (Tier::Ram, Tier::Ssd) | (Tier::Ssd, Tier::Ram) => {
                    self.metrics.swap_ram_ssd += block.size as u64;
                }
                (Tier::Vram, Tier::Ssd) | (Tier::Ssd, Tier::Vram) => {
                    self.metrics.swap_vram_ram += block.size as u64;
                    self.metrics.swap_ram_ssd += block.size as u64;
                }
                _ => {}
            }
        }
    }

    fn update_memory_peaks(&mut self) {
        let vram_used: usize = self
            .blocks
            .iter()
            .filter(|block| block.tier == Tier::Vram)
            .map(|block| block.size)
            .sum();

        let ram_used: usize = self
            .blocks
            .iter()
            .filter(|block| block.tier == Tier::Ram)
            .map(|block| block.size)
            .sum();

        self.metrics.vram_peak = self.metrics.vram_peak.max(vram_used);
        self.metrics.ram_peak = self.metrics.ram_peak.max(ram_used);
    }
}

fn compare_scores(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}
