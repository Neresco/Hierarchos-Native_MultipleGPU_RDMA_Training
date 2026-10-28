//! ZeRO-3 sharding support for the Vulkan transformer training path.
//!
//! The design is documented in [`DESIGN_ZER03.md`](../DESIGN_ZER03.md). The core
//! idea: the full model is represented as one flat f32 buffer (the concatenation
//! of every trainable's `values` in `collect_trainables` order, plus the shared
//! embedding). That flat buffer is sharded into `world_size` contiguous slices;
//! rank `r` persists only slice `r` (in VRAM or the offload store). Before a
//! chunk of layers is computed, its slices are all-gathered back into the full
//! buffer; after the chunk executes the gathered memory is freed and the rank
//! reverts to its slice.
//!
//! This module holds the pure sharding math (unit-testable without a GPU) and
//! the host-side layout bookkeeping. The GPU gather/scatter and the integration
//! with the forward/backward live in `transformer.rs`.

/// Describe one parameter's slice inside the flat values buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSlot {
    /// Inclusive start (f32 elements) in the flat buffer.
    pub start: usize,
    /// Element count (the parameter's `len`).
    pub len: usize,
    /// Index in the collect order (stable identity for logging).
    pub index: usize,
}

impl ParamSlot {
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }
}

/// A sharding layout over a flat buffer of `total` f32 elements, split into
/// `world_size` equal contiguous slices.
#[derive(Debug, Clone)]
pub struct ShardLayout {
    total: usize,
    world_size: usize,
}

impl ShardLayout {
    pub fn new(total: usize, world_size: usize) -> Self {
        assert!(world_size >= 1);
        assert!(total > 0, "flat buffer must be non-empty");
        Self { total, world_size }
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn total(&self) -> usize {
        self.total
    }

    /// Length of one slice (ceil division; the last slice may be shorter).
    pub fn slice_len(&self) -> usize {
        (self.total + self.world_size - 1) / self.world_size
    }

    /// The range owned by rank `rank`. Clamped so ranks beyond a short buffer
    /// get an empty (contiguous) slice rather than a negative range.
    pub fn slice_range(&self, rank: usize) -> std::ops::Range<usize> {
        let sl = self.slice_len();
        let start = (rank * sl).min(self.total);
        let end = ((rank + 1) * sl).min(self.total);
        start..end
    }

    /// Number of equal slices (== world_size).
    pub fn segment_count(&self) -> usize {
        self.world_size
    }

    /// Segments in owner order for `ring_allgather_sharded` (seg[r] = slice r).
    pub fn segments(&self) -> Vec<std::ops::Range<usize>> {
        (0..self.world_size).map(|r| self.slice_range(r)).collect()
    }
}

/// Chunk the flat buffer into pieces whose total bytes stay under
/// `chunk_bytes`. Chunks are contiguous ranges and partition the whole buffer.
pub fn chunk_ranges(total: usize, chunk_bytes: usize) -> Vec<std::ops::Range<usize>> {
    assert!(total > 0 && chunk_bytes > 0);
    let elem_bytes = 4usize; // f32
    let per_chunk = (chunk_bytes / elem_bytes).max(1);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total {
        let end = (start + per_chunk).min(total);
        out.push(start..end);
        start = end;
    }
    out
}

/// Given a set of param slots and a flat chunk range, return the slots that
/// intersect the chunk (used to know which parameters to materialize for a
/// chunked gather).
pub fn slots_in_chunk(slots: &[ParamSlot], chunk: &std::ops::Range<usize>) -> Vec<ParamSlot> {
    slots
        .iter()
        .filter(|s| s.range().end > chunk.start && s.range().start < chunk.end)
        .copied()
        .collect()
}

/// Group layers into chunks whose combined value bytes stay under `budget`.
///
/// `layer_sizes` is the number of f32 elements per layer (in order). Returns
/// `(start_layer, end_layer)` half-open ranges; each range's total elements fit
/// the budget. Layers larger than the budget get their own chunk.
pub fn layer_chunks(layer_sizes: &[usize], budget: usize) -> Vec<(usize, usize)> {
    assert!(budget > 0);
    let budget_elems = budget / 4;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;
    for (i, &size) in layer_sizes.iter().enumerate() {
        if acc + size > budget_elems && acc > 0 {
            chunks.push((start, i));
            start = i;
            acc = 0;
        }
        acc += size;
    }
    if start < layer_sizes.len() {
        chunks.push((start, layer_sizes.len()));
    }
    chunks
}

/// Host-side storage of one rank's slice of the flat buffer, backed by a plain
/// Vec (which can itself be placed in the offload store for RAM/SSD persistence).
#[derive(Debug)]
pub struct ShardStore {
    /// This rank's slice, contiguous.
    data: Vec<f32>,
    layout: ShardLayout,
    rank: usize,
}

impl ShardStore {
    /// Build this rank's slice by taking `layout.slice_range(rank)` out of the
    /// full host buffer `full`.
    pub fn from_full(full: &[f32], layout: &ShardLayout, rank: usize) -> Self {
        assert_eq!(full.len(), layout.total());
        let range = layout.slice_range(rank);
        let data = full[range].to_vec();
        Self { data, layout: layout.clone(), rank }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn slice_len(&self) -> usize {
        self.data.len()
    }

    /// View of this rank's slice.
    pub fn slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view of this rank's slice (to receive a reduce-scattered grad).
    pub fn slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_ranges_partition_the_buffer() {
        for total in [1usize, 7, 24, 100, 129] {
            for world in [1usize, 2, 3, 6] {
                let layout = ShardLayout::new(total, world);
                let segs = layout.segments();
                assert_eq!(segs.len(), world);
                // contiguous and covering.
                assert_eq!(segs[0].start, 0);
                assert_eq!(segs[world - 1].end, total);
                for w in 0..world - 1 {
                    assert_eq!(segs[w].end, segs[w + 1].start, "gap/overlap at {w}");
                }
                let sum: usize = segs.iter().map(|s| s.len()).sum();
                assert_eq!(sum, total);
            }
        }
    }

    #[test]
    fn chunk_ranges_cover_without_overlap() {
        for total in [1usize, 100, 101, 1000] {
            for cb in [4usize, 40, 4000] {
                let ranges = chunk_ranges(total, cb);
                let sum: usize = ranges.iter().map(|r| r.len()).sum();
                assert_eq!(sum, total);
                assert_eq!(ranges[0].start, 0);
                assert_eq!(ranges.last().unwrap().end, total);
            }
        }
    }

    #[test]
    fn shard_store_round_trips_through_gather() {
        // Simulate a world=3 gather of a 30-element buffer.
        let world = 3usize;
        let total = 30usize;
        let layout = ShardLayout::new(total, world);
        let full: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut stores: Vec<ShardStore> =
            (0..world).map(|r| ShardStore::from_full(&full, &layout, r)).collect();

        // Gather: place each store's slice into the reconstructed buffer.
        let mut gathered = vec![f32::NAN; total];
        for r in 0..world {
            let range = layout.slice_range(r);
            gathered[range].copy_from_slice(stores[r].slice());
        }
        assert_eq!(gathered, full);

        // Reduce-scatter: sum a per-rank delta into each store's slice.
        for r in 0..world {
            for v in stores[r].slice_mut().iter_mut() {
                *v += r as f32;
            }
        }
        let mut expected = vec![0f32; total];
        for r in 0..world {
            let range = layout.slice_range(r);
            for e in range {
                expected[e] = full[e] + r as f32;
            }
        }
        let mut checked = vec![f32::NAN; total];
        for r in 0..world {
            let range = layout.slice_range(r);
            checked[range].copy_from_slice(stores[r].slice());
        }
        assert_eq!(checked, expected);
    }

    #[test]
    fn layer_chunks_cover_all_under_budget() {
        let sizes = vec![10usize, 10, 10, 10, 10, 10]; // 6 layers, 10 elems each
        // budget 40 bytes = 10 elems -> one layer per chunk.
        let c = layer_chunks(&sizes, 40);
        let total: usize = c.iter().map(|(a, b)| b - a).sum();
        assert_eq!(total, 6);
        for (a, b) in &c {
            let elems: usize = sizes[*a..*b].iter().sum();
            assert!(elems * 4 <= 40);
        }
        // budget 300 bytes = 75 elems -> all 6 layers in one chunk.
        let c = layer_chunks(&sizes, 300);
        assert_eq!(c, vec![(0usize, 6)]);
        // a single huge layer gets its own chunk even if over budget.
        let c = layer_chunks(&vec![1000usize], 40);
        assert_eq!(c, vec![(0usize, 1)]);
    }

    #[test]
    fn slots_in_chunk_selects_intersections() {
        let slots = vec![
            ParamSlot { start: 0, len: 10, index: 0 },
            ParamSlot { start: 10, len: 10, index: 1 },
            ParamSlot { start: 20, len: 10, index: 2 },
        ];
        let picked = slots_in_chunk(&slots, &(5..25));
        let idx: Vec<usize> = picked.iter().map(|s| s.index).collect();
        assert_eq!(idx, vec![0, 1, 2]);
        let picked = slots_in_chunk(&slots, &(10..20));
        assert_eq!(picked.iter().map(|s| s.index).collect::<Vec<_>>(), vec![1]);
    }
}
