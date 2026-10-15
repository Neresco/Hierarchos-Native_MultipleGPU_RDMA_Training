//! Golden output digests for the kernel parity harnesses.
//!
//! A parity harness normally proves a kernel change is bit-exact by comparing
//! against the previous artifact directory. That proof is unavailable exactly
//! where it matters most: CI, or any machine that does not ship the pre-change
//! `.spv` files. Recording a digest of every invocation's output vector turns
//! the same proof into a portable one — hash today's outputs and compare them
//! with the recorded digests, with no baseline directory present.
//!
//! Digests are keyed by device because bit-identical arithmetic is only
//! guaranteed within one driver and ISA. A golden file therefore accumulates one
//! digest per device name, so an AMD, Intel, or NVIDIA runner each check the
//! entry recorded on their own hardware; an invocation recorded on a different
//! device is reported as unverified instead of quietly passing.
//!
//! This module is included by path from the harness binaries, the same way they
//! include `vulkan.rs`, so it adds no library surface.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bump when the digest contract changes; an older file then fails loudly
/// instead of comparing incomparable digests.
pub const SCHEMA: u32 = 1;

#[derive(Default, Serialize, Deserialize)]
pub struct GoldenFile {
    pub schema: u32,
    /// Invocation label -> device name -> hex digest.
    pub entries: BTreeMap<String, BTreeMap<String, String>>,
}

/// Hex SHA-256 over the little-endian bit patterns of one output vector.
pub fn digest_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Hex SHA-256 over one byte range, used for buffers read as `u32`.
pub fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Digest of several output vectors that belong to one invocation.
pub fn digest_parts(parts: &[&[f32]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((parts.len() as u64).to_le_bytes());
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        for value in part.iter() {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Clone, Copy, PartialEq)]
pub enum GoldenMode {
    Compare,
    Write,
}

pub struct Golden {
    mode: GoldenMode,
    path: PathBuf,
    file: GoldenFile,
    device: String,
    strict: bool,
    matched: usize,
    recorded: usize,
    unverified: usize,
    mismatches: Vec<String>,
    missing: Vec<String>,
}

impl Golden {
    pub fn compare(path: impl AsRef<Path>, device: &str) -> Result<Self> {
        Self::load(path.as_ref(), device, GoldenMode::Compare)
    }

    pub fn write(path: impl AsRef<Path>, device: &str) -> Result<Self> {
        Self::load(path.as_ref(), device, GoldenMode::Write)
    }

    fn load(path: &Path, device: &str, mode: GoldenMode) -> Result<Self> {
        let file = if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading golden file {}", path.display()))?;
            let parsed: GoldenFile = serde_json::from_str(&text)
                .with_context(|| format!("parsing golden file {}", path.display()))?;
            if parsed.schema != SCHEMA {
                bail!(
                    "golden file {} has schema {}, this harness records schema {SCHEMA}",
                    path.display(),
                    parsed.schema
                );
            }
            parsed
        } else {
            if mode == GoldenMode::Compare {
                bail!(
                    "golden file {} does not exist; record it first with --write-golden",
                    path.display()
                );
            }
            GoldenFile { schema: SCHEMA, ..Default::default() }
        };
        Ok(Self {
            mode,
            path: path.to_path_buf(),
            file,
            device: device.to_string(),
            strict: false,
            matched: 0,
            recorded: 0,
            unverified: 0,
            mismatches: Vec::new(),
            missing: Vec::new(),
        })
    }

    /// Refuse to pass when any invocation had no digest recorded for this
    /// device. A CI runner on unseen hardware would otherwise verify nothing
    /// and still exit zero.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Record or verify one invocation's outputs. `label` must identify the
    /// kernel and its case; the caller owns the naming convention.
    pub fn observe_digest(&mut self, label: &str, digest: String) {
        match self.mode {
            GoldenMode::Write => {
                self.file
                    .entries
                    .entry(label.to_string())
                    .or_default()
                    .insert(self.device.clone(), digest);
                self.recorded += 1;
            }
            GoldenMode::Compare => {
                let Some(by_device) = self.file.entries.get(label) else {
                    self.missing.push(label.to_string());
                    return;
                };
                match by_device.get(&self.device) {
                    Some(expected) if *expected == digest => self.matched += 1,
                    Some(expected) => self.mismatches.push(format!(
                        "{label}: digest {digest} != recorded {expected} on {}",
                        self.device
                    )),
                    None => {
                        self.unverified += 1;
                        if self.unverified <= 8 {
                            eprintln!(
                                "golden unverified {label}: no digest recorded for {} (file has {})",
                                self.device,
                                by_device.keys().cloned().collect::<Vec<_>>().join(", ")
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn observe(&mut self, label: &str, values: &[f32]) {
        let digest = digest_f32(values);
        self.observe_digest(label, digest);
    }

    pub fn observe_parts(&mut self, label: &str, parts: &[&[f32]]) {
        let digest = digest_parts(parts);
        self.observe_digest(label, digest);
    }

    /// Write the accumulated digests, or report the comparison outcome. Any
    /// mismatch, or any invocation missing a digest for this device, fails.
    pub fn finish(mut self) -> Result<()> {
        match self.mode {
            GoldenMode::Write => {
                self.file.schema = SCHEMA;
                let text = serde_json::to_string_pretty(&self.file)?;
                std::fs::write(&self.path, text)
                    .with_context(|| format!("writing golden file {}", self.path.display()))?;
                println!(
                    "GOLDEN RECORDED: {} invocation(s) for {} -> {}",
                    self.recorded,
                    self.device,
                    self.path.display()
                );
                Ok(())
            }
            GoldenMode::Compare => {
                for mismatch in &self.mismatches {
                    eprintln!("golden mismatch {mismatch}");
                }
                for label in &self.missing {
                    eprintln!("golden missing {label}: no digest recorded for any device");
                }
                if self.strict && self.unverified > 0 {
                    bail!(
                        "{} invocation(s) have no digest recorded for {}; record them with --write-golden before enforcing --golden-strict",
                        self.unverified,
                        self.device
                    );
                }
                if !self.mismatches.is_empty() || !self.missing.is_empty() {
                    bail!(
                        "{} golden digest mismatch(es), {} unrecorded invocation(s), {} verified, {} unverified on another device",
                        self.mismatches.len(),
                        self.missing.len(),
                        self.matched,
                        self.unverified
                    );
                }
                println!(
                    "PASS: golden digests verified for {} invocation(s) on {} ({} unverified on another device)",
                    self.matched, self.device, self.unverified
                );
                Ok(())
            }
        }
    }
}

/// Parse the golden flags a harness exposes. Returns `(compare, write, strict)`.
pub fn golden_flags(arguments: &[String]) -> Result<(Option<PathBuf>, Option<PathBuf>, bool)> {
    let mut compare = None;
    let mut write = None;
    let mut strict = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--golden" => {
                compare = Some(PathBuf::from(
                    arguments.get(index + 1).context("missing golden file path")?,
                ));
                index += 2;
            }
            "--write-golden" => {
                write = Some(PathBuf::from(
                    arguments.get(index + 1).context("missing golden file path")?,
                ));
                index += 2;
            }
            "--golden-strict" => {
                strict = true;
                index += 1;
            }
            _ => index += 1,
        }
    }
    if compare.is_some() && write.is_some() {
        bail!("--golden and --write-golden are mutually exclusive");
    }
    if strict && compare.is_none() {
        bail!("--golden-strict needs a --golden file to verify against");
    }
    Ok((compare, write, strict))
}
