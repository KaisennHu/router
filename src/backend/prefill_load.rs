// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Polls a worker's `/load` endpoint for its remaining prefill work.
//!
//! The sample is worker-level: the same value applies to every data-parallel
//! rank of that worker, and it is only ever replaced by a newer `sample_seq`.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use serde::Deserialize;
use tokio::time::Instant;
use tracing::{debug, warn};

/// Latest remaining-prefill sample reported by one worker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillSample {
    /// Work across all requests the worker knows about; queued requests count
    /// as their prompt length.
    pub remaining_tokens: u64,
    /// Worker-side sample counter, used to reject stale or duplicated polls.
    pub sample_seq: u64,
    /// When this router received the sample.
    pub received_at: Instant,
}

impl PrefillSample {
    fn is_newer_than(&self, previous: Option<Self>) -> bool {
        previous.is_none_or(|previous| self.sample_seq > previous.sample_seq)
    }
}

#[derive(Debug, Deserialize)]
struct LoadResponse {
    #[serde(default)]
    prefill_tokens_remaining: Option<u64>,
    #[serde(default)]
    sample_seq: Option<u64>,
}

/// Latest sample per worker base URL.
#[derive(Debug, Default)]
pub struct PrefillLoadTable {
    samples: DashMap<String, PrefillSample>,
}

impl PrefillLoadTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Latest sample for `instance_id`, if the worker has reported one.
    pub fn get(&self, instance_id: &str) -> Option<PrefillSample> {
        self.samples.get(instance_id).map(|sample| *sample)
    }

    fn store(&self, instance_id: &str, sample: PrefillSample) {
        self.samples.insert(instance_id.to_owned(), sample);
    }
}

/// Poll one worker's `/load` until `shutdown` flips to `true`.
///
/// Failures are logged and retried on the next tick: a worker that cannot be
/// polled simply keeps whatever sample it last reported.
pub async fn poll_worker(
    client: reqwest::Client,
    instance_id: String,
    worker_url: String,
    table: Arc<PrefillLoadTable>,
    interval: Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    let url = format!("{}/load", worker_url.trim_end_matches('/'));
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            _ = ticker.tick() => {}
        }
        let response = match client.get(&url).send().await {
            Ok(response) => response,
            Err(err) => {
                debug!(%instance_id, error = %err, "prefill load poll failed");
                continue;
            }
        };
        if !response.status().is_success() {
            debug!(%instance_id, status = %response.status(), "prefill load poll rejected");
            continue;
        }
        let body: LoadResponse = match response.json().await {
            Ok(body) => body,
            Err(err) => {
                warn!(%instance_id, error = %err, "prefill load poll returned invalid JSON");
                continue;
            }
        };
        let (Some(remaining_tokens), Some(sample_seq)) =
            (body.prefill_tokens_remaining, body.sample_seq)
        else {
            // The worker predates the fields; keep whatever we have.
            continue;
        };
        let sample = PrefillSample {
            remaining_tokens,
            sample_seq,
            received_at: Instant::now(),
        };
        if sample.is_newer_than(table.get(&instance_id)) {
            table.store(&instance_id, sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrefillSample;

    #[test]
    fn newer_sample_wins_and_duplicates_are_ignored() {
        let first = PrefillSample {
            remaining_tokens: 10,
            sample_seq: 1,
            ..Default::default()
        };
        let duplicate = PrefillSample {
            remaining_tokens: 99,
            sample_seq: 1,
            ..Default::default()
        };
        let newer = PrefillSample {
            remaining_tokens: 5,
            sample_seq: 2,
            ..Default::default()
        };

        assert!(first.is_newer_than(None));
        assert!(!duplicate.is_newer_than(Some(first)));
        assert!(newer.is_newer_than(Some(first)));
    }
}
