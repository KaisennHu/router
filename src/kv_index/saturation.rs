//! Engine saturation, read from gauges older frontends already publish.
//!
//! `vllm:cache_config_info` gives an absolute capacity, but only recent builds export
//! it. The gauges below exist on older ones too, and they say the same thing in relative
//! terms: how full the KV pool is, and whether the engine is already queuing on capacity.
//! Admission against them needs no configured number, which is what "discover it" means
//! for a deployment that publishes no absolute capacity.

use crate::protocols::prometheus_text::for_each_sample;

/// Share of the KV pool at which an engine stops being a candidate. Below 100 because an
/// engine needs room to finish what it already holds; a run that reached 96.8% is what
/// produced the measured TPOT tail.
pub(crate) const DECODE_MAX_KV_USAGE: f64 = 0.90;

/// Watermark in force, set once from `--decode-max-kv-usage`.
///
/// Stored as bits so the read is a plain atomic load on the request path.
static WATERMARK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(DECODE_MAX_KV_USAGE.to_bits());

/// Override the watermark. Values outside `0.0..=1.0` are ignored, so a bad flag cannot
/// disable admission silently.
pub fn set_decode_max_kv_usage(value: f64) {
    if (0.0..=1.0).contains(&value) {
        WATERMARK.store(value.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

fn watermark() -> f64 {
    f64::from_bits(WATERMARK.load(std::sync::atomic::Ordering::Relaxed))
}

/// What one engine reports about its own pressure.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Saturation {
    /// KV pool usage as a fraction, when the engine publishes it.
    pub(crate) kv_usage: Option<f64>,
    /// Requests in model-execution batches.
    pub(crate) running: u64,
    /// Requests queued for scheduling, including those queued on capacity.
    pub(crate) waiting: u64,
    /// Requests queued specifically because the engine ran out of KV capacity.
    pub(crate) waiting_capacity: u64,
}

impl Saturation {
    /// Whether this engine should stop receiving new Decode work.
    ///
    /// A queue that already names capacity as its reason is stronger evidence than the
    /// usage gauge: the engine has decided it cannot place another sequence.
    pub(crate) fn is_saturated(&self) -> bool {
        self.waiting_capacity > 0 || self.kv_usage.is_some_and(|usage| usage >= watermark())
    }
}

/// Parse the gauges out of one `/metrics` body.
pub(crate) fn parse(text: &str) -> Saturation {
    let mut saturation = Saturation::default();
    for_each_sample(text, |name, labels, value| match name {
        "vllm:kv_cache_usage_perc" => {
            saturation.kv_usage = value.parse::<f64>().ok().map(|usage| usage.clamp(0.0, 1.0));
        }
        "vllm:num_requests_running" => {
            if let Ok(parsed) = value.parse::<f64>() {
                saturation.running = saturation.running.saturating_add(parsed.max(0.0) as u64);
            }
        }
        "vllm:num_requests_waiting" => {
            if let Ok(parsed) = value.parse::<f64>() {
                saturation.waiting = saturation.waiting.saturating_add(parsed.max(0.0) as u64);
            }
        }
        "vllm:num_requests_waiting_by_reason" => {
            let reason = labels.get("reason").map(String::as_str).unwrap_or_default();
            if reason == "capacity" {
                if let Ok(parsed) = value.parse::<f64>() {
                    saturation.waiting_capacity = saturation
                        .waiting_capacity
                        .saturating_add(parsed.max(0.0) as u64);
                }
            }
        }
        _ => {}
    });
    saturation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pressure_gauges() {
        let text = "\
# HELP vllm:kv_cache_usage_perc usage
vllm:kv_cache_usage_perc{engine=\"0\"} 0.93
vllm:num_requests_running{engine=\"0\"} 7
vllm:num_requests_waiting{engine=\"0\"} 2
vllm:num_requests_waiting_by_reason{engine=\"0\",reason=\"capacity\"} 1
vllm:num_requests_waiting_by_reason{engine=\"0\",reason=\"deferred\"} 1
";
        let saturation = parse(text);
        assert_eq!(saturation.kv_usage, Some(0.93));
        assert_eq!(saturation.running, 7);
        assert_eq!(saturation.waiting, 2);
        assert_eq!(saturation.waiting_capacity, 1);
        assert!(saturation.is_saturated(), "capacity queue means saturated");
    }

    #[test]
    fn healthy_engine_is_not_saturated() {
        let text = "\
vllm:kv_cache_usage_perc{engine=\"0\"} 0.42
vllm:num_requests_running{engine=\"0\"} 3
vllm:num_requests_waiting{engine=\"0\"} 0
vllm:num_requests_waiting_by_reason{engine=\"0\",reason=\"capacity\"} 0
";
        let saturation = parse(text);
        assert!(!saturation.is_saturated());
        assert_eq!(saturation.running, 3);
    }

    #[test]
    fn usage_alone_can_saturate() {
        let text = "vllm:kv_cache_usage_perc{engine=\"0\"} 0.95\n";
        assert!(parse(text).is_saturated());
    }

    #[test]
    fn missing_gauges_stay_unsaturated() {
        let saturation = parse("# nothing useful\n");
        assert_eq!(saturation, Saturation::default());
        assert!(!saturation.is_saturated());
    }
}
