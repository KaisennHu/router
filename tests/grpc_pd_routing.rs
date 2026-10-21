//! P/D disaggregation over `grpc://` prefill and decode pools.
//!
//! Two in-process `Inference` mocks stand in for vLLM Rust workers. The
//! router must send P a unary one-token `Generate` with the connector's
//! `kv_transfer_params`, then send D a `GenerateStream` that carries P's
//! `FinishInfo.kv_transfer_params`. No GPU or KV connector is involved.
//!
//!   cargo test --test grpc_pd_routing

mod common;

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

use common::create_test_context;
use common::mock_vllm_rs::{MockVllmRs, MockVllmRsServer};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use vllm_router_rs::config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode};
use vllm_router_rs::policies::PolicyFactory;
use vllm_router_rs::protocols::spec::ChatCompletionRequest;
use vllm_router_rs::routers::http::vllm_pd_router::VllmPDRouter;
use vllm_router_rs::routers::RouterTrait;

fn prefill_kv() -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_engine_id": "P",
        "remote_block_ids": [[7, 8]],
        "remote_host": "prefill",
        "remote_port": 5600,
        "tp_size": 1,
    })
}

/// Process-wide recorder; series are labelled by each test's own mock URL.
fn metrics() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| PrometheusBuilder::new().install_recorder().unwrap())
}

fn worker_counter(name: &str, worker: &str) -> f64 {
    let prefix = format!("{name}{{worker=\"{worker}\"}} ");
    metrics()
        .render()
        .lines()
        .find_map(|line| line.strip_prefix(&prefix).and_then(|n| n.parse().ok()))
        .unwrap_or(0.0)
}

async fn pd_router(prefill: &MockVllmRsServer, decode: &MockVllmRsServer) -> VllmPDRouter {
    pd_router_with_dp(prefill, decode, 1).await
}

async fn pd_router_with_dp(
    prefill: &MockVllmRsServer,
    decode: &MockVllmRsServer,
    dp_size: usize,
) -> VllmPDRouter {
    metrics();
    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(prefill.grpc_url.clone(), None)],
            decode_urls: vec![decode.grpc_url.clone()],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        policy: PolicyConfig::RoundRobin,
        kv_connector: KvConnector::Nixl,
        worker_startup_timeout_secs: 5,
        intra_node_data_parallel_size: dp_size,
        ..RouterConfig::default()
    };
    vllm_router_rs::config::ConfigValidator::validate(&config).unwrap();
    let ctx = create_test_context(config);
    // Separate P and D policy instances, as `RouterFactory` installs them.
    let policies = &ctx.policy_registry;
    policies.set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
    policies.set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
    let router = VllmPDRouter::new(
        vec![(prefill.grpc_url.clone(), None)],
        vec![decode.grpc_url.clone()],
        None,
        &ctx,
    )
    .await
    .expect("grpc P/D workers should pass grpc.health.v1");
    router.pin_test_token_ids(vec![1, 2, 3]);
    router
}

fn chat(stream: bool) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

#[tokio::test]
async fn grpc_pd_hands_prefill_kv_params_to_decode() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;

    for stream in [false, true] {
        let response = router.route_chat(None, &chat(stream), None).await;
        assert!(response.status().is_success(), "stream={stream}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("hello from worker"),
            "stream={stream}: {body}"
        );
    }

    let p = prefill.captured();
    let d = decode.captured();
    assert_eq!((p.len(), d.len()), (2, 2));
    for (p, d) in p.iter().zip(&d) {
        assert!(p.unary, "prefill must be unary Generate");
        assert_eq!((p.max_new_tokens, p.min_new_tokens), (1, 1));
        let sent = p.kv_transfer_params.as_ref().expect("prefill kv params");
        assert_eq!(sent["do_remote_decode"], true);
        assert_eq!(sent["do_remote_prefill"], false);

        assert!(!d.unary, "decode must stream");
        assert_eq!(d.max_new_tokens, 8);
        assert_eq!(d.kv_transfer_params.as_ref(), Some(&prefill_kv()));
        assert_eq!(d.token_ids, p.token_ids);
        assert_eq!(d.request_id, p.request_id);
        assert!(!d.had_text_prompt && !p.had_text_prompt);
    }
}

#[tokio::test]
async fn grpc_pd_prefill_failure_skips_decode() {
    let prefill = MockVllmRsServer::spawn().await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    prefill.state.failures_remaining.store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(false), None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(prefill.captured().len(), 1);
    assert!(decode.captured().is_empty());
    let errors = "vllm_router_pd_prefill_errors_total";
    assert_eq!(worker_counter(errors, &prefill.grpc_url), 1.0);
}

#[tokio::test]
async fn grpc_pd_decode_failure_after_prefill_records_decode_error() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;
    decode.state.failures_remaining.store(1, Ordering::Relaxed);

    let response = router.route_chat(None, &chat(false), None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    let (p, d) = (prefill.captured(), decode.captured());
    assert_eq!((p.len(), d.len()), (1, 1));
    assert_eq!(d[0].kv_transfer_params.as_ref(), Some(&prefill_kv()));
    assert_eq!(
        worker_counter("vllm_router_pd_decode_errors_total", &decode.grpc_url),
        1.0
    );
    assert_eq!(
        worker_counter("vllm_router_pd_prefill_errors_total", &prefill.grpc_url),
        0.0
    );
}

#[tokio::test]
async fn grpc_pd_sends_each_leg_its_dp_rank() {
    let prefill = MockVllmRsServer::spawn_with_state(MockVllmRs {
        finish_kv_transfer_params: Some(prefill_kv()),
        ..Default::default()
    })
    .await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router_with_dp(&prefill, &decode, 2).await;

    for _ in 0..4 {
        let response = router.route_chat(None, &chat(false), None).await;
        assert!(response.status().is_success());
    }

    let ranks = |captured: &[common::mock_vllm_rs::CapturedGrpcGenerate]| {
        captured
            .iter()
            .map(|c| c.dp_rank.clone().expect("x-data-parallel-rank"))
            .collect::<BTreeSet<_>>()
    };
    let (p, d) = (prefill.captured(), decode.captured());
    assert_eq!(p.len(), 4);
    assert_eq!(d.len(), 4);
    let expected = BTreeSet::from(["0".to_string(), "1".to_string()]);
    assert_eq!(ranks(&p), expected, "prefill ranks");
    assert_eq!(ranks(&d), expected, "decode ranks");
    for (p, d) in p.iter().zip(&d) {
        assert_eq!(p.request_id, d.request_id);
        assert_eq!(d.kv_transfer_params.as_ref(), Some(&prefill_kv()));
    }
}

#[tokio::test]
async fn grpc_pd_rejects_non_chat_routes_before_dispatch() {
    let prefill = MockVllmRsServer::spawn().await;
    let decode = MockVllmRsServer::spawn().await;
    let router = pd_router(&prefill, &decode).await;

    let response = router
        .route_transparent(
            None,
            "/v1/completions",
            &axum::http::Method::POST,
            json!({"model": "test-model", "prompt": "hi", "max_tokens": 4}),
        )
        .await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("/v1/chat/completions"));
    assert!(prefill.captured().is_empty() && decode.captured().is_empty());
}
