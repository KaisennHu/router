//! Fan-out for batched `/v1/completions` on `grpc://` pools.
//!
//! A gRPC `GenerateRequest` carries exactly one prompt, so a batched completion
//! becomes N independent router requests (each with its own gRPC P/D admission)
//! that are merged back into one OpenAI reply. Choices keep the HTTP frontend's
//! numbering: `index = prompt_index * n + choice_index`, which is what vLLM
//! emits as `index=len(choices)` across prompts and choices.

use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::protocols::spec::{CompletionRequest, PromptInput};

/// One single-prompt request per prompt in the client's batch.
pub fn split_prompt(prompt: &PromptInput) -> Vec<PromptInput> {
    match prompt {
        PromptInput::StringArray(items) => items.iter().cloned().map(PromptInput::String).collect(),
        PromptInput::IntBatch(items) => items.iter().cloned().map(PromptInput::IntArray).collect(),
        single => vec![single.clone()],
    }
}

/// Copy the batched request with exactly one prompt.
pub fn with_prompt(request: &CompletionRequest, prompt: PromptInput) -> CompletionRequest {
    let mut single = request.clone();
    single.prompt = prompt;
    single
}

/// Merge responses that each carry `choice_stride` choices.
///
/// `choice_stride` is `n` when one request answers each batched prompt, and 1
/// when `n` was fanned out into `n` single-choice requests. `prompt_repeat`
/// divides the summed prompt tokens back to one count per prompt.
pub async fn merge_with(
    responses: Vec<Response>,
    request: &CompletionRequest,
    choice_stride: u32,
    prompt_repeat: u32,
) -> Response {
    if responses.is_empty() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "no sub-responses to merge",
        )
            .into_response();
    }
    if request.stream {
        merge_streams(responses, request, choice_stride, prompt_repeat).await
    } else {
        merge_bodies(responses, prompt_repeat).await
    }
}

async fn merge_bodies(responses: Vec<Response>, prompt_repeat: u32) -> Response {
    let mut merged: Option<Value> = None;
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    for response in responses {
        if !response.status().is_success() {
            return response;
        }
        let bytes = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
            Ok(bytes) => bytes,
            Err(error) => return (StatusCode::BAD_GATEWAY, error.to_string()).into_response(),
        };
        let mut value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => return (StatusCode::BAD_GATEWAY, error.to_string()).into_response(),
        };
        if let Some(usage) = value.get("usage") {
            prompt_tokens += usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            completion_tokens += usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
        match merged.take() {
            Some(mut existing) => {
                let base = existing
                    .get("choices")
                    .and_then(Value::as_array)
                    .map(Vec::len)
                    .unwrap_or(0);
                if let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) {
                    for (offset, choice) in choices.iter_mut().enumerate() {
                        choice["index"] = json!(base + offset);
                    }
                    if let Some(target) = existing.get_mut("choices").and_then(Value::as_array_mut)
                    {
                        target.extend(choices.iter().cloned());
                    }
                }
                merged = Some(existing);
            }
            None => merged = Some(value),
        }
    }
    let Some(mut merged) = merged else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "no completion bodies to merge",
        )
            .into_response();
    };
    if prompt_repeat > 1 {
        // The same prompt was sent once per choice; count it once.
        prompt_tokens /= u64::from(prompt_repeat);
    }
    merged["usage"] = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    });
    axum::Json(merged).into_response()
}

async fn merge_streams(
    responses: Vec<Response>,
    request: &CompletionRequest,
    choice_stride: u32,
    prompt_repeat: u32,
) -> Response {
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let shared_id = Arc::new(Mutex::new(None::<String>));
    let shared_meta = Arc::new(Mutex::new(None::<(String, String, u64)>));
    let mut readers = Vec::with_capacity(responses.len());
    for (index, response) in responses.into_iter().enumerate() {
        if !response.status().is_success() {
            return response;
        }
        let out = tx.clone();
        let shared_id = shared_id.clone();
        let shared_meta = shared_meta.clone();
        let base = index as u64 * u64::from(choice_stride);
        readers.push(tokio::spawn(async move {
            forward_frames(response, out, shared_id, shared_meta, base).await
        }));
    }
    let coordinator = tx;
    tokio::spawn(async move {
        let mut prompt_tokens = 0u64;
        let mut completion_tokens = 0u64;
        let mut saw_usage = false;
        for reader in readers {
            if let Ok(Some(usage)) = reader.await {
                saw_usage = true;
                prompt_tokens += usage
                    .get("prompt_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                completion_tokens += usage
                    .get("completion_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
            }
        }
        if prompt_repeat > 1 {
            prompt_tokens /= u64::from(prompt_repeat);
        }
        if include_usage && saw_usage {
            let (id, model, created) = shared_meta
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| ("cmpl-merged".to_string(), String::new(), 0));
            let frame = format!(
                "data: {}\n\n",
                json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })
            );
            let _ = coordinator.send(Ok(Bytes::from(frame)));
        }
        let _ = coordinator.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
    });
    let body = Body::from_stream(UnboundedReceiverStream::new(rx));
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Forward one sub-response's SSE frames with remapped choice indices.
/// Returns the usage chunk it saw, if any.
async fn forward_frames(
    response: Response,
    out: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    shared_id: Arc<Mutex<Option<String>>>,
    shared_meta: Arc<Mutex<Option<(String, String, u64)>>>,
    base: u64,
) -> Option<Value> {
    let mut stream = response.into_body().into_data_stream();
    let mut pending: Vec<u8> = Vec::new();
    let mut usage: Option<Value> = None;
    'outer: while let Some(chunk) = stream.next().await {
        let Ok(bytes) = chunk else { break };
        pending.extend_from_slice(&bytes);
        while let Some(position) = find_frame_end(&pending) {
            let frame: Vec<u8> = pending.drain(..position + 2).collect();
            let payload = &frame[..position];
            let Some(payload) = payload.strip_prefix(b"data: ") else {
                continue;
            };
            if payload == b"[DONE]" {
                break 'outer;
            }
            let Ok(mut value) = serde_json::from_slice::<Value>(payload) else {
                let mut line = Vec::from(&b"data: "[..]);
                line.extend_from_slice(payload);
                line.extend_from_slice(b"\n\n");
                if out.send(Ok(Bytes::from(line))).is_err() {
                    break 'outer;
                }
                continue;
            };
            let has_choices = value
                .get("choices")
                .and_then(Value::as_array)
                .is_some_and(|choices| !choices.is_empty());
            if !has_choices {
                if let Some(chunk_usage) = value.get("usage").cloned() {
                    if chunk_usage.is_object() {
                        usage = Some(chunk_usage);
                        continue;
                    }
                }
            }
            if let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) {
                for choice in choices {
                    if let Some(index) = choice.get("index").and_then(Value::as_u64) {
                        choice["index"] = json!(base + index);
                    }
                }
            }
            let mut id_guard = shared_id.lock().unwrap();
            if let Some(id) = value.get("id").and_then(Value::as_str) {
                if id_guard.is_none() {
                    *id_guard = Some(id.to_string());
                } else if let Some(shared) = id_guard.as_ref() {
                    value["id"] = json!(shared);
                }
            }
            drop(id_guard);
            if let (Some(id), Some(model), Some(created)) = (
                value.get("id").and_then(Value::as_str),
                value.get("model").and_then(Value::as_str),
                value.get("created").and_then(Value::as_u64),
            ) {
                let mut meta = shared_meta.lock().unwrap();
                if meta.is_none() {
                    *meta = Some((id.to_string(), model.to_string(), created));
                }
            }
            let mut line = Vec::from(&b"data: "[..]);
            line.extend_from_slice(value.to_string().as_bytes());
            line.extend_from_slice(b"\n\n");
            if out.send(Ok(Bytes::from(line))).is_err() {
                break 'outer;
            }
        }
    }
    usage
}

fn find_frame_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(2).position(|window| window == b"\n\n")
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn body_response(body: Value) -> Response {
        (StatusCode::OK, axum::Json(body)).into_response()
    }

    fn completion_request() -> CompletionRequest {
        serde_json::from_value(json!({"prompt": ["a", "b"], "n": 1})).unwrap()
    }

    #[test]
    fn splits_every_batched_prompt_shape() {
        let text = split_prompt(&PromptInput::StringArray(vec!["a".into(), "b".into()]));
        assert_eq!(
            text,
            vec![
                PromptInput::String("a".into()),
                PromptInput::String("b".into())
            ]
        );

        let ids = split_prompt(&PromptInput::IntBatch(vec![vec![1, 2], vec![3]]));
        assert_eq!(
            ids,
            vec![
                PromptInput::IntArray(vec![1, 2]),
                PromptInput::IntArray(vec![3])
            ]
        );

        let single = split_prompt(&PromptInput::String("a".into()));
        assert_eq!(single.len(), 1);
    }

    #[tokio::test]
    async fn merges_bodies_with_sequential_indices_and_summed_usage() {
        let responses = vec![
            body_response(json!({
                "id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
                "choices": [{"text": "A", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            })),
            body_response(json!({
                "id": "cmpl-2", "object": "text_completion", "created": 1, "model": "m",
                "choices": [{"text": "B", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            })),
        ];
        let merged = merge_with(responses, &completion_request(), 1, 1).await;
        assert_eq!(merged.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(merged.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[1]["index"], 1);
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["completion_tokens"], 2);
        assert_eq!(json["usage"]["total_tokens"], 7);
    }

    #[tokio::test]
    async fn merge_with_fans_out_choices_and_counts_the_prompt_once() {
        let responses = vec![
            body_response(json!({
                "id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
                "choices": [{"text": "A", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })),
            body_response(json!({
                "id": "cmpl-2", "object": "text_completion", "created": 1, "model": "m",
                "choices": [{"text": "B", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })),
        ];
        let merged = merge_with(responses, &completion_request(), 1, 2).await;
        assert_eq!(merged.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(merged.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[1]["index"], 1);
        // `n` repeats the same prompt: prompt tokens count once, output tokens
        // accumulate across the sampled choices.
        assert_eq!(json["usage"]["prompt_tokens"], 3);
        assert_eq!(json["usage"]["completion_tokens"], 4);
        assert_eq!(json["usage"]["total_tokens"], 7);
    }

    #[tokio::test]
    async fn a_failed_sub_request_is_returned_instead_of_a_partial_merge() {
        let responses = vec![(StatusCode::SERVICE_UNAVAILABLE, "down").into_response()];
        let merged = merge_with(responses, &completion_request(), 1, 1).await;
        assert_eq!(merged.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
