//! OpenAI chat fields → `vllm-proto` `GenerateRequest`.
//!
//! This version always sets `prompt = TokenIds`. Proto also has text and
//! `media` fields; they are unused here. KV-transfer fields are set only by
//! the P/D router. Sampling-default merge still lives in worker `vllm-text`;
//! this only fills proto.

use crate::backend::pb;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, GenerationRequest, PromptInput, StringOrArray,
};

/// Passthrough keys the gRPC pipeline forwards to the worker. Anything else is
/// rejected: a field the adapter cannot express must not be dropped silently.
const SUPPORTED_OTHER: [&str; 7] = [
    "add_special_tokens",
    "cache_salt",
    "documents",
    "lora_name",
    "priority",
    "truncate_prompt_tokens",
    "watermarking",
];

/// Completions-only response option; never part of the chat shape.
const RETURN_TOKEN_IDS: &str = "return_token_ids";
/// Completions-only formatting option: logprob token text as `token_id:<id>`.
const RETURN_TOKENS_AS_TOKEN_IDS: &str = "return_tokens_as_token_ids";
/// Completions-only prompt-logprob option (`-1` means every prompt position).
const PROMPT_LOGPROBS: &str = "prompt_logprobs";
/// Completions-only vLLM extension: report logprobs for these token ids.
const LOGPROB_TOKEN_IDS: &str = "logprob_token_ids";

// Consumed by routing, never sent as generation options.
const ROUTING_ONLY: [&str; 4] = ["vllm_xargs", "agent_hint", "session_id", "user_id"];

fn ensure_supported_other<'a>(
    keys: impl Iterator<Item = &'a str>,
    endpoint: &str,
) -> Result<(), String> {
    let unsupported: Vec<&str> = keys
        .filter(|key| !SUPPORTED_OTHER.contains(key) && !ROUTING_ONLY.contains(key))
        .collect();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unsupported gRPC {endpoint} fields: {}",
            unsupported.join(", ")
        ))
    }
}

pub fn requires_worker_output_text(request: &ChatCompletionRequest) -> bool {
    !request.no_stop_trim
        && (request.stop.as_ref().is_some_and(|stop| match stop {
            StringOrArray::String(value) => !value.is_empty(),
            StringOrArray::Array(values) => !values.is_empty(),
        }) || request
            .stop_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty()))
}

/// Wire adapter: OpenAI chat fields → crates.io `vllm-proto` 0.2.
/// Client-visible options of a `/v1/completions` request that the chat shape
/// has no field for. `None` options mean the caller is on the chat path.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompletionOptions {
    /// Echo the prompt text before the generated text.
    pub echo: bool,
    /// `logprobs`: top-N token logprobs per position.
    pub logprobs: Option<u32>,
    /// `prompt_logprobs` from vLLM extensions; echo implies it when `logprobs`
    /// is set, matching the HTTP frontend.
    pub prompt_logprobs: Option<u32>,
    /// `echo` with `max_tokens == 0`: answer with the prompt only, so the
    /// engine still has to produce one token (the HTTP frontend does the same).
    pub echo_only: bool,
    /// Render logprob token text as `token_id:<id>` instead of decoded text.
    pub return_tokens_as_token_ids: bool,
    /// vLLM `logprob_token_ids`: report logprobs only for these token ids.
    pub logprob_token_ids: Option<Vec<u32>>,
    /// Prompt text prepended to the reply when `echo` is set. Filled per
    /// prompt by the frontend, which owns the tokenizer.
    pub echo_text: Option<String>,
}

pub fn chat_to_generate_request(
    request: &ChatCompletionRequest,
    token_ids: Vec<u32>,
    request_id: String,
) -> Result<pb::GenerateRequest, String> {
    chat_to_generate_request_with_options(request, token_ids, request_id, None)
}

/// Chat conversion with optional `/v1/completions` response semantics.
pub fn chat_to_generate_request_with_options(
    request: &ChatCompletionRequest,
    token_ids: Vec<u32>,
    request_id: String,
    completion: Option<&CompletionOptions>,
) -> Result<pb::GenerateRequest, String> {
    if completion.is_none() && request.n.unwrap_or(1) > 1 {
        return Err("n > 1 is not supported by vLLM 0.29 GenerateStream; use n=1".to_string());
    }
    if completion.is_none() && (request.logprobs || request.top_logprobs.is_some()) {
        return Err(
            "chat logprobs are not yet supported by the router gRPC response adapter".to_string(),
        );
    }
    if request.functions.is_some() || request.function_call.is_some() {
        return Err(
            "deprecated functions/function_call are not supported on gRPC; use tools/tool_choice"
                .to_string(),
        );
    }
    if request.lora_path.is_some() {
        return Err(
            "lora_path cannot be mapped to the gRPC loaded LoRA name; use lora_name".to_string(),
        );
    }
    if request.return_hidden_states || (completion.is_none() && request.echo == Some(true)) {
        return Err(
            "return_hidden_states and echo are not supported by the gRPC response adapter"
                .to_string(),
        );
    }
    ensure_supported_other(request.other.keys().map(String::as_str), "chat")?;
    other_bool(request, "add_special_tokens")?;
    // Validate types before using the same scope as routing.
    other_string(request, "lora_name")?;
    other_string(request, "cache_salt")?;
    let (lora_name, cache_salt) = request.cache_identity();
    let max_new_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(0);
    // `echo` with `max_tokens: 0` still needs one engine token so the stream
    // can finish; the reply drops it and returns the prompt text only.
    let max_new_tokens = if completion.is_some_and(|options| options.echo_only) {
        1
    } else {
        max_new_tokens
    };
    let prompt_logprobs = completion.and_then(completion_prompt_logprobs);

    let (stop_strings, stop_token_ids) = split_stops(request)?;
    let logit_bias = request
        .logit_bias
        .as_ref()
        .map(|bias| {
            bias.iter()
                .map(|(token, value)| {
                    token
                        .parse::<u32>()
                        .map(|id| (id, *value))
                        .map_err(|_| format!("logit_bias key is not a token id: {token}"))
                })
                .collect::<Result<_, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let mut decoding = pb::DecodingParameters {
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: request.repetition_penalty.unwrap_or(0.0),
        logit_bias,
        ..Default::default()
    };
    decoding.structured_output = structured_output(request)?;

    if let Some(top_k) = request.top_k {
        if top_k == -1 {
            return Err(
                "top_k=-1 is not supported by the vLLM gRPC protocol; omit top_k or use a positive value"
                    .to_string(),
            );
        }
        if top_k < 0 {
            return Err("top_k must be positive for the vLLM gRPC protocol".to_string());
        }
    }

    Ok(pb::GenerateRequest {
        request_id,
        model: request.model.clone().unwrap_or_default(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: token_ids,
        })),
        // EngineFrontend resolves model generation-config inheritance before
        // this adapter. The 1.0 fallback covers direct/test callers and matches
        // OpenAI when the model has no configured default; protobuf omission
        // cannot be used because vLLM gRPC rewrites it to greedy 0.0.
        temperature: Some(request.temperature.unwrap_or(1.0)),
        sampling: Some(pb::RandomSampling {
            num_sequences: request.n.unwrap_or(0),
            top_k: request.top_k.unwrap_or(0) as u32,
            top_p: request.top_p.unwrap_or(0.0),
            min_p: request.min_p.unwrap_or(0.0),
            seed: request.seed,
        }),
        decoding: Some(decoding),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens,
            min_new_tokens: request.min_tokens.unwrap_or(0),
            stop_token_ids,
            stop_strings,
            include_stop_strings: request.no_stop_trim,
            ignore_eos: request.ignore_eos,
        }),
        response: Some(pb::ResponseOptions {
            // Worker text carries the exact character-level trim boundary for
            // hidden stop strings/tokens; token IDs alone cannot represent it.
            output_text: Some(requires_worker_output_text(request)),
            output_token_ids: true,
            skip_special_tokens: Some(request.skip_special_tokens),
            output_logprobs: completion.is_some_and(|options| options.logprobs.is_some()),
            output_candidates: completion.and_then(|options| {
                options
                    .logprob_token_ids
                    .as_ref()
                    .map(|ids| pb::CandidateTokens {
                        select: Some(pb::candidate_tokens::Select::TokenIds(pb::TokenIds {
                            ids: ids.clone(),
                        })),
                    })
                    .or_else(|| options.logprobs.and_then(top_n_candidates))
            }),
            prompt_logprobs: prompt_logprobs.is_some(),
            prompt_candidates: prompt_logprobs.and_then(top_n_candidates),
            ..Default::default()
        }),
        kv: Some(pb::KvCacheParameters {
            cache_salt: cache_salt.unwrap_or_default().to_string(),
            ..Default::default()
        }),
        truncate_prompt_tokens: other_u32(request, "truncate_prompt_tokens")?.unwrap_or(0),
        priority: other_i32(request, "priority")?.unwrap_or(0),
        session_id: request
            .session_params
            .as_ref()
            .and_then(|params| params.get("session_id"))
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        lora_name: lora_name.unwrap_or_default().to_string(),
        watermarking: other_bool(request, "watermarking")?,
        ..Default::default()
    })
}

/// Prompt logprobs the completion frontend asks the engine for: an explicit
/// `prompt_logprobs` wins, otherwise `echo` + `logprobs` implies it, matching
/// the HTTP frontend's `prompt_logprobs = logprobs` default.
fn completion_prompt_logprobs(options: &CompletionOptions) -> Option<u32> {
    options
        .prompt_logprobs
        .or(options.logprobs.filter(|_| options.echo))
}

fn top_n_candidates(top_n: u32) -> Option<pb::CandidateTokens> {
    (top_n > 0).then_some(pb::CandidateTokens {
        select: Some(pb::candidate_tokens::Select::TopN(top_n)),
    })
}

fn structured_output(
    request: &ChatCompletionRequest,
) -> Result<Option<pb::decoding_parameters::StructuredOutput>, String> {
    use pb::decoding_parameters::StructuredOutput;

    let mut selected = Vec::new();
    if let Some(regex) = &request.regex {
        selected.push(StructuredOutput::Regex(regex.clone()));
    }
    if let Some(ebnf) = &request.ebnf {
        selected.push(StructuredOutput::Grammar(ebnf.clone()));
    }
    if let Some(format) = &request.response_format {
        match format {
            crate::protocols::spec::ResponseFormat::Text => {}
            crate::protocols::spec::ResponseFormat::JsonObject => {
                selected.push(StructuredOutput::JsonObject(true));
            }
            crate::protocols::spec::ResponseFormat::JsonSchema { json_schema } => {
                selected.push(StructuredOutput::Json(json_schema.schema.to_string()));
            }
        }
    }
    if let Some(params) = &request.structured_outputs {
        if let Some(json) = &params.json {
            selected.push(StructuredOutput::Json(json.to_string()));
        }
        if let Some(regex) = &params.regex {
            selected.push(StructuredOutput::Regex(regex.clone()));
        }
        if let Some(choices) = &params.choice {
            selected.push(StructuredOutput::Choice(
                pb::decoding_parameters::StringChoices {
                    choices: choices.clone(),
                },
            ));
        }
        if let Some(grammar) = &params.grammar {
            selected.push(StructuredOutput::Grammar(grammar.clone()));
        }
        if params.json_object == Some(true) {
            selected.push(StructuredOutput::JsonObject(true));
        }
        if let Some(tag) = &params.structural_tag {
            selected.push(StructuredOutput::StructuralTag(tag.clone()));
        }
    }
    match selected.len() {
        0 => Ok(None),
        1 => Ok(selected.pop()),
        _ => Err("only one structured output constraint may be specified".to_string()),
    }
}

fn split_stops(request: &ChatCompletionRequest) -> Result<(Vec<String>, Vec<u32>), String> {
    let stop_strings = match &request.stop {
        Some(StringOrArray::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(StringOrArray::Array(items)) => items.clone(),
        _ => Vec::new(),
    };
    let stop_token_ids = request
        .stop_token_ids
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|id| u32::try_from(*id).map_err(|_| format!("stop_token_ids cannot contain {id}")))
        .collect::<Result<_, _>>()?;
    Ok((stop_strings, stop_token_ids))
}

fn other_string<'a>(
    request: &'a ChatCompletionRequest,
    key: &str,
) -> Result<Option<&'a str>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{key} must be a string"))
        })
        .transpose()
}

fn other_bool(request: &ChatCompletionRequest, key: &str) -> Result<Option<bool>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| format!("{key} must be a boolean"))
        })
        .transpose()
}

fn other_u32(request: &ChatCompletionRequest, key: &str) -> Result<Option<u32>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("{key} must be a non-negative 32-bit integer"))
        })
        .transpose()
}

fn other_i32(request: &ChatCompletionRequest, key: &str) -> Result<Option<i32>, String> {
    request
        .other
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| format!("{key} must be a 32-bit integer"))
        })
        .transpose()
}

/// Turn a chat request into the prefill leg of a P/D pair: one sampled token,
/// with the connector's `kv_transfer_params`.
pub fn limit_to_prefill(proto: &mut pb::GenerateRequest) {
    let stopping = proto.stopping.get_or_insert_with(Default::default);
    stopping.max_new_tokens = 1;
    stopping.min_new_tokens = 1;
}

pub fn set_kv_transfer_params(
    proto: &mut pb::GenerateRequest,
    params: &serde_json::Value,
) -> Result<(), String> {
    proto
        .kv
        .get_or_insert_with(Default::default)
        .kv_transfer_params = Some(json_to_proto_struct(params)?);
    Ok(())
}

/// protobuf `Struct` numbers are `f64`; integers above 2^53 lose precision.
pub fn json_to_proto_struct(value: &serde_json::Value) -> Result<prost_types::Struct, String> {
    match json_to_proto_value(value).kind {
        Some(prost_types::value::Kind::StructValue(fields)) => Ok(fields),
        _ => Err("kv_transfer_params must be a JSON object".to_string()),
    }
}

fn json_to_proto_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    use serde_json::Value;

    let kind = match value {
        Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        Value::Bool(value) => Kind::BoolValue(*value),
        Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        Value::String(value) => Kind::StringValue(value.clone()),
        Value::Array(items) => Kind::ListValue(prost_types::ListValue {
            values: items.iter().map(json_to_proto_value).collect(),
        }),
        Value::Object(map) => Kind::StructValue(prost_types::Struct {
            fields: map
                .iter()
                .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

/// Whole numbers come back as JSON integers, so block ids and ports keep the
/// shape the HTTP connector JSON uses.
pub fn proto_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), proto_value_to_json(value)))
            .collect(),
    )
}

fn proto_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    use serde_json::Value;

    const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;
    match &value.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(value)) => Value::Bool(*value),
        Some(Kind::NumberValue(value)) => {
            if value.fract() == 0.0 && value.abs() <= MAX_EXACT_INTEGER {
                Value::from(*value as i64)
            } else {
                serde_json::Number::from_f64(*value).map_or(Value::Null, Value::Number)
            }
        }
        Some(Kind::StringValue(value)) => Value::String(value.clone()),
        Some(Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(proto_value_to_json).collect())
        }
        Some(Kind::StructValue(fields)) => proto_struct_to_json(fields),
    }
}

/// One prompt sequence lowered from `/v1/completions`, in client order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionPrompt {
    Ids(Vec<u32>),
    Text(String),
}

impl CompletionPrompt {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Ids(_) => None,
        }
    }
}

/// A `/v1/completions` request lowered onto the gRPC pipeline.
#[derive(Debug)]
pub struct PreparedCompletion {
    /// Chat-shaped carrier of every sampling/output field the pipeline reads.
    pub chat: ChatCompletionRequest,
    /// One entry per prompt sequence, in client order. Text prompts are
    /// tokenized by the frontend, which owns the model tokenizer.
    pub prompts: Vec<CompletionPrompt>,
    /// Echo text per prompt: the client's string, or `None` for ids until the
    /// frontend decodes them with a tokenizer.
    pub echo_texts: Vec<Option<String>>,
    pub emit_token_ids: bool,
    pub options: CompletionOptions,
}

impl PreparedCompletion {
    /// Token ids of the only prompt, when the client sent ids.
    pub fn single_prompt_ids(&self) -> Option<&[u32]> {
        match self.prompts.as_slice() {
            [CompletionPrompt::Ids(ids)] => Some(ids),
            _ => None,
        }
    }
}

/// Completions → chat shape, prompts, and response options for a gRPC pool.
///
/// Every OpenAI prompt form is accepted here. Text stays text until the
/// frontend tokenizes it with the model tokenizer; client ids pass through
/// untouched. Fields the gRPC wire cannot carry are rejected instead of being
/// dropped, because dropping them would answer a different request.
pub fn prepare_grpc_completion(request: &CompletionRequest) -> Result<PreparedCompletion, String> {
    if request.suffix.is_some() {
        return Err(
            "suffix requires FIM rendering on the serving engine; the gRPC /v1/completions path cannot express it"
                .to_string(),
        );
    }
    if request
        .best_of
        .is_some_and(|best_of| best_of > request.n.unwrap_or(1))
    {
        return Err("best_of is not supported by the vLLM v1 completions API; use n".to_string());
    }
    if request.return_hidden_states {
        return Err(
            "return_hidden_states is not supported by the gRPC response adapter".to_string(),
        );
    }
    if request.other.contains_key("documents") {
        return Err("documents is not supported by the gRPC /v1/completions path".to_string());
    }

    let (prompts, echo_texts) = lower_completion_prompts(&request.prompt)?;
    // Router-side only: the adapter asks the worker for ids unconditionally, so
    // this gates what the reply writes back rather than the wire request.
    let emit_token_ids = match request.other.get(RETURN_TOKEN_IDS) {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| "return_token_ids must be a boolean".to_string())?,
    };
    let return_tokens_as_token_ids = match request.other.get(RETURN_TOKENS_AS_TOKEN_IDS) {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| "return_tokens_as_token_ids must be a boolean".to_string())?,
    };
    let logprob_token_ids = match request.other.get(LOGPROB_TOKEN_IDS) {
        None => None,
        Some(value) => {
            let ids = value
                .as_array()
                .ok_or_else(|| "logprob_token_ids must be an array of token ids".to_string())?
                .iter()
                .map(|id| {
                    id.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| {
                            "logprob_token_ids must be non-negative 32-bit integers".to_string()
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if request.logprobs.is_none() {
                return Err("logprob_token_ids requires logprobs".to_string());
            }
            Some(ids)
        }
    };
    // `-1` asks vLLM for every prompt position; the gRPC bool has no count to
    // carry, so it maps to "on, no candidate tokens".
    let prompt_logprobs = match request.other.get(PROMPT_LOGPROBS) {
        None => None,
        Some(value) => Some(
            value
                .as_i64()
                .filter(|value| *value == -1 || *value >= 0)
                .and_then(|value| u32::try_from(value.max(0)).ok())
                .ok_or_else(|| {
                    "prompt_logprobs must be a non-negative integer or -1".to_string()
                })?,
        ),
    };
    if let Some(top_n) = request.logprobs {
        if top_n > 20 {
            return Err("logprobs must be between 0 and 20".to_string());
        }
    }
    ensure_supported_other(
        request
            .other
            .keys()
            .map(String::as_str)
            .filter(|key| !COMPLETION_ONLY_OTHER.contains(key)),
        "/v1/completions",
    )?;
    let other = request
        .other
        .iter()
        .filter(|(key, _)| {
            !COMPLETION_ONLY_OTHER.contains(&key.as_str()) && !ROUTING_ONLY.contains(&key.as_str())
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let structured_outputs = request
        .json_schema
        .as_ref()
        .map(|schema| {
            serde_json::from_str::<serde_json::Value>(schema)
                .map(|json| crate::protocols::spec::StructuredOutputsParams {
                    json: Some(json),
                    ..Default::default()
                })
                .map_err(|error| format!("json_schema must be valid JSON: {error}"))
        })
        .transpose()?;
    let chat = ChatCompletionRequest {
        model: request.model.clone(),
        messages: Vec::new(),
        temperature: request.temperature,
        top_p: request.top_p,
        n: request.n,
        stream: request.stream,
        stream_options: request.stream_options.clone(),
        stop: request.stop.clone(),
        max_tokens: request.max_tokens,
        max_completion_tokens: None,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        logit_bias: request.logit_bias.clone(),
        user: request.user.clone(),
        seed: request.seed,
        logprobs: false,
        top_logprobs: None,
        response_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        functions: None,
        function_call: None,
        top_k: request.top_k,
        min_p: request.min_p,
        min_tokens: request.min_tokens,
        repetition_penalty: request.repetition_penalty,
        regex: request.regex.clone(),
        ebnf: request.ebnf.clone(),
        stop_token_ids: request.stop_token_ids.clone(),
        no_stop_trim: request.no_stop_trim,
        ignore_eos: request.ignore_eos,
        add_generation_prompt: None,
        continue_final_message: false,
        skip_special_tokens: request.skip_special_tokens,
        lora_path: request.lora_path.clone(),
        session_params: request.session_params.clone(),
        chat_template_kwargs: None,
        return_hidden_states: false,
        echo: None,
        reasoning_effort: None,
        structured_outputs,
        other,
        ..Default::default()
    };
    Ok(PreparedCompletion {
        chat,
        prompts,
        echo_texts,
        emit_token_ids,
        options: CompletionOptions {
            echo: request.echo,
            logprobs: request.logprobs,
            prompt_logprobs,
            echo_only: request.max_tokens == Some(0),
            return_tokens_as_token_ids,
            logprob_token_ids,
            echo_text: None,
        },
    })
}

/// Lower every accepted prompt form into one entry per generated sequence.
fn lower_completion_prompts(
    prompt: &PromptInput,
) -> Result<(Vec<CompletionPrompt>, Vec<Option<String>>), String> {
    match prompt {
        PromptInput::String(text) => {
            if text.is_empty() {
                return Err("completion prompt must not be empty".to_string());
            }
            Ok((
                vec![CompletionPrompt::Text(text.clone())],
                vec![Some(text.clone())],
            ))
        }
        PromptInput::StringArray(texts) => {
            if texts.is_empty() {
                return Err("completion prompt list must not be empty".to_string());
            }
            if texts.iter().any(|text| text.is_empty()) {
                return Err("completion prompts must not be empty strings".to_string());
            }
            Ok((
                texts.iter().cloned().map(CompletionPrompt::Text).collect(),
                texts.iter().cloned().map(Some).collect(),
            ))
        }
        PromptInput::IntArray(ids) => Ok((
            vec![CompletionPrompt::Ids(completion_ids(ids)?)],
            vec![None],
        )),
        PromptInput::IntBatch(batch) => {
            if batch.is_empty() {
                return Err("completion prompt list must not be empty".to_string());
            }
            let ids = batch
                .iter()
                .map(|ids| completion_ids(ids))
                .collect::<Result<Vec<_>, _>>()?;
            let echo_texts = vec![None; ids.len()];
            Ok((
                ids.into_iter().map(CompletionPrompt::Ids).collect(),
                echo_texts,
            ))
        }
    }
}

fn completion_ids(ids: &[i32]) -> Result<Vec<u32>, String> {
    if ids.is_empty() {
        return Err("completion prompt token_ids must not be empty".to_string());
    }
    ids.iter()
        .map(|id| {
            u32::try_from(*id)
                .map_err(|_| format!("prompt token ids must be non-negative 32-bit integers: {id}"))
        })
        .collect()
}

/// Completion-only extension keys: consumed by the frontend, never forwarded.
const COMPLETION_ONLY_OTHER: [&str; 4] = [
    RETURN_TOKEN_IDS,
    RETURN_TOKENS_AS_TOKEN_IDS,
    PROMPT_LOGPROBS,
    LOGPROB_TOKEN_IDS,
];

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::backend::pb;

    fn chat(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn completion_routing_metadata_preserves_worker_cache_scope() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "prompt": [1, 2], "lora_name": "adapter", "cache_salt": "tenant",
            "session_id": "s", "user_id": "u", "agent_hint": {"agent_id": "a"},
            "vllm_xargs": {"agentic_context": {"session_id": "s"}},
            "session_params": {"session_id": "typed-session"}, "return_token_ids": true
        }))
        .unwrap();
        let prepared = prepare_grpc_completion(&request).unwrap();
        assert_eq!(prepared.single_prompt_ids(), Some(&[1, 2][..]));
        assert!(prepared.emit_token_ids);
        assert_eq!(prepared.chat.cache_identity(), request.cache_identity());
        assert_eq!(prepared.chat.session_params, request.session_params);
        for key in ROUTING_ONLY {
            assert!(!prepared.chat.other.contains_key(key));
        }
        let proto = chat_to_generate_request_with_options(
            &prepared.chat,
            prepared.single_prompt_ids().unwrap().to_vec(),
            "r".into(),
            Some(&prepared.options),
        )
        .unwrap();
        assert_eq!(proto.lora_name, "adapter");
        assert_eq!(proto.kv.unwrap().cache_salt, "tenant");
    }

    #[test]
    fn token_ids_only_no_text_prompt() {
        // This-version default: convert always emits TokenIds. Not a forever
        // ban on a text prompt field if that is added later.
        let req = chat(serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0.2
        }));
        let proto = chat_to_generate_request(&req, vec![11, 22, 33], "req-1".into()).unwrap();
        match proto.prompt.unwrap() {
            pb::generate_request::Prompt::TokenIds(ids) => {
                assert_eq!(ids.ids, vec![11, 22, 33]);
            }
            other => panic!("expected token_ids, got {other:?}"),
        }
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 32);
        assert_eq!(proto.temperature, Some(0.2));
        assert_eq!(proto.model, "qwen");
        assert_eq!(proto.response.unwrap().output_text, Some(false));
    }

    #[test]
    fn prefers_max_completion_tokens() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "max_completion_tokens": 64
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 64);
    }

    #[test]
    fn preserves_openai_defaults_when_omitted() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.temperature, Some(1.0));
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 0);
    }

    #[test]
    fn missing_model_stays_empty_on_grpc_wire() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.model, "");
    }

    #[test]
    fn rejects_top_k_minus_one_for_grpc() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": -1
        }));
        let err = chat_to_generate_request(&req, vec![1], "r".into()).unwrap_err();
        assert!(err.contains("top_k=-1"));

        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": -2
        }));
        let err = chat_to_generate_request(&req, vec![1], "r".into()).unwrap_err();
        assert!(err.contains("top_k must be positive"));
    }

    #[test]
    fn rejects_unadapted_multi_choice_and_logprobs() {
        let n = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2
        }));
        assert!(chat_to_generate_request(&n, vec![1], "r".into())
            .unwrap_err()
            .contains("n > 1"));

        let logprobs = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true
        }));
        assert!(chat_to_generate_request(&logprobs, vec![1], "r".into())
            .unwrap_err()
            .contains("logprobs"));
    }

    #[test]
    fn lowers_json_schema_and_logit_bias() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": {"42": -10.0},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            }
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        let decoding = proto.decoding.unwrap();
        assert_eq!(decoding.logit_bias.get(&42), Some(&-10.0));
        assert!(matches!(
            decoding.structured_output,
            Some(pb::decoding_parameters::StructuredOutput::Json(_))
        ));
    }

    #[test]
    fn requests_worker_text_for_hidden_stop_boundary() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END"]
        }));
        let proto = chat_to_generate_request(&req, vec![1], "r".into()).unwrap();
        assert_eq!(proto.response.unwrap().output_text, Some(true));

        let visible = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["END"],
            "no_stop_trim": true
        }));
        let proto = chat_to_generate_request(&visible, vec![1], "r".into()).unwrap();
        assert_eq!(proto.response.unwrap().output_text, Some(false));
    }

    #[test]
    fn kv_transfer_params_round_trip_through_proto_struct() {
        let params = serde_json::json!({
            "do_remote_prefill": true,
            "do_remote_decode": false,
            "remote_block_ids": [[7, 8], [9]],
            "remote_engine_id": "P",
            "remote_port": 5600,
            "remote_host": null,
            "ratio": 0.5,
        });
        let proto = json_to_proto_struct(&params).unwrap();
        assert_eq!(proto_struct_to_json(&proto), params);
        assert!(json_to_proto_struct(&serde_json::json!([1])).is_err());
    }

    #[test]
    fn prefill_leg_samples_one_token_and_carries_kv_params() {
        let req = chat(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "cache_salt": "salt"
        }));
        let mut proto = chat_to_generate_request(&req, vec![1, 2], "r".into()).unwrap();
        limit_to_prefill(&mut proto);
        set_kv_transfer_params(&mut proto, &serde_json::json!({"do_remote_decode": true})).unwrap();
        let stopping = proto.stopping.unwrap();
        assert_eq!((stopping.max_new_tokens, stopping.min_new_tokens), (1, 1));
        let kv = proto.kv.unwrap();
        assert_eq!(kv.cache_salt, "salt");
        assert_eq!(
            proto_struct_to_json(kv.kv_transfer_params.as_ref().unwrap()),
            serde_json::json!({"do_remote_decode": true})
        );
    }

    fn completion(json: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn lowers_completion_onto_chat_shape() {
        let req = completion(serde_json::json!({
            "model": "qwen",
            "prompt": [1, 2, 3],
            "max_tokens": 16,
            "temperature": 0.3,
            "top_k": 10,
            "stop": ["END"],
            "return_token_ids": true
        }));
        let prepared = prepare_grpc_completion(&req).unwrap();
        assert_eq!(prepared.single_prompt_ids(), Some(&[1, 2, 3][..]));
        assert!(prepared.emit_token_ids);
        assert_eq!(prepared.chat.max_tokens, Some(16));
        assert_eq!(prepared.chat.top_k, Some(10));
        assert!(prepared.chat.messages.is_empty());
        assert!(prepared.chat.stop.is_some());
    }

    #[test]
    fn accepts_every_prompt_form() {
        let single_text =
            prepare_grpc_completion(&completion(serde_json::json!({"prompt": "hi"}))).unwrap();
        assert_eq!(
            single_text.prompts,
            vec![CompletionPrompt::Text("hi".into())]
        );
        assert_eq!(single_text.echo_texts, vec![Some("hi".to_string())]);

        let text_batch =
            prepare_grpc_completion(&completion(serde_json::json!({"prompt": ["a", "b"]})))
                .unwrap();
        assert_eq!(text_batch.prompts.len(), 2);
        assert_eq!(text_batch.echo_texts.len(), 2);

        let id_batch =
            prepare_grpc_completion(&completion(serde_json::json!({"prompt": [[1, 2], [3]]})))
                .unwrap();
        assert_eq!(
            id_batch.prompts,
            vec![
                CompletionPrompt::Ids(vec![1, 2]),
                CompletionPrompt::Ids(vec![3])
            ]
        );
    }

    #[test]
    fn multi_choice_and_logprobs_are_lowered_not_rejected() {
        let req = completion(serde_json::json!({
            "prompt": [1], "n": 3, "logprobs": 2, "echo": true
        }));
        let prepared = prepare_grpc_completion(&req).unwrap();
        assert_eq!(prepared.chat.n, Some(3));
        assert_eq!(prepared.options.logprobs, Some(2));
        assert!(prepared.options.echo);
        // `echo` + `logprobs` implies prompt logprobs at the proto layer; the
        // option itself only carries an explicit `prompt_logprobs`.
        assert!(prepared.options.prompt_logprobs.is_none());

        let proto = chat_to_generate_request_with_options(
            &prepared.chat,
            vec![1],
            "r".into(),
            Some(&prepared.options),
        )
        .unwrap();
        assert_eq!(proto.sampling.unwrap().num_sequences, 3);
        let response = proto.response.unwrap();
        assert!(response.output_logprobs);
        assert!(response.prompt_logprobs);
        assert!(matches!(
            response.output_candidates.unwrap().select,
            Some(pb::candidate_tokens::Select::TopN(2))
        ));
    }

    #[test]
    fn logprob_token_ids_selects_candidate_tokens() {
        let req = completion(serde_json::json!({
            "prompt": [1], "logprobs": 3, "logprob_token_ids": [7, 8]
        }));
        let prepared = prepare_grpc_completion(&req).unwrap();
        let proto = chat_to_generate_request_with_options(
            &prepared.chat,
            vec![1],
            "r".into(),
            Some(&prepared.options),
        )
        .unwrap();
        let response = proto.response.unwrap();
        assert!(matches!(
            response.output_candidates.unwrap().select,
            Some(pb::candidate_tokens::Select::TokenIds(ids)) if ids.ids == vec![7, 8]
        ));

        let missing = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1], "logprob_token_ids": [7]
        })))
        .unwrap_err();
        assert!(missing.contains("requires logprobs"), "{missing}");
    }

    #[test]
    fn echo_with_zero_max_tokens_still_asks_for_one_token() {
        let req = completion(serde_json::json!({
            "prompt": [1], "echo": true, "max_tokens": 0
        }));
        let prepared = prepare_grpc_completion(&req).unwrap();
        assert!(prepared.options.echo_only);
        let proto = chat_to_generate_request_with_options(
            &prepared.chat,
            vec![1],
            "r".into(),
            Some(&prepared.options),
        )
        .unwrap();
        assert_eq!(proto.stopping.unwrap().max_new_tokens, 1);
    }

    #[test]
    fn json_schema_becomes_structured_output() {
        let req = completion(serde_json::json!({
            "prompt": [1],
            "json_schema": "{\"type\":\"object\"}"
        }));
        let prepared = prepare_grpc_completion(&req).unwrap();
        let proto = chat_to_generate_request_with_options(
            &prepared.chat,
            vec![1],
            "r".into(),
            Some(&prepared.options),
        )
        .unwrap();
        assert!(proto.decoding.unwrap().structured_output.is_some());
    }

    #[test]
    fn token_ids_are_withheld_unless_requested() {
        let default =
            prepare_grpc_completion(&completion(serde_json::json!({"prompt": [1]}))).unwrap();
        assert!(!default.emit_token_ids);

        let explicit_false = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1],
            "return_token_ids": false
        })))
        .unwrap();
        assert!(!explicit_false.emit_token_ids);

        let wrong_type = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1],
            "return_token_ids": "yes"
        })))
        .unwrap_err();
        assert!(wrong_type.contains("boolean"), "{wrong_type}");
    }

    #[test]
    fn rejects_fields_the_grpc_adapter_cannot_express() {
        for (body, field) in [
            (
                serde_json::json!({"prompt": [1], "suffix": "tail"}),
                "suffix",
            ),
            (serde_json::json!({"prompt": [1], "best_of": 3}), "best_of"),
            (
                serde_json::json!({"prompt": [1], "return_hidden_states": true}),
                "return_hidden_states",
            ),
            (
                serde_json::json!({"prompt": [1], "documents": [{"content": "x"}]}),
                "documents",
            ),
        ] {
            let err = prepare_grpc_completion(&completion(body)).unwrap_err();
            assert!(err.contains(field), "{err}");
        }

        // `best_of <= n` is a no-op, not a rejection.
        let noop = prepare_grpc_completion(&completion(
            serde_json::json!({"prompt": [1], "best_of": 1, "n": 2}),
        ))
        .unwrap();
        assert_eq!(noop.chat.n, Some(2));
    }

    #[test]
    fn rejects_unknown_passthrough_fields_instead_of_dropping_them() {
        let err = prepare_grpc_completion(&completion(serde_json::json!({
            "prompt": [1],
            "guided_decoding_backend": "xgrammar"
        })))
        .unwrap_err();
        assert!(err.contains("guided_decoding_backend"), "{err}");

        let chat_err = chat_to_generate_request(
            &chat(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "guided_decoding_backend": "xgrammar"
            })),
            vec![1],
            "r".into(),
        )
        .unwrap_err();
        assert!(chat_err.contains("guided_decoding_backend"), "{chat_err}");
    }

    #[test]
    fn rejects_prompts_without_usable_token_ids() {
        let negative = prepare_grpc_completion(&completion(serde_json::json!({"prompt": [1, -2]})))
            .unwrap_err();
        assert!(negative.contains("non-negative"), "{negative}");

        let empty =
            prepare_grpc_completion(&completion(serde_json::json!({"prompt": []}))).unwrap_err();
        assert!(empty.contains("must not be empty"), "{empty}");
    }
}
