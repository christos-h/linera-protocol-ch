#![cfg_attr(target_arch = "wasm32", no_main)]

mod random;
mod state;
mod token;

use std::{io::Cursor, sync::Arc};

use async_graphql::{Context, EmptyMutation, EmptySubscription, Object, Request, Response, Schema};
use async_trait::async_trait;
use candle_core::{
    quantized::{ggml_file, gguf_file},
    Device, Tensor,
};
use candle_transformers::{
    generation::LogitsProcessor,
    models::{quantized_llama as model, quantized_llama::ModelWeights},
};
use linera_sdk::{
    abi::ServiceAbi, base::WithServiceAbi, views::RegisterView, Service, ServiceRuntime,
    ViewStateStorage,
};
use log::error;
use thiserror::Error;
use tokenizers::Tokenizer;

use self::state::Llm;
use crate::token::TokenOutputStream;

pub struct LlmService {
    state: Llm,
    runtime: ServiceRuntime<Self>,
}

linera_sdk::service!(LlmService);

impl WithServiceAbi for LlmService {
    type Abi = llm::LlmAbi;
}

struct QueryRoot {}

#[Object]
impl QueryRoot {
    async fn prompt(&self, ctx: &Context<'_>, prompt: String) -> Result<String, ServiceError> {
        let model_context = ctx.data::<ModelContext>()?;
        Ok(model_context.run_model(&prompt)?)
    }
}

struct ModelContext {
    model: Vec<u8>,
    tokenizer: Vec<u8>,
    /// Group query attention
    gqa: usize,
}

impl Service for LlmService {
    type Error = ServiceError;
    type Storage = ViewStateStorage<Self>;
    type State = Llm;
    type Parameters = ();

    async fn new(state: Self::State, runtime: ServiceRuntime<Self>) -> Result<Self, Self::Error> {
        Ok(LlmService { state, runtime })
    }

    async fn handle_query(&self, request: Request) -> Result<Response, Self::Error> {
        let prompt_string = &request.query; // TODO is this correct?
        let raw_weights = self.runtime.fetch_url(
            "http://localhost:10001/model.bin",
        );
        error!("got weights: {}B", raw_weights.len());
        let tokenizer_bytes = self.runtime.fetch_url(
            "http://localhost:10001/tokenizer.json",
        );
        let model_context = ModelContext {
            model: raw_weights,
            tokenizer: tokenizer_bytes,
            gqa: 1,
        };
        let schema = Schema::build(QueryRoot {}, EmptyMutation, EmptySubscription)
            .data(model_context)
            .finish();
        error!("prompt: {}", prompt_string);
        let response = schema.execute(request).await;
        return Ok(response);
    }
}

const SYSTEM_MESSAGE: &'static str =
    "You are LineraBot, a helpful chatbot for a company called Linera.";

impl ModelContext {
    fn load_model(&self, model_weights: Vec<u8>, gqa: usize) -> Result<ModelWeights, ServiceError> {
        let cursor = &mut Cursor::new(model_weights);
        let model_contents = gguf_file::Content::read(cursor).unwrap();
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model_contents.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }

        error!(
            "loaded {:?} tensors ({}B) ",
            model_contents.tensor_infos.len(),
            total_size_in_bytes,
        );

        Ok(ModelWeights::from_gguf(
            model_contents,
            cursor,
            &Device::Cpu,
        )?)
    }

    // Copied mostly from https://github.com/huggingface/candle/blob/57267cd53612ede04090853680125b17956804f3/candle-examples/examples/quantized/main.rs
    fn run_model(&self, prompt_string: &str) -> Result<String, ServiceError> {
        let raw_weights = &self.model;
        let tokenizer_bytes = &self.tokenizer;
        let gqa = self.gqa;
        let prompt_string = format!(
            r#"
<|system|>
{SYSTEM_MESSAGE}</s>
<|user|>
{prompt_string}</s>
<|assistant|>
        "#
        );

        let mut output = String::new();
        let mut model = self.load_model(raw_weights.clone(), gqa)?;

        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| ServiceError::Tokenizer(format!("{}", e)))?;
        error!("tokenizer: {:?}", tokenizer);
        let mut token_output_stream = TokenOutputStream::new(tokenizer);
        let tokens = token_output_stream
            .tokenizer()
            .encode(prompt_string, true)
            .map_err(|e| ServiceError::Tokenizer(format!("{}", e)))?;

        let prompt_tokens = tokens.get_ids().to_vec();
        error!("prompt tokens: {:?}", prompt_tokens);
        let to_sample = 20usize.saturating_sub(1); // 1000 is the default value in candle example
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens = vec![];
        let seed = 299792458; // taken as the default value from the candle example.
        let mut logits_processor = LogitsProcessor::new(seed, None, None);

        let mut next_token = 0;
        for (pos, token) in prompt_tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }

        all_tokens.push(next_token);
        if let Some(t) = token_output_stream.next_token(next_token)? {
            output.push_str(&t);
        }

        let eos_token = "</s>";
        let eos_token = *token_output_stream
            .tokenizer()
            .get_vocab(true)
            .get(eos_token)
            .unwrap();
        let mut sampled = 0;
        let repeat_penatly = 1.1; // taken from candle example
        let repeat_last_n = 64; // taken from candle example
        for index in 0..to_sample {
            error!("index: {}", index);
            let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;

            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penatly,
                &all_tokens[start_at..],
            )?;

            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = token_output_stream.next_token(next_token)? {
                output.push_str(&t);
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = token_output_stream
            .decode_rest()
            .map_err(|e| ServiceError::Candle(e))?
        {
            output.push_str(&rest)
        }
        Ok(output)
    }
}

/// An error that can occur while querying the service.
#[derive(Debug, Error)]
pub enum ServiceError {
    /// Query not supported by the application.
    #[error("Queries not supported by application")]
    QueriesNotSupported,

    /// Invalid query argument; could not deserialize request.
    #[error("Invalid query argument; could not deserialize request")]
    InvalidQuery(#[from] serde_json::Error),
    // Add error variants here.
    /// Invalid query argument; could not deserialize request.
    #[error("Candle error")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error")]
    Tokenizer(String),

    #[error("GraphQL error")]
    GraphQL(String),
}

impl From<async_graphql::Error> for ServiceError {
    fn from(value: async_graphql::Error) -> Self {
        Self::GraphQL(value.message)
    }
}
