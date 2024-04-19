// Copyright (c) Zefchain Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(target_arch = "wasm32", no_main)]

mod random;
mod state;
mod token;

use std::io::{Cursor, Seek, SeekFrom};

use async_graphql::{Context, EmptyMutation, EmptySubscription, Object, Request, Response, Schema};
use candle_core::{quantized::{ggml_file, gguf_file}, Device, Tensor, IndexOp};
use candle_transformers::{
    generation::LogitsProcessor,
    models::{
        llama2_c as llama, llama2_c, llama2_c::Llama, llama2_c_weights, quantized_llama as qllama,
        quantized_llama::ModelWeights,
    },
};
use linera_sdk::{base::WithServiceAbi, Service, ServiceRuntime, ViewStateStorage};
use log::{error, info, warn};
use thiserror::Error;
use tokenizers::Tokenizer;

use self::state::Llm;
use crate::token::TokenOutputStream;

pub struct LlmService {
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
        model_context.run_model(&prompt)
    }
}

enum Model {
    Llama {
        model: Llama,
        config: llama2_c::Config,
        cache: llama2_c::Cache,
    },
    Qllama(ModelWeights),
}

impl Model {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, candle_core::Error> {
        match self {
            Model::Llama {
                model: llama,
                config,
                cache,
            } => llama.forward(input, index_pos, cache),
            Model::Qllama(model) => model.forward(input, index_pos),
        }
    }
}

struct ModelContext {
    model: Vec<u8>,
    tokenizer: Vec<u8>,
}

impl Service for LlmService {
    type Error = ServiceError;
    type Storage = ViewStateStorage<Self>;
    type State = Llm;
    type Parameters = ();

    async fn new(_state: Self::State, runtime: ServiceRuntime<Self>) -> Result<Self, Self::Error> {
        Ok(LlmService { runtime })
    }

    async fn handle_query(&self, request: Request) -> Result<Response, Self::Error> {
        let query_string = &request.query;
        info!("query: {}", query_string);
        let raw_weights = self.runtime.fetch_url("http://localhost:10001/model.bin");
        info!("got weights: {}B", raw_weights.len());
        let tokenizer_bytes = self
            .runtime
            .fetch_url("http://localhost:10001/tokenizer.json");
        let model_context = ModelContext {
            model: raw_weights,
            tokenizer: tokenizer_bytes,
        };
        let schema = Schema::build(QueryRoot {}, EmptyMutation, EmptySubscription)
            .data(model_context)
            .finish();
        let response = schema.execute(request).await;
        Ok(response)
    }
}

const SYSTEM_MESSAGE: &str = "You are LineraBot, a helpful chatbot for a company called Linera.";

impl ModelContext {
    fn try_load_gguf(cursor: &mut Cursor<Vec<u8>>) -> Result<ModelWeights, ServiceError> {
        info!("trying to load model assuming gguf");
        let model_contents = gguf_file::Content::read(cursor)?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model_contents.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }

        info!(
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

    fn try_load_ggml(cursor: &mut Cursor<Vec<u8>>) -> Result<ModelWeights, ServiceError> {
        info!("trying to load model assuming ggml");
        let model_contents = ggml_file::Content::read(cursor, &Device::Cpu)?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model_contents.tensors.iter() {
            let elem_count = tensor.shape().elem_count();
            total_size_in_bytes +=
                elem_count * tensor.dtype().type_size() / tensor.dtype().block_size();
        }

        info!(
            "loaded {:?} tensors ({}B) ",
            model_contents.tensors.len(),
            total_size_in_bytes,
        );

        Ok(ModelWeights::from_ggml(model_contents, 1)?)
    }

    fn try_load_non_quantized(cursor: &mut Cursor<Vec<u8>>) -> Result<Model, ServiceError> {
        let config = llama2_c::Config::from_reader(cursor)?;
        println!("{config:?}");
        let weights =
            llama2_c_weights::TransformerWeights::from_reader(cursor, &config, &Device::Cpu)?;
        let vb = weights.var_builder(&config, &Device::Cpu)?;
        let cache = llama2_c::Cache::new(true, &config, vb.pp("rot"))?;
        let llama = Llama::load(vb, config.clone())?;
        Ok(Model::Llama {
            model: llama,
            config,
            cache,
        })
    }

    fn load_model(&self, model_weights: Vec<u8>) -> Result<Model, ServiceError> {
        let mut cursor = Cursor::new(model_weights);
        if let Ok(model) = Self::try_load_gguf(&mut cursor) {
            return Ok(Model::Qllama(model))
        }
        cursor.seek(SeekFrom::Start(0)).expect("seeking to 0");
        if let Ok(model) = Self::try_load_ggml(&mut cursor) {
            return Ok(Model::Qllama(model))
        }
        cursor.seek(SeekFrom::Start(0)).expect("seeking to 0");
        if let Ok(model) = Self::try_load_non_quantized(&mut cursor) {
            return Ok(model)
        }
        // might need a 'model not supported variant'
        Err(ServiceError::QueriesNotSupported)
    }

    // Copied mostly from https://github.com/huggingface/candle/blob/57267cd53612ede04090853680125b17956804f3/candle-examples/examples/quantized/main.rs
    fn run_model(&self, prompt_string: &str) -> Result<String, ServiceError> {
        let raw_weights = &self.model;
        let tokenizer_bytes = &self.tokenizer;
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
        let mut model = self.load_model(raw_weights.clone())?;

        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| ServiceError::Tokenizer(format!("{}", e)))?;
        info!("tokenizer: {:?}", tokenizer);
        println!("starting the inference loop");
        let mut logits_processor = LogitsProcessor::new(299792458, None, None);
        let mut index_pos = 0;

        let mut tokens = tokenizer
            .encode(prompt_string, true)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut tokenizer = TokenOutputStream::new(tokenizer);

        for index in 0.. {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;
            let logits = model.forward(&input, index_pos)?;
            let logits = logits.i((0, logits.dim(1)? - 1))?;
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                output.push_str(&t);
            }
        }
        if let Some(rest) = tokenizer.decode_rest().unwrap() {
            output.push_str(&rest);
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
    #[error("Candle error {0}")]
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
