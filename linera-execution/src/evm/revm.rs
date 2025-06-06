// Copyright (c) Zefchain Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Code specific to the usage of the [Revm](https://bluealloy.github.io/revm/) runtime.

use core::ops::Range;
use std::{collections::BTreeSet, convert::TryFrom};

#[cfg(with_metrics)]
use linera_base::prometheus_util::MeasureLatency as _;
use linera_base::{
    crypto::CryptoHash,
    data_types::{Bytecode, Resources, SendMessageRequest, StreamUpdate},
    ensure,
    identifiers::{ApplicationId, ChainId, StreamName},
    vm::{EvmQuery, VmRuntime},
};
use revm::{primitives::Bytes, InspectCommitEvm, InspectEvm, Inspector};
use revm_context::{
    result::{ExecutionResult, Output, SuccessReason},
    BlockEnv, Cfg, ContextTr, Evm, Journal, LocalContextTr, TxEnv,
};
use revm_database::WrapDatabaseRef;
use revm_handler::{
    instructions::EthInstructions, EthPrecompiles, MainnetContext, PrecompileProvider,
};
use revm_interpreter::{
    CallInput, CallInputs, CallOutcome, CreateInputs, CreateOutcome, CreateScheme, Gas, InputsImpl,
    InstructionResult, InterpreterResult,
};
use revm_primitives::{address, hardfork::SpecId, Address, Log, TxKind};
use revm_state::EvmState;
use serde::{Deserialize, Serialize};

use crate::{
    evm::database::{DatabaseRuntime, StorageStats, EVM_SERVICE_GAS_LIMIT},
    BaseRuntime, ContractRuntime, ContractSyncRuntimeHandle, EvmExecutionError, EvmRuntime,
    ExecutionError, ServiceRuntime, ServiceSyncRuntimeHandle, UserContract, UserContractInstance,
    UserContractModule, UserService, UserServiceInstance, UserServiceModule,
};

/// This is the selector of the `execute_message` that should be called
/// only from a submitted message
const EXECUTE_MESSAGE_SELECTOR: &[u8] = &[173, 125, 234, 205];

/// This is the selector of the `instantiate` that should be called
/// only when creating a new instance of a shared contract
const INSTANTIATE_SELECTOR: &[u8] = &[156, 163, 60, 158];

fn forbid_execute_operation_origin(vec: &[u8]) -> Result<(), ExecutionError> {
    ensure!(
        vec != EXECUTE_MESSAGE_SELECTOR,
        ExecutionError::EvmError(EvmExecutionError::OperationCallExecuteMessage)
    );
    ensure!(
        vec != INSTANTIATE_SELECTOR,
        ExecutionError::EvmError(EvmExecutionError::OperationCallInstantiate)
    );
    Ok(())
}

fn ensure_message_length(actual_length: usize, min_length: usize) -> Result<(), ExecutionError> {
    ensure!(
        actual_length >= min_length,
        ExecutionError::EvmError(EvmExecutionError::OperationIsTooShort)
    );
    Ok(())
}

/// The selector when calling for `InterpreterResult`. This is a fictional
/// selector that does not correspond to a real function.
const INTERPRETER_RESULT_SELECTOR: &[u8] = &[1, 2, 3, 4];

#[cfg(test)]
mod tests {
    use revm_primitives::keccak256;

    use crate::evm::revm::{EXECUTE_MESSAGE_SELECTOR, INSTANTIATE_SELECTOR};

    // The function keccak256 is not const so we cannot build the execute_message
    // selector directly.
    #[test]
    fn check_execute_message_selector() {
        let selector = &keccak256("execute_message(bytes)".as_bytes())[..4];
        assert_eq!(selector, EXECUTE_MESSAGE_SELECTOR);
    }

    #[test]
    fn check_instantiate_selector() {
        let selector = &keccak256("instantiate(bytes)".as_bytes())[..4];
        assert_eq!(selector, INSTANTIATE_SELECTOR);
    }
}

fn has_instantiation_function(module: &[u8]) -> bool {
    let push4 = 0x63; // An EVM instruction
    let mut vec = vec![push4];
    vec.extend(INSTANTIATE_SELECTOR);
    module.windows(5).any(|window| window == vec)
}

#[cfg(with_metrics)]
mod metrics {
    use std::sync::LazyLock;

    use linera_base::prometheus_util::{exponential_bucket_latencies, register_histogram_vec};
    use prometheus::HistogramVec;

    pub static CONTRACT_INSTANTIATION_LATENCY: LazyLock<HistogramVec> = LazyLock::new(|| {
        register_histogram_vec(
            "evm_contract_instantiation_latency",
            "EVM contract instantiation latency",
            &[],
            exponential_bucket_latencies(1.0),
        )
    });

    pub static SERVICE_INSTANTIATION_LATENCY: LazyLock<HistogramVec> = LazyLock::new(|| {
        register_histogram_vec(
            "evm_service_instantiation_latency",
            "EVM service instantiation latency",
            &[],
            exponential_bucket_latencies(1.0),
        )
    });
}

fn get_revm_instantiation_bytes(value: Vec<u8>) -> Vec<u8> {
    use alloy_primitives::Bytes;
    use alloy_sol_types::{sol, SolCall};
    sol! {
        function instantiate(bytes value);
    }
    let bytes = Bytes::copy_from_slice(&value);
    let argument = instantiateCall { value: bytes };
    argument.abi_encode()
}

fn get_revm_execute_message_bytes(value: Vec<u8>) -> Vec<u8> {
    use alloy_primitives::Bytes;
    use alloy_sol_types::{sol, SolCall};
    sol! {
        function execute_message(bytes value);
    }
    let bytes = Bytes::copy_from_slice(&value);
    let argument = execute_messageCall { value: bytes };
    argument.abi_encode()
}

#[derive(Clone)]
pub enum EvmContractModule {
    #[cfg(with_revm)]
    Revm { module: Vec<u8> },
}

impl EvmContractModule {
    /// Creates a new [`EvmContractModule`] using the EVM module with the provided `contract_bytecode`.
    pub async fn new(
        contract_bytecode: Bytecode,
        runtime: EvmRuntime,
    ) -> Result<Self, EvmExecutionError> {
        match runtime {
            #[cfg(with_revm)]
            EvmRuntime::Revm => Self::from_revm(contract_bytecode).await,
        }
    }

    /// Creates a new [`EvmContractModule`] using the EVM module in `contract_bytecode_file`.
    #[cfg(with_fs)]
    pub async fn from_file(
        contract_bytecode_file: impl AsRef<std::path::Path>,
        runtime: EvmRuntime,
    ) -> Result<Self, EvmExecutionError> {
        Self::new(
            Bytecode::load_from_file(contract_bytecode_file)
                .await
                .map_err(anyhow::Error::from)
                .map_err(EvmExecutionError::LoadContractModule)?,
            runtime,
        )
        .await
    }

    /// Creates a new [`EvmContractModule`] using Revm with the provided bytecode files.
    pub async fn from_revm(contract_bytecode: Bytecode) -> Result<Self, EvmExecutionError> {
        let module = contract_bytecode.bytes;
        Ok(EvmContractModule::Revm { module })
    }
}

impl UserContractModule for EvmContractModule {
    fn instantiate(
        &self,
        runtime: ContractSyncRuntimeHandle,
    ) -> Result<UserContractInstance, ExecutionError> {
        #[cfg(with_metrics)]
        let _instantiation_latency = metrics::CONTRACT_INSTANTIATION_LATENCY.measure_latency();

        let instance: UserContractInstance = match self {
            #[cfg(with_revm)]
            EvmContractModule::Revm { module } => {
                Box::new(RevmContractInstance::prepare(module.to_vec(), runtime))
            }
        };

        Ok(instance)
    }
}

/// A user service in a compiled EVM module.
#[derive(Clone)]
pub enum EvmServiceModule {
    #[cfg(with_revm)]
    Revm { module: Vec<u8> },
}

impl EvmServiceModule {
    /// Creates a new [`EvmServiceModule`] using the EVM module with the provided bytecode.
    pub async fn new(
        service_bytecode: Bytecode,
        runtime: EvmRuntime,
    ) -> Result<Self, EvmExecutionError> {
        match runtime {
            #[cfg(with_revm)]
            EvmRuntime::Revm => Self::from_revm(service_bytecode).await,
        }
    }

    /// Creates a new [`EvmServiceModule`] using the EVM module in `service_bytecode_file`.
    #[cfg(with_fs)]
    pub async fn from_file(
        service_bytecode_file: impl AsRef<std::path::Path>,
        runtime: EvmRuntime,
    ) -> Result<Self, EvmExecutionError> {
        Self::new(
            Bytecode::load_from_file(service_bytecode_file)
                .await
                .map_err(anyhow::Error::from)
                .map_err(EvmExecutionError::LoadServiceModule)?,
            runtime,
        )
        .await
    }

    /// Creates a new [`EvmServiceModule`] using Revm with the provided bytecode files.
    pub async fn from_revm(contract_bytecode: Bytecode) -> Result<Self, EvmExecutionError> {
        let module = contract_bytecode.bytes;
        Ok(EvmServiceModule::Revm { module })
    }
}

impl UserServiceModule for EvmServiceModule {
    fn instantiate(
        &self,
        runtime: ServiceSyncRuntimeHandle,
    ) -> Result<UserServiceInstance, ExecutionError> {
        #[cfg(with_metrics)]
        let _instantiation_latency = metrics::SERVICE_INSTANTIATION_LATENCY.measure_latency();

        let instance: UserServiceInstance = match self {
            #[cfg(with_revm)]
            EvmServiceModule::Revm { module } => {
                Box::new(RevmServiceInstance::prepare(module.to_vec(), runtime))
            }
        };

        Ok(instance)
    }
}

// This is the precompile address that contains the Linera specific
// functionalities accessed from the EVM.
const PRECOMPILE_ADDRESS: Address = address!("000000000000000000000000000000000000000b");

// This is the zero address of the contract
const ZERO_ADDRESS: Address = address!("0000000000000000000000000000000000000000");

fn u8_slice_to_application_id(vec: &[u8]) -> ApplicationId {
    // In calls the length is 32, so no problem unwrapping
    let hash = CryptoHash::try_from(vec).unwrap();
    ApplicationId::new(hash)
}

fn address_to_user_application_id(address: Address) -> ApplicationId {
    let mut vec = vec![0_u8; 32];
    vec[..20].copy_from_slice(address.as_ref());
    ApplicationId::new(CryptoHash::try_from(&vec as &[u8]).unwrap())
}

/// Some functionalities from the BaseRuntime
#[derive(Debug, Serialize, Deserialize)]
enum BasePrecompileTag {
    /// Key prefix for `chain_id`
    ChainId,
    /// Key prefix for `application_creator_chain_id`
    ApplicationCreatorChainId,
    /// Key prefix for `chain_ownership`
    ChainOwnership,
    /// Key prefix for `read_data_blob`
    ReadDataBlob,
    /// Key prefix for `assert_data_blob_exists`
    AssertDataBlobExists,
}

/// Some functionalities from the ContractRuntime not in BaseRuntime
#[derive(Debug, Serialize, Deserialize)]
enum ContractPrecompileTag {
    /// Key prefix for `try_call_application`
    TryCallApplication,
    /// Key prefix for `validation_round`
    ValidationRound,
    /// Key prefix for `send_message`
    SendMessage,
    /// Key prefix for `message_id`
    MessageId,
    /// Key prefix for `message_is_bouncing`
    MessageIsBouncing,
}

/// Some functionalities from the ServiceRuntime not in BaseRuntime
#[derive(Debug, Serialize, Deserialize)]
enum ServicePrecompileTag {
    /// Try query application
    TryQueryApplication,
}

/// Key prefixes used to transmit precompiles.
#[derive(Debug, Serialize, Deserialize)]
enum PrecompileTag {
    Base(BasePrecompileTag),
    Contract(ContractPrecompileTag),
    Service(ServicePrecompileTag),
}

fn get_precompile_output(
    output: Vec<u8>,
    gas_limit: u64,
) -> Result<Option<InterpreterResult>, String> {
    // The gas usage is set to `gas_limit` and no spending is being done on it.
    // This means that for REVM, it looks like the precompile call costs nothing.
    // This is because the costs of the EVM precompile calls is accounted for
    // separately in Linera.
    let output = Bytes::copy_from_slice(&output);
    let result = InstructionResult::default();
    let gas = Gas::new(gas_limit);
    Ok(Some(InterpreterResult {
        result,
        output,
        gas,
    }))
}

fn base_runtime_call<Runtime: BaseRuntime>(
    tag: BasePrecompileTag,
    vec: &[u8],
    context: &mut Ctx<'_, Runtime>,
) -> Result<Vec<u8>, String> {
    let mut runtime = context
        .db()
        .0
        .runtime
        .lock()
        .expect("The lock should be possible");
    match tag {
        BasePrecompileTag::ChainId => {
            ensure!(vec.is_empty(), format!("vec should be empty"));
            let chain_id = runtime
                .chain_id()
                .map_err(|error| format!("ChainId error: {error}"))?;
            bcs::to_bytes(&chain_id).map_err(|error| format!("ChainId serialization error {error}"))
        }
        BasePrecompileTag::ApplicationCreatorChainId => {
            ensure!(vec.is_empty(), format!("vec should be empty"));
            let chain_id = runtime
                .application_creator_chain_id()
                .map_err(|error| format!("ApplicationCreatorChainId error: {error}"))?;
            bcs::to_bytes(&chain_id).map_err(|error| format!("ChainId serialization error {error}"))
        }
        BasePrecompileTag::ChainOwnership => {
            ensure!(vec.is_empty(), format!("vec should be empty"));
            let chain_ownership = runtime
                .chain_ownership()
                .map_err(|error| format!("ChainOwnership error: {error}"))?;
            bcs::to_bytes(&chain_ownership)
                .map_err(|error| format!("ChainOwnership serialization error {error}"))
        }
        BasePrecompileTag::ReadDataBlob => {
            ensure!(vec.len() == 32, format!("vec.size() should be 32"));
            let hash = CryptoHash::try_from(vec).unwrap();
            let blob = runtime
                .read_data_blob(&hash)
                .map_err(|error| format!("ReadDataBlob error: {error}"))?;
            Ok(blob)
        }
        BasePrecompileTag::AssertDataBlobExists => {
            ensure!(vec.len() == 32, format!("vec.size() should be 32"));
            let hash = CryptoHash::try_from(vec).unwrap();
            runtime
                .assert_data_blob_exists(&hash)
                .map_err(|error| format!("AssertDataBlobExists error: {error}"))?;
            Ok(Vec::new())
        }
    }
}

type Ctx<'a, Runtime> = MainnetContext<WrapDatabaseRef<&'a mut DatabaseRuntime<Runtime>>>;

fn precompile_addresses() -> BTreeSet<Address> {
    let mut addresses = BTreeSet::new();
    for address in EthPrecompiles::default().warm_addresses() {
        addresses.insert(address);
    }
    addresses.insert(PRECOMPILE_ADDRESS);
    addresses
}

#[derive(Debug, Default)]
struct ContractPrecompile {
    inner: EthPrecompiles,
}

impl<'a, Runtime: ContractRuntime> PrecompileProvider<Ctx<'a, Runtime>> for ContractPrecompile {
    type Output = InterpreterResult;

    fn set_spec(&mut self, spec: <<Ctx<'a, Runtime> as ContextTr>::Cfg as Cfg>::Spec) -> bool {
        <EthPrecompiles as PrecompileProvider<Ctx<'a, Runtime>>>::set_spec(&mut self.inner, spec)
    }

    fn run(
        &mut self,
        context: &mut Ctx<'a, Runtime>,
        address: &Address,
        inputs: &InputsImpl,
        is_static: bool,
        gas_limit: u64,
    ) -> Result<Option<InterpreterResult>, String> {
        if address == &PRECOMPILE_ADDRESS {
            let input = get_precompile_argument(context, &inputs.input);
            let output = Self::call_or_fail(&input, gas_limit, context)?;
            return get_precompile_output(output, gas_limit);
        }
        self.inner
            .run(context, address, inputs, is_static, gas_limit)
    }

    fn warm_addresses(&self) -> Box<impl Iterator<Item = Address>> {
        let mut addresses = self.inner.warm_addresses().collect::<Vec<Address>>();
        addresses.push(PRECOMPILE_ADDRESS);
        Box::new(addresses.into_iter())
    }

    fn contains(&self, address: &Address) -> bool {
        address == &PRECOMPILE_ADDRESS || self.inner.contains(address)
    }
}

const MESSAGE_IS_BOUNCING_NONE: u8 = 0;
const MESSAGE_IS_BOUNCING_SOME_TRUE: u8 = 1;
const MESSAGE_IS_BOUNCING_SOME_FALSE: u8 = 2;

impl<'a> ContractPrecompile {
    fn contract_runtime_call<Runtime: ContractRuntime>(
        tag: ContractPrecompileTag,
        vec: &[u8],
        _gas_limit: u64,
        context: &mut Ctx<'a, Runtime>,
    ) -> Result<Vec<u8>, String> {
        let mut runtime = context
            .db()
            .0
            .runtime
            .lock()
            .expect("The lock should be possible");
        match tag {
            ContractPrecompileTag::TryCallApplication => {
                ensure!(vec.len() >= 32, format!("vec.size() should be at least 32"));
                let target = u8_slice_to_application_id(&vec[0..32]);
                let argument = vec[32..].to_vec();
                let authenticated = true;
                runtime
                    .try_call_application(authenticated, target, argument)
                    .map_err(|error| format!("TryCallApplication error: {error}"))
            }
            ContractPrecompileTag::ValidationRound => {
                ensure!(vec.is_empty(), format!("vec should be empty"));
                let value = runtime
                    .validation_round()
                    .map_err(|error| format!("ValidationRound error: {error}"))?;
                bcs::to_bytes(&value).map_err(|error| format!("u32 serialization error {error}"))
            }
            ContractPrecompileTag::SendMessage => {
                ensure!(vec.len() >= 32, format!("vec.size() should be at least 32"));
                let destination = ChainId(
                    CryptoHash::try_from(&vec[..32])
                        .map_err(|error| format!("TryError: {error}"))?,
                );
                let authenticated = true;
                let is_tracked = true;
                let grant = Resources::default();
                let message = vec[32..].to_vec();
                let send_message_request = SendMessageRequest {
                    destination,
                    authenticated,
                    is_tracked,
                    grant,
                    message,
                };
                runtime
                    .send_message(send_message_request)
                    .map_err(|error| format!("SendMessage error: {error}"))?;
                Ok(vec![])
            }
            ContractPrecompileTag::MessageId => {
                ensure!(vec.is_empty(), format!("vec should be empty"));
                let message_id = runtime
                    .message_id()
                    .map_err(|error| format!("MessageId error {error}"))?;
                bcs::to_bytes(&message_id)
                    .map_err(|error| format!("MessageId serialization error {error}"))
            }
            ContractPrecompileTag::MessageIsBouncing => {
                ensure!(vec.is_empty(), format!("vec should be empty"));
                let message_is_bouncing = runtime
                    .message_is_bouncing()
                    .map_err(|error| format!("MessageIsBouncing error {error}"))?;
                let value = match message_is_bouncing {
                    None => MESSAGE_IS_BOUNCING_NONE,
                    Some(true) => MESSAGE_IS_BOUNCING_SOME_TRUE,
                    Some(false) => MESSAGE_IS_BOUNCING_SOME_FALSE,
                };
                Ok(vec![value])
            }
        }
    }

    fn call_or_fail<Runtime: ContractRuntime>(
        vec: &[u8],
        gas_limit: u64,
        context: &mut Ctx<'a, Runtime>,
    ) -> Result<Vec<u8>, String> {
        ensure!(vec.len() >= 2, format!("vec.size() should be at least 2"));
        match bcs::from_bytes(&vec[..2]).map_err(|error| format!("{error}"))? {
            PrecompileTag::Base(base_tag) => base_runtime_call(base_tag, &vec[2..], context),
            PrecompileTag::Contract(contract_tag) => {
                Self::contract_runtime_call(contract_tag, &vec[2..], gas_limit, context)
            }
            PrecompileTag::Service(_) => {
                Err("Service tags are not available in ContractPrecompile".to_string())
            }
        }
    }
}

#[derive(Debug, Default)]
struct ServicePrecompile {
    inner: EthPrecompiles,
}

impl<'a> ServicePrecompile {
    fn service_runtime_call<Runtime: ServiceRuntime>(
        tag: ServicePrecompileTag,
        vec: &[u8],
        context: &mut Ctx<'a, Runtime>,
    ) -> Result<Vec<u8>, String> {
        let mut runtime = context
            .db()
            .0
            .runtime
            .lock()
            .expect("The lock should be possible");
        match tag {
            ServicePrecompileTag::TryQueryApplication => {
                let target = u8_slice_to_application_id(&vec[..32]);
                let argument = vec[32..].to_vec();
                runtime
                    .try_query_application(target, argument)
                    .map_err(|error| format!("TryQueryApplication error: {error}"))
            }
        }
    }

    fn call_or_fail<Runtime: ServiceRuntime>(
        vec: &[u8],
        _gas_limit: u64,
        context: &mut Ctx<'a, Runtime>,
    ) -> Result<Vec<u8>, String> {
        ensure!(vec.len() >= 2, format!("vec.size() should be at least 2"));
        match bcs::from_bytes(&vec[..2]).map_err(|error| format!("{error}"))? {
            PrecompileTag::Base(base_tag) => base_runtime_call(base_tag, &vec[2..], context),
            PrecompileTag::Contract(_) => {
                Err("Contract tags are not available in ServiceContractCall".to_string())
            }
            PrecompileTag::Service(service_tag) => {
                Self::service_runtime_call(service_tag, &vec[2..], context)
            }
        }
    }
}

impl<'a, Runtime: ServiceRuntime> PrecompileProvider<Ctx<'a, Runtime>> for ServicePrecompile {
    type Output = InterpreterResult;

    fn set_spec(&mut self, spec: <<Ctx<'a, Runtime> as ContextTr>::Cfg as Cfg>::Spec) -> bool {
        <EthPrecompiles as PrecompileProvider<Ctx<'a, Runtime>>>::set_spec(&mut self.inner, spec)
    }

    fn run(
        &mut self,
        context: &mut Ctx<'a, Runtime>,
        address: &Address,
        inputs: &InputsImpl,
        is_static: bool,
        gas_limit: u64,
    ) -> Result<Option<InterpreterResult>, String> {
        if address == &PRECOMPILE_ADDRESS {
            let input = get_precompile_argument(context, &inputs.input);
            let output = Self::call_or_fail(&input, gas_limit, context)?;
            return get_precompile_output(output, gas_limit);
        }
        self.inner
            .run(context, address, inputs, is_static, gas_limit)
    }

    fn warm_addresses(&self) -> Box<impl Iterator<Item = Address>> {
        let mut addresses = self.inner.warm_addresses().collect::<Vec<Address>>();
        addresses.push(PRECOMPILE_ADDRESS);
        Box::new(addresses.into_iter())
    }

    fn contains(&self, address: &Address) -> bool {
        address == &PRECOMPILE_ADDRESS || self.inner.contains(address)
    }
}

fn failing_outcome() -> CallOutcome {
    let result = InstructionResult::Revert;
    let output = Bytes::default();
    let gas = Gas::default();
    let result = InterpreterResult {
        result,
        output,
        gas,
    };
    let memory_offset = Range::default();
    CallOutcome {
        result,
        memory_offset,
    }
}

fn get_interpreter_result(
    result: &[u8],
    inputs: &mut CallInputs,
) -> Result<InterpreterResult, ExecutionError> {
    let mut result = bcs::from_bytes::<InterpreterResult>(result)?;
    // This effectively means that no cost is incurred by the call to another contract.
    // This is fine since the costs are incurred by the other contract itself.
    result.gas = Gas::new(inputs.gas_limit);
    Ok(result)
}

struct CallInterceptorContract<Runtime> {
    db: DatabaseRuntime<Runtime>,
    // This is the contract address of the contract being created.
    contract_address: Address,
    precompile_addresses: BTreeSet<Address>,
}

impl<Runtime> Clone for CallInterceptorContract<Runtime> {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            contract_address: self.contract_address,
            precompile_addresses: self.precompile_addresses.clone(),
        }
    }
}

fn get_argument<Ctx: ContextTr>(context: &mut Ctx, argument: &mut Vec<u8>, input: &CallInput) {
    match input {
        CallInput::Bytes(bytes) => {
            argument.extend(bytes.to_vec());
        }
        CallInput::SharedBuffer(range) => {
            if let Some(slice) = context.local().shared_memory_buffer_slice(range.clone()) {
                argument.extend(&*slice);
            }
        }
    };
}

fn get_call_argument<Ctx: ContextTr>(context: &mut Ctx, input: &CallInput) -> Vec<u8> {
    let mut argument: Vec<u8> = INTERPRETER_RESULT_SELECTOR.to_vec();
    get_argument(context, &mut argument, input);
    argument
}

fn get_precompile_argument<Ctx: ContextTr>(context: &mut Ctx, input: &CallInput) -> Vec<u8> {
    let mut argument = Vec::new();
    get_argument(context, &mut argument, input);
    argument
}

impl<'a, Runtime: ContractRuntime> Inspector<Ctx<'a, Runtime>>
    for CallInterceptorContract<Runtime>
{
    fn create(
        &mut self,
        _context: &mut Ctx<'a, Runtime>,
        inputs: &mut CreateInputs,
    ) -> Option<CreateOutcome> {
        inputs.scheme = CreateScheme::Custom {
            address: self.contract_address,
        };
        None
    }

    fn call(
        &mut self,
        context: &mut Ctx<'a, Runtime>,
        inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        let result = self.call_or_fail(context, inputs);
        match result {
            Err(_error) => {
                // An alternative way would be to return None, which would induce
                // Revm to call the smart contract in its database, where it is
                // non-existent.
                Some(failing_outcome())
            }
            Ok(result) => result,
        }
    }
}

impl<Runtime: ContractRuntime> CallInterceptorContract<Runtime> {
    fn call_or_fail(
        &mut self,
        context: &mut Ctx<'_, Runtime>,
        inputs: &mut CallInputs,
    ) -> Result<Option<CallOutcome>, ExecutionError> {
        // Every call to a contract passes by this function.
        // Three kinds:
        // --- Call to the PRECOMPILE smart contract.
        // --- Call to the EVM smart contract itself
        // --- Call to other EVM smart contract
        if self.precompile_addresses.contains(&inputs.target_address)
            || inputs.target_address == self.contract_address
        {
            // Precompile calls are handled by the precompile code.
            // The EVM smart contract is being called
            return Ok(None);
        }
        // Other smart contracts calls are handled by the runtime
        let target = address_to_user_application_id(inputs.target_address);
        let argument = get_call_argument(context, &inputs.input);
        let authenticated = true;
        let result = {
            let mut runtime = self.db.runtime.lock().expect("The lock should be possible");
            runtime.try_call_application(authenticated, target, argument)?
        };
        let call_outcome = CallOutcome {
            result: get_interpreter_result(&result, inputs)?,
            memory_offset: inputs.return_memory_offset.clone(),
        };
        Ok(Some(call_outcome))
    }
}

struct CallInterceptorService<Runtime> {
    db: DatabaseRuntime<Runtime>,
    // This is the contract address of the contract being created.
    contract_address: Address,
    precompile_addresses: BTreeSet<Address>,
}

impl<Runtime> Clone for CallInterceptorService<Runtime> {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            contract_address: self.contract_address,
            precompile_addresses: self.precompile_addresses.clone(),
        }
    }
}

impl<'a, Runtime: ServiceRuntime> Inspector<Ctx<'a, Runtime>> for CallInterceptorService<Runtime> {
    fn create(
        &mut self,
        _context: &mut Ctx<'a, Runtime>,
        inputs: &mut CreateInputs,
    ) -> Option<CreateOutcome> {
        inputs.scheme = CreateScheme::Custom {
            address: self.contract_address,
        };
        None
    }

    fn call(
        &mut self,
        context: &mut Ctx<'a, Runtime>,
        inputs: &mut CallInputs,
    ) -> Option<CallOutcome> {
        let result = self.call_or_fail(context, inputs);
        match result {
            Err(_error) => {
                // An alternative way would be to return None, which would induce
                // Revm to call the smart contract in its database, where it is
                // non-existent.
                Some(failing_outcome())
            }
            Ok(result) => result,
        }
    }
}

impl<Runtime: ServiceRuntime> CallInterceptorService<Runtime> {
    fn call_or_fail(
        &mut self,
        context: &mut Ctx<'_, Runtime>,
        inputs: &mut CallInputs,
    ) -> Result<Option<CallOutcome>, ExecutionError> {
        // Every call to a contract passes by this function.
        // Three kinds:
        // --- Call to the PRECOMPILE smart contract.
        // --- Call to the EVM smart contract itself
        // --- Call to other EVM smart contract
        if self.precompile_addresses.contains(&inputs.target_address)
            || inputs.target_address == self.contract_address
        {
            // Precompile calls are handled by the precompile code.
            // The EVM smart contract is being called
            return Ok(None);
        }
        // Other smart contracts calls are handled by the runtime
        let target = address_to_user_application_id(inputs.target_address);
        let argument = get_call_argument(context, &inputs.input);
        let result = {
            let evm_query = EvmQuery::Query(argument);
            let evm_query = serde_json::to_vec(&evm_query)?;
            let mut runtime = self.db.runtime.lock().expect("The lock should be possible");
            runtime.try_query_application(target, evm_query)?
        };
        let call_outcome = CallOutcome {
            result: get_interpreter_result(&result, inputs)?,
            memory_offset: inputs.return_memory_offset.clone(),
        };
        Ok(Some(call_outcome))
    }
}

pub struct RevmContractInstance<Runtime> {
    module: Vec<u8>,
    db: DatabaseRuntime<Runtime>,
}

enum Choice {
    Create,
    Call,
}

#[derive(Debug)]
struct ExecutionResultSuccess {
    reason: SuccessReason,
    gas_final: u64,
    logs: Vec<Log>,
    output: Output,
}

impl ExecutionResultSuccess {
    fn interpreter_result_and_logs(self) -> Result<(u64, Vec<u8>, Vec<Log>), ExecutionError> {
        let result: InstructionResult = self.reason.into();
        let Output::Call(output) = self.output else {
            unreachable!("The output should have been created from a Choice::Call");
        };
        let gas = Gas::new(0);
        let result = InterpreterResult {
            result,
            output,
            gas,
        };
        let result = bcs::to_bytes(&result)?;
        Ok((self.gas_final, result, self.logs))
    }

    fn output_and_logs(self) -> (u64, Vec<u8>, Vec<Log>) {
        let Output::Call(output) = self.output else {
            unreachable!("The output should have been created from a Choice::Call");
        };
        let output = output.as_ref().to_vec();
        (self.gas_final, output, self.logs)
    }

    // Checks that the contract has been correctly instantiated
    fn check_contract_initialization(&self, expected_address: Address) -> Result<(), String> {
        // Checks that the output is the expected one.
        let Output::Create(_, contract_address) = self.output else {
            return Err("Input should be Choice::Create".to_string());
        };
        // Checks that the contract address exists.
        let contract_address = contract_address.ok_or("Deployment failed")?;
        // Checks that the created contract address is the one of the `ApplicationId`.
        if contract_address == expected_address {
            Ok(())
        } else {
            Err("Contract address is not the same as ApplicationId".to_string())
        }
    }
}

impl<Runtime> UserContract for RevmContractInstance<Runtime>
where
    Runtime: ContractRuntime,
{
    fn instantiate(&mut self, argument: Vec<u8>) -> Result<(), ExecutionError> {
        self.db.set_contract_address()?;
        self.initialize_contract()?;
        if has_instantiation_function(&self.module) {
            let instantiation_argument = serde_json::from_slice::<Vec<u8>>(&argument)?;
            let argument = get_revm_instantiation_bytes(instantiation_argument);
            let result = self.transact_commit(Choice::Call, &argument)?;
            self.write_logs(result.logs, "instantiate")?;
        }
        Ok(())
    }

    fn execute_operation(&mut self, operation: Vec<u8>) -> Result<Vec<u8>, ExecutionError> {
        self.db.set_contract_address()?;
        ensure_message_length(operation.len(), 4)?;
        let (gas_final, output, logs) = if &operation[..4] == INTERPRETER_RESULT_SELECTOR {
            ensure_message_length(operation.len(), 8)?;
            forbid_execute_operation_origin(&operation[4..8])?;
            let result = self.init_transact_commit(Choice::Call, &operation[4..])?;
            result.interpreter_result_and_logs()?
        } else {
            ensure_message_length(operation.len(), 4)?;
            forbid_execute_operation_origin(&operation[..4])?;
            let result = self.init_transact_commit(Choice::Call, &operation)?;
            result.output_and_logs()
        };
        self.consume_fuel(gas_final)?;
        self.write_logs(logs, "operation")?;
        Ok(output)
    }

    fn execute_message(&mut self, message: Vec<u8>) -> Result<(), ExecutionError> {
        self.db.set_contract_address()?;
        let operation = get_revm_execute_message_bytes(message);
        let result = self.init_transact_commit(Choice::Call, &operation)?;
        let (gas_final, output, logs) = result.output_and_logs();
        self.consume_fuel(gas_final)?;
        self.write_logs(logs, "message")?;
        assert_eq!(output.len(), 0);
        Ok(())
    }

    fn process_streams(&mut self, _streams: Vec<StreamUpdate>) -> Result<(), ExecutionError> {
        // TODO(#3785): Implement process_streams for EVM
        todo!("Streams are not implemented for Ethereum smart contracts yet.")
    }

    fn finalize(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }
}

fn process_execution_result(
    storage_stats: StorageStats,
    result: ExecutionResult,
) -> Result<ExecutionResultSuccess, ExecutionError> {
    match result {
        ExecutionResult::Success {
            reason,
            gas_used,
            gas_refunded,
            logs,
            output,
        } => {
            let mut gas_final = gas_used;
            gas_final -= storage_stats.storage_costs();
            assert_eq!(gas_refunded, storage_stats.storage_refund());
            Ok(ExecutionResultSuccess {
                reason,
                gas_final,
                logs,
                output,
            })
        }
        ExecutionResult::Revert { gas_used, output } => {
            let error = EvmExecutionError::Revert { gas_used, output };
            Err(ExecutionError::EvmError(error))
        }
        ExecutionResult::Halt { gas_used, reason } => {
            let error = EvmExecutionError::Halt { gas_used, reason };
            Err(ExecutionError::EvmError(error))
        }
    }
}

impl<Runtime> RevmContractInstance<Runtime>
where
    Runtime: ContractRuntime,
{
    pub fn prepare(module: Vec<u8>, runtime: Runtime) -> Self {
        let db = DatabaseRuntime::new(runtime);
        Self { module, db }
    }

    /// Executes the transaction. If needed initializes the contract.
    fn init_transact_commit(
        &mut self,
        ch: Choice,
        vec: &[u8],
    ) -> Result<ExecutionResultSuccess, ExecutionError> {
        // An application can be instantiated in Linera sense, but not in EVM sense,
        // that is the contract entries corresponding to the deployed contract may
        // be missing.
        if !self.db.is_initialized()? {
            self.initialize_contract()?;
        }
        self.transact_commit(ch, vec)
    }

    /// Initializes the contract.
    fn initialize_contract(&mut self) -> Result<(), ExecutionError> {
        let mut vec_init = self.module.clone();
        let constructor_argument = self.db.constructor_argument()?;
        vec_init.extend_from_slice(&constructor_argument);
        let result = self.transact_commit(Choice::Create, &vec_init)?;
        result
            .check_contract_initialization(self.db.contract_address)
            .map_err(|error| {
                ExecutionError::EvmError(EvmExecutionError::IncorrectContractCreation(error))
            })?;
        self.write_logs(result.logs, "deploy")
    }

    fn transact_commit(
        &mut self,
        ch: Choice,
        input: &[u8],
    ) -> Result<ExecutionResultSuccess, ExecutionError> {
        let (kind, data) = match ch {
            Choice::Create => (TxKind::Create, Bytes::copy_from_slice(input)),
            Choice::Call => {
                let data = Bytes::copy_from_slice(input);
                (TxKind::Call(self.db.contract_address), data)
            }
        };
        let inspector = CallInterceptorContract {
            db: self.db.clone(),
            contract_address: self.db.contract_address,
            precompile_addresses: precompile_addresses(),
        };
        let block_env = self.db.get_contract_block_env()?;
        let gas_limit = {
            let mut runtime = self.db.runtime.lock().expect("The lock should be possible");
            runtime.remaining_fuel(VmRuntime::Evm)?
        };
        let nonce = self.db.get_nonce(&ZERO_ADDRESS)?;
        let result = {
            let ctx: revm_context::Context<
                BlockEnv,
                _,
                _,
                _,
                Journal<WrapDatabaseRef<&mut DatabaseRuntime<Runtime>>>,
                (),
            > = revm_context::Context::<BlockEnv, _, _, _, _, _>::new(
                WrapDatabaseRef(&mut self.db),
                SpecId::PRAGUE,
            )
            .with_block(block_env);
            let instructions = EthInstructions::new_mainnet();
            let mut evm = Evm::new_with_inspector(
                ctx,
                inspector.clone(),
                instructions,
                ContractPrecompile::default(),
            );
            evm.inspect_commit(
                TxEnv {
                    kind,
                    data,
                    nonce,
                    gas_limit,
                    ..TxEnv::default()
                },
                inspector,
            )
            .map_err(|error| {
                let error = format!("{:?}", error);
                let error = EvmExecutionError::TransactCommitError(error);
                ExecutionError::EvmError(error)
            })
        }?;
        let storage_stats = self.db.take_storage_stats();
        self.db.commit_changes()?;
        process_execution_result(storage_stats, result)
    }

    fn consume_fuel(&mut self, gas_final: u64) -> Result<(), ExecutionError> {
        let mut runtime = self.db.runtime.lock().expect("The lock should be possible");
        runtime.consume_fuel(gas_final, VmRuntime::Evm)
    }

    fn write_logs(&mut self, logs: Vec<Log>, origin: &str) -> Result<(), ExecutionError> {
        // TODO(#3758): Extracting Ethereum events from the Linera events.
        if !logs.is_empty() {
            let mut runtime = self.db.runtime.lock().expect("The lock should be possible");
            let block_height = runtime.block_height()?;
            let stream_name = bcs::to_bytes("ethereum_event")?;
            let stream_name = StreamName(stream_name);
            for log in &logs {
                let value = bcs::to_bytes(&(origin, block_height.0, log))?;
                runtime.emit(stream_name.clone(), value)?;
            }
        }
        Ok(())
    }
}

pub struct RevmServiceInstance<Runtime> {
    module: Vec<u8>,
    db: DatabaseRuntime<Runtime>,
}

impl<Runtime> RevmServiceInstance<Runtime>
where
    Runtime: ServiceRuntime,
{
    pub fn prepare(module: Vec<u8>, runtime: Runtime) -> Self {
        let db = DatabaseRuntime::new(runtime);
        Self { module, db }
    }
}

impl<Runtime> UserService for RevmServiceInstance<Runtime>
where
    Runtime: ServiceRuntime,
{
    fn handle_query(&mut self, argument: Vec<u8>) -> Result<Vec<u8>, ExecutionError> {
        self.db.set_contract_address()?;
        let evm_query = serde_json::from_slice(&argument)?;
        let query = match evm_query {
            EvmQuery::Query(vec) => vec,
            EvmQuery::Mutation(operation) => {
                let mut runtime = self.db.runtime.lock().expect("The lock should be possible");
                runtime.schedule_operation(operation)?;
                return Ok(Vec::new());
            }
        };

        ensure_message_length(query.len(), 4)?;
        // We drop the logs since the "eth_call" execution does not return any log.
        // Also, for handle_query, we do not have associated costs.
        // More generally, there is gas costs associated to service operation.
        let answer = if &query[..4] == INTERPRETER_RESULT_SELECTOR {
            let result = self.init_transact(&query[4..])?;
            let (_gas_final, answer, _logs) = result.interpreter_result_and_logs()?;
            answer
        } else {
            let result = self.init_transact(&query)?;
            let (_gas_final, output, _logs) = result.output_and_logs();
            serde_json::to_vec(&output)?
        };
        Ok(answer)
    }
}

impl<Runtime> RevmServiceInstance<Runtime>
where
    Runtime: ServiceRuntime,
{
    fn init_transact(&mut self, vec: &[u8]) -> Result<ExecutionResultSuccess, ExecutionError> {
        // In case of a shared application, we need to instantiate it first
        // However, since in ServiceRuntime, we cannot modify the storage,
        // therefore the compiled contract is saved in the changes.
        if !self.db.is_initialized()? {
            let changes = {
                let mut vec_init = self.module.clone();
                let constructor_argument = self.db.constructor_argument()?;
                vec_init.extend_from_slice(&constructor_argument);
                let (result, changes) = self.transact(TxKind::Create, &vec_init)?;
                result
                    .check_contract_initialization(self.db.contract_address)
                    .map_err(|error| {
                        ExecutionError::EvmError(EvmExecutionError::IncorrectContractCreation(
                            error,
                        ))
                    })?;
                changes
            };
            self.db.changes = changes;
        }
        ensure_message_length(vec.len(), 4)?;
        forbid_execute_operation_origin(&vec[..4])?;
        let kind = TxKind::Call(self.db.contract_address);
        let (execution_result, _) = self.transact(kind, vec)?;
        Ok(execution_result)
    }

    fn transact(
        &mut self,
        kind: TxKind,
        input: &[u8],
    ) -> Result<(ExecutionResultSuccess, EvmState), ExecutionError> {
        let data = Bytes::copy_from_slice(input);

        let block_env = self.db.get_service_block_env()?;
        let inspector = CallInterceptorService {
            db: self.db.clone(),
            contract_address: self.db.contract_address,
            precompile_addresses: precompile_addresses(),
        };
        let nonce = self.db.get_nonce(&ZERO_ADDRESS)?;
        let result_state = {
            let ctx: revm_context::Context<
                BlockEnv,
                _,
                _,
                _,
                Journal<WrapDatabaseRef<&mut DatabaseRuntime<Runtime>>>,
                (),
            > = revm_context::Context::<BlockEnv, _, _, _, _, _>::new(
                WrapDatabaseRef(&mut self.db),
                SpecId::PRAGUE,
            )
            .with_block(block_env);
            let instructions = EthInstructions::new_mainnet();
            let mut evm = Evm::new_with_inspector(
                ctx,
                inspector.clone(),
                instructions,
                ServicePrecompile::default(),
            );
            evm.inspect(
                TxEnv {
                    kind,
                    data,
                    nonce,
                    gas_limit: EVM_SERVICE_GAS_LIMIT,
                    ..TxEnv::default()
                },
                inspector,
            )
            .map_err(|error| {
                let error = format!("{:?}", error);
                let error = EvmExecutionError::TransactCommitError(error);
                ExecutionError::EvmError(error)
            })
        }?;
        let storage_stats = self.db.take_storage_stats();
        Ok((
            process_execution_result(storage_stats, result_state.result)?,
            result_state.state,
        ))
    }
}
