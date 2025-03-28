import os
from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

import torch
import flash_attn_2_cuda as flash_attn_gpu

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, LoRAModulePath, PromptAdapterPath, BaseModelPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

from middleware import RequestResponseLoggingMiddleware

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["VLLM_DO_NOT_TRACK"] = "0"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "8"
os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-63"
os.environ["RAY_DEDUP_LOGS"] = "0" 
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

logger = logging.getLogger("ray.serve")

app = FastAPI()
app.add_middleware(RequestResponseLoggingMiddleware)


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        logger.info(f"Starting with engine args: {engine_args.model}")
        logger.info(f"vLLM Attention Backend: {os.getenv('VLLM_ATTENTION_BACKEND')}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _remove_quantized_method_name(
        self, 
        model_name: BaseModelPath
    )->str:
        return BaseModelPath(
            name=model_name.name
                      .replace("-AWQ", "")
                      .replace("-QAT", "")
                      .replace("-GGUF", "")
                      .replace("-QAT-INT8", "")
                      .replace("-GGUF-INT8", ""),
            model_path=model_name.model_path)
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, 
        request: ChatCompletionRequest, 
        raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            base_model_path = BaseModelPath(
                name=self.engine_args.model,
                model_path=self.engine_args.model)
            if self.engine_args.served_model_name is not None:
                served_model_names = self._remove_quantized_method_name(base_model_path) # self.engine_args.served_model_name
            else:
                served_model_names = [self._remove_quantized_method_name(base_model_path)] # [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    model_config=model_config,
                    base_model_paths=served_model_names,
                    lora_modules=self.lora_modules,
                    prompt_adapters=self.prompt_adapters),
                response_role=self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
                return_tokens_as_token_ids=False,
                enable_reasoning=False,
                reasoning_parser=None,
                enable_auto_tools=False,
                tool_parser=None,
                enable_prompt_tokens_details=False,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request=request,
            raw_request=raw_request
        )
        if isinstance(generator, ErrorResponse):
            logger.error(f"Error response: {generator}")
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = [
        "--enforce_eager", 
        # "--ray_workers_use_nsight", 
        "--trust_remote_code",
        # "--disable_sliding_window",
        # "--enable_prefix_caching"
        ]
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    if "accelerator" in cli_args.keys():
        accelerator = cli_args.pop("accelerator")
    else:
        accelerator = "GPU"
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.enforce_eager = True
    # engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        print(f"Adding accelerator {accelerator}")
        pg_resources.append({"CPU": 1, accelerator: 1})  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    serve.start(
        proxy_location="EveryNode", 
        http_options={"host": "0.0.0.0", "port": cli_args.get("port", 8031)},
        )
    return (VLLMDeployment.options(
                # placement_group_bundles=[{
                #     "CPU": 1.0, 
                #     "GPU": float(torch.cuda.is_available())
                #     }], 
                # placement_group_strategy="STRICT_PACK"
            )
            .bind(
                engine_args,
                parsed_args.response_role,
                parsed_args.lora_modules,
                parsed_args.prompt_adapters,
                cli_args.get("request_logger"),
                parsed_args.chat_template,
            ))

