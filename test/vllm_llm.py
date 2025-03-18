from vllm import LLM, SamplingParams
from transformers import Gemma3ForConditionalGeneration, Gemma3Config
# prompts = [
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(
#     model="google/gemma-3-27b-it",
#     tensor_parallel_size=1,
#     use_v2_block_manager=True,
# )
# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-27b-it")
