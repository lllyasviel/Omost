from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
# from mlx_lm import load, PreTrainedTokenizer, TokenizerWrapper
import mlx
import mlx_lm
import transformers as tf
# from transformers import AutoTokenizer, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
import numpy as np
import torch

def load_mlx_lm(llm_name: str) -> Tuple[mlx.nn.Module, tf.PreTrainedTokenizer]:
  llm_model, llm_tokenizer = mlx_lm.load(llm_name)
  return MLX_LLM_TransformersWrapper(llm_model, llm_tokenizer), llm_tokenizer

class MLX_LLM_TransformersWrapper(mlx.nn.Module):
  def __init__(self, model: mlx.nn.Module, tokenizer: tf.PreTrainedTokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self,
    input_ids: np.ndarray,
    streamer: tf.TextIteratorStreamer, #Optional["BaseStreamer"] = None,
    # inputs: Optional[torch.Tensor] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_new_tokens: int = 100,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    **kwargs
  ) -> Union[GenerateOutput, torch.LongTensor]:

    if streamer is not None:
      streamer.put(input_ids.cpu())

    # has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    # return self.__stream_generate(self.model, self.tokenizer, input_ids, max_new_tokens, **kwargs)


  def __stream_generate(self,
      model: torch.nn.Module,
      tokenizer: tf.PreTrainedTokenizer,
      prompt: Union[str, np.ndarray],
      max_tokens: int = 100,
      **kwargs,
  ) -> Union[str, Generator[str, None, None]]:
      """
      A generator producing text based on the given prompt from the model.

      Args:
          prompt (mx.array): The input prompt.
          model (nn.Module): The model to use for generation.
          max_tokens (int): The ma
          kwargs: The remaining options get passed to :func:`generate_step`.
            See :func:`generate_step` for more details.

      Yields:
          Generator[Tuple[mx.array, mx.array]]: A generator producing text.
      """
      # if not isinstance(tokenizer, TokenizerWrapper):
      #     tokenizer = TokenizerWrapper(tokenizer)

      if isinstance(prompt, str):
        prompt_tokens = mx.array(tokenizer.encode(prompt))
      else:
        prompt_tokens = mx.array(prompt)

      detokenizer = tokenizer.detokenizer
      detokenizer.reset()
      print("generating...")
      for (token, prob), n in zip(
          generate_step(
            prompt=prompt_tokens,
            model=model,
            temp=kwargs.pop("temperature", 1.0),
            **kwargs),
          range(max_tokens),
      ):
          print(f"n: {n}")
          if token == tokenizer.eos_token_id:
              print("EOS")
              break
          detokenizer.add_token(token)
          print(f"Token: {token}")
          # Yield the last segment if streaming
          yield detokenizer.last_segment

      detokenizer.finalize()
      yield detokenizer.last_segment
