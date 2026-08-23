"""
core/model_manager.py

ModelManager: loads OpenCUA-7B + optional LoRA adapter, handles base64
image decoding, the media-placeholder token expansion (id 151664 in
the OpenCUA tokenizer), and both vision and text-only generation.

`generate_vision` accepts a `greedy_after_code` flag — when True and
temperature > 0, attaches the GreedyAfterCodeHeader logits processor
to switch to greedy decoding once the Code section header is reached.

This is lifted from framework_api_v2.py with no behavioral changes.
"""

import base64
import logging
from io import BytesIO

import torch
from PIL import Image
from transformers import LogitsProcessorList

from core.decoupled import GreedyAfterCodeHeader

log = logging.getLogger("core.model_manager")


class ModelManager:
    """Loads the model + optional LoRA adapter and runs inference."""

    # The single sentinel token that OpenCUA's chat template inserts to mark
    # an image's position. At runtime we expand each occurrence into N copies
    # where N = product of the image_grid_thw dims // 4. This must match
    # opencua_api.py's behavior exactly or the model's cross-attention to
    # image patches breaks.
    MEDIA_PLACEHOLDER_TOKEN_ID = 151664

    def __init__(self, model_dir, adapter_repo=None):
        self.model_dir = model_dir
        self.adapter_repo = adapter_repo
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.has_adapter = False

    def load(self):
        """Load model, tokenizer, image processor, and optionally LoRA adapter."""
        # Imports are deferred so that smoke tests / preprocess can import
        # this module without paying the transformers import cost when
        # they only use the static methods below.
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

        log.info(f"Loading tokenizer from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )

        log.info(f"Loading image processor from {self.model_dir}...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True, use_fast=False
        )

        log.info(f"Loading model from {self.model_dir}...")
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        if self.adapter_repo:
            log.info(f"Loading LoRA adapter from {self.adapter_repo}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_repo)
            self.has_adapter = True
            log.info("LoRA adapter loaded.")

        self.model.eval()
        log.info(f"Model ready. Device: {next(self.model.parameters()).device}")

    # ------------------------------------------------------------------ Images

    @staticmethod
    def decode_image(base64_string):
        """Decode a base64 string to PIL Image."""
        image_bytes = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def extract_images(messages):
        """Extract all PIL images from chat-format messages (data:image/png URLs)."""
        images = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url:
                        encoded = url.replace("data:image/png;base64,", "")
                        images.append(ModelManager.decode_image(encoded))
        return images

    # ------------------------------------------------------------------ Generation

    def generate_vision(self, messages, max_new_tokens=1024,
                        temperature=0.0, top_p=0.9, stop_at=None,
                        use_adapter=False, greedy_after_code=False):
        """
        Generate from chat messages containing images.

        Args:
            messages: Chat messages, may contain image_url parts with base64 data
            max_new_tokens: Max tokens to generate
            temperature: 0 = greedy, > 0 = sampling
            top_p: Nucleus sampling threshold (only used when temperature > 0)
            stop_at: Optional str or list of strs to truncate output at
                     (truncates at the earliest match across the list)
            use_adapter: Whether to enable the LoRA adapter (needs has_adapter=True)
            greedy_after_code: If True and temperature > 0, attach the
                GreedyAfterCodeHeader logits processor — temperature sampling
                for Thought/Action, greedy decoding from the Code section onward.

        Returns:
            Generated text string (post-decoding, with stop-string truncation).
        """
        images = self.extract_images(messages)

        # Tokenize with chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        # Process images and expand media placeholder tokens
        if images:
            info = self.image_processor.preprocess(images=images)
            pixel_values = torch.tensor(info["pixel_values"]).to(
                dtype=torch.bfloat16, device=self.model.device
            )
            grid_thws = torch.tensor(info["image_grid_thw"])

            num_image_tokens = (
                grid_thws[0, 0] * grid_thws[0, 1] * grid_thws[0, 2]
            ) // 4
            expanded = []
            for tid in input_ids:
                if tid == self.MEDIA_PLACEHOLDER_TOKEN_ID:
                    expanded.extend([self.MEDIA_PLACEHOLDER_TOKEN_ID] * num_image_tokens)
                else:
                    expanded.append(tid)
            input_ids = expanded
        else:
            pixel_values = None
            grid_thws = None

        input_tensor = torch.tensor([input_ids]).to(self.model.device)
        attention_mask = torch.ones_like(input_tensor)
        prompt_len = input_tensor.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        # Decoupled generation: greedy once we hit Code section
        if greedy_after_code and temperature > 0:
            processor = GreedyAfterCodeHeader(self.tokenizer, prompt_len)
            gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
            log.info("    Decoupled generation: temp sampling → greedy at Code")

        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
            gen_kwargs["image_grid_thw"] = grid_thws

        # PeftModel has adapter enabled by default after from_pretrained().
        # We use disable_adapter() as a context manager when we want the base model.
        with torch.no_grad():
            if not use_adapter and self.has_adapter:
                with self.model.disable_adapter():
                    output_ids = self.model.generate(input_tensor, **gen_kwargs)
            else:
                output_ids = self.model.generate(input_tensor, **gen_kwargs)

        generated = output_ids[:, prompt_len:]
        text = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # Truncate at the earliest stop string
        if stop_at:
            stop_strings = stop_at if isinstance(stop_at, list) else [stop_at]
            earliest_pos = len(text)
            for s in stop_strings:
                pos = text.find(s)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            if earliest_pos < len(text):
                text = text[:earliest_pos].strip()

        return text

    def generate_text_only(self, messages, max_new_tokens=512,
                           temperature=0.0, use_adapter=False):
        """
        Generate from text-only messages (no images).
        Used for world model transition prediction and LLM-as-judge scoring.
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        input_tensor = torch.tensor([input_ids]).to(self.model.device)
        attention_mask = torch.ones_like(input_tensor)
        prompt_len = input_tensor.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            if not use_adapter and self.has_adapter:
                with self.model.disable_adapter():
                    output_ids = self.model.generate(input_tensor, **gen_kwargs)
            else:
                output_ids = self.model.generate(input_tensor, **gen_kwargs)

        generated = output_ids[:, prompt_len:]
        text = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return text