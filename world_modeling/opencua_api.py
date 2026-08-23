from flask import Flask, request, jsonify
import base64
import sys
from PIL import Image
from io import BytesIO
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
)

MODEL_DIR = "OpenCUA-7B"

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True)
# image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)

def truncate_image_data(obj, max_length=50):
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == "url" and isinstance(value, str):
                # Truncate image data
                result[key] = f"{value[:max_length]}..." if len(value) > max_length else value
            else:
                result[key] = truncate_image_data(value, max_length)
        return result
    
    elif isinstance(obj, list):
        return [truncate_image_data(item, max_length) for item in obj]
    
    else:
        return obj

def decode_image(base64_string) -> Image.Image:
    image_bytes = base64.b64decode(base64_string)
    image_file = BytesIO(image_bytes)
    image = Image.open(image_file).convert('RGB')
    return image

def get_user_images(messages):
    rgb_images = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", [])
            for item in content:
                if item.get("type") == "image_url":
                    encoded_with_prefix = item.get("image_url", {}).get("url")
                    if encoded_with_prefix:
                        encoded_image = encoded_with_prefix.replace("data:image/png;base64,", "")
                        rgb_image = decode_image(encoded_image)
                        rgb_images.append(rgb_image)
    for index, image in enumerate(rgb_images, start=1):
        filename = f"screenshot_step_{index}.png"
        # image.save(filename)
    return rgb_images



@app.route('/v1/chat/completions', methods=["POST"])
def create_chat_completion():
    data = request.json

    # cleaned = truncate_image_data(data, max_length=50)
    # print(json.dumps(cleaned, indent=2))

    messages = data.get('messages', [])
    rgb_images = get_user_images(messages)

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    info = image_processor.preprocess(images=rgb_images)
    pixel_values = torch.tensor(info['pixel_values']).to(dtype=torch.bfloat16, device=model.device)
    grid_thws = torch.tensor(info['image_grid_thw'])


    # FIX: Calculate token count per image and replace single
    # media placeholder token with multiple copies equal to that count
    num_image_tokens = (grid_thws[0, 0] * grid_thws[0, 1] * grid_thws[0, 2]) // 4
    media_placeholder_token_id = 151664
    input_ids_expanded = []
    for token_id in input_ids:
        if token_id == media_placeholder_token_id:
            input_ids_expanded.extend([media_placeholder_token_id] * num_image_tokens)
        else:
            input_ids_expanded.append(token_id)

    input_ids = torch.tensor([input_ids_expanded]).to(model.device)
    
    generated_ids = model.generate(
        input_ids, 
        pixel_values=pixel_values, 
        image_grid_thw=grid_thws,   # FIX: parameter name is image_grid_thw instead of grid_thws
        max_new_tokens=512,
        temperature=0
        )

    prompt_len = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # print("="*100)
    # for text in output_text:
    #     print(text)
    #     print("="*100)


    response = {"choices": [{"finish_reason": "stop", "message": {"content": output_text[0]}}]}

    return jsonify(response), 200


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}, 200


@app.route('/shutdown', methods=['POST'])
def shutdown():
    sys.exit(0)
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=9009)