# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Model configuration registry — RK3588 port.

Ported from upstream multimodal-embedding-serving/src/models/config.py.
Changes:
- Removed MobileCLIP, CN-CLIP, SigLIP, Blip2 model families entirely.
- Retained CLIP and QwenText families.
- Replaced use_openvino / EMBEDDING_USE_OV with use_npu / USE_NPU.
- Added onnx_path / rknn_path config keys for the CLIP handler.
- Added the Qwen/Qwen3-VL-Embedding-2B entry (primary model for this port).
"""

import os


def default_image_probs(image_features, text_features):
    """
    Compute a probability distribution over images for each text embedding using CLIP-style scaled cosine similarity.
    
    Parameters:
        image_features (Tensor): Image feature matrix of shape (num_images, D).
        text_features (Tensor): Text feature matrix of shape (num_texts, D).
    
    Returns:
        Tensor: A matrix of shape (num_texts, num_images) where each row is a probability distribution over images for the corresponding text (values in [0, 1], rows sum to 1).
    """
    image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    return image_probs


# ---------------------------------------------------------------------------
# Model configurations — only CLIP and Qwen text embedders are retained.
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "CLIP": {
        "clip-vit-b-32": {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "image_size": 224,
            "handler_class": "CLIPHandler",
            "image_probs": default_image_probs,
        },
        "clip-vit-b-16": {
            "model_name": "ViT-B-16",
            "pretrained": "openai",
            "image_size": 224,
            "handler_class": "CLIPHandler",
            "image_probs": default_image_probs,
        },
        "clip-vit-l-14": {
            "model_name": "ViT-L-14",
            "pretrained": "datacomp_xl_s13b_b90k",
            "image_size": 224,
            "handler_class": "CLIPHandler",
            "image_probs": default_image_probs,
        },
        "clip-vit-h-14": {
            "model_name": "ViT-H-14",
            "pretrained": "laion2b_s32b_b79k",
            "image_size": 224,
            "handler_class": "CLIPHandler",
            "image_probs": default_image_probs,
        },
    },
    "QwenText": {
        # Primary model for this port — 2048-dim output matches EMBEDDING_DIM.
        "qwen3-vl-embedding-2b": {
            "hf_model_id": "Qwen/Qwen3-VL-Embedding-2B",
            "handler_class": "QwenEmbeddingHandler",
            "max_length": 8192,
            "instruction_template": "Instruct: {task_description}\\nQuery:{query}",
            "task_description": "Given a web search query, retrieve relevant passages that answer the query",
            "modalities": ["text"],
            "trust_remote_code": True,
        },
        "qwen3-embedding-0.6b": {
            "hf_model_id": "Qwen/Qwen3-Embedding-0.6B",
            "handler_class": "QwenEmbeddingHandler",
            "max_length": 8192,
            "instruction_template": "Instruct: {task_description}\\nQuery:{query}",
            "task_description": "Given a web search query, retrieve relevant passages that answer the query",
            "modalities": ["text"],
            "trust_remote_code": True,
        },
        "qwen3-embedding-4b": {
            "hf_model_id": "Qwen/Qwen3-Embedding-4B",
            "handler_class": "QwenEmbeddingHandler",
            "max_length": 8192,
            "instruction_template": "Instruct: {task_description}\\nQuery:{query}",
            "task_description": "Given a web search query, retrieve relevant passages that answer the query",
            "modalities": ["text"],
            "trust_remote_code": True,
        },
    },
}


def get_model_config(model_id: str, device=None, use_npu=None, onnx_path=None, rknn_path=None) -> dict:
    """
    Return a resolved model configuration dictionary for the given model identifier with optional runtime overrides.
    
    Parameters:
        model_id (str): Model identifier as "type/name" or just "name"; if only "name" is given, the registry is searched across types.
        device (str, optional): Inference device override (defaults to USE_DEVICE env or "cpu").
        use_npu (bool | str | None, optional): Whether to use NPU; if None, resolved from USE_NPU env (truthy values: "true","1","yes","on").
        onnx_path (str, optional): Path override for CLIP vision ONNX file (defaults to CLIP_ONNX_PATH env or "./models/clip_vision.onnx").
        rknn_path (str, optional): Path override for CLIP vision RKNN file (defaults to CLIP_RKNN_PATH env or "./models/clip_vision.rknn").
    
    Returns:
        dict: A copy of the base model configuration merged with the resolved runtime overrides (includes keys such as "device", "use_npu", "onnx_path", "rknn_path", and "npu_core").
    
    Raises:
        ValueError: If the model cannot be located or if the specified model type or name is unsupported.
    """
    # Handle both "type/name" and "name" formats.
    if "/" in model_id:
        model_type, model_name = model_id.split("/", 1)
    else:
        for model_type, models in MODEL_CONFIGS.items():
            if model_id in models:
                model_name = model_id
                break
        else:
            raise ValueError(f"Model '{model_id}' not found in any model type")

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type '{model_type}' not supported")

    if model_name not in MODEL_CONFIGS[model_type]:
        raise ValueError(f"Model '{model_name}' not found in '{model_type}'")

    config = MODEL_CONFIGS[model_type][model_name].copy()

    # Resolve use_npu from argument or env variable.
    if use_npu is None:
        use_npu = os.getenv("USE_NPU", "false").lower() in ("true", "1", "yes", "on")

    config.update({
        "device": device or os.getenv("USE_DEVICE", "cpu"),
        "use_npu": use_npu,
        "onnx_path": onnx_path or os.getenv("CLIP_ONNX_PATH", "./models/clip_vision.onnx"),
        "rknn_path": rknn_path or os.getenv("CLIP_RKNN_PATH", "./models/clip_vision.rknn"),
        "npu_core": os.getenv("NPU_CORE", "NPU_CORE_0"),
    })

    return config


def list_available_models() -> dict:
    """
    List all available models grouped by type.

    Returns:
        dict mapping model type name to list of model names.
    """
    return {
        model_type: list(models.keys())
        for model_type, models in MODEL_CONFIGS.items()
    }
