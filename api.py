# api.py - FramePack Studio Production API
# A complete REST API that mirrors all Gradio functionality

import os
import sys
import torch
import json
import uuid
import time
import shutil
import base64
import random
import asyncio
import traceback
import requests
from datetime import datetime
from pathlib import PurePath, Path
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator, model_validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import threading
import aiohttp
from collections import defaultdict

# --- 1. ENVIRONMENT SETUP (Mirrors studio.py) ---
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- 2. FRAMEWORK IMPORTS ---
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.memory import DynamicSwapInstaller, gpu, get_cuda_free_memory_gb
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.bucket_tools import find_nearest_bucket
from modules.video_queue import VideoJobQueue, JobStatus
from modules.pipelines.worker import worker
from modules.settings import Settings
from modules import DUMMY_LORA_NAME
from modules.generators import create_model_generator

# --- AI FEATURES IMPORTS (Phase 1) ---
from modules.llm_enhancer import enhance_prompt, unload_enhancing_model
from modules.llm_captioner import caption_image, unload_captioning_model

# --- POST-PROCESSING IMPORTS (Phase 2) ---
from modules.toolbox_app import tb_processor, tb_filter_presets_data

# --- PHASE 3 IMPORTS: Workflow Presets ---
from modules.toolbox_app import (
    tb_workflow_presets_data, 
    TB_WORKFLOW_PRESETS_FILE,
    _get_default_workflow_params,
    _initialize_workflow_presets,
    TB_DEFAULT_FILTER_SETTINGS
)
from modules.toolbox.toolbox_processor import VideoProcessor, DummyProgress
from modules.toolbox.system_monitor import SystemMonitor

# --- PHASE 4 IMPORTS: Filter Presets & Model Management ---
from modules.toolbox_app import (
    TB_BUILT_IN_PRESETS_FILE,
    _initialize_presets as _initialize_filter_presets
)

# --- 3. LOGGING SETUP ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FramePackAPI")

# --- 4. CONFIGURATION ---
# Load API secret from environment variable or use default (for dev only!)
API_SECRET = os.environ.get("FRAMEPACK_API_SECRET", "CHANGE_ME_IN_PRODUCTION")
if API_SECRET == "CHANGE_ME_IN_PRODUCTION":
    logger.warning("⚠️  WARNING: Using default API secret! Set FRAMEPACK_API_SECRET environment variable in production!")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))  # requests per window
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # window in seconds

# Base URL for callbacks - MUST be set to your public RunPod URL!
# Example: https://wt2g7dvyejayg0-8000.proxy.runpod.net
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
if BASE_URL == "http://localhost:8000":
    logger.warning("⚠️  WARNING: BASE_URL is localhost! Set BASE_URL environment variable to your public RunPod URL for callbacks to work!")

# --- 5. GLOBAL VARIABLES (CRITICAL for worker.py) ---
print("--- [API] INITIALIZING FRAMEPACK ENGINE ---")
logger.info("Initializing FramePack Engine...")

# Mock Args (worker checks args.offline)
class Args:
    def __init__(self):
        self.offline = False
        self.share = False
        self.server = '0.0.0.0'
        self.port = 8000
        self.inbrowser = False
        self.lora = None

args = Args()

# Initialize Settings
settings = Settings()
api_output_dir = settings.get("output_dir", './outputs')
os.makedirs(api_output_dir, exist_ok=True)

# Initialize Stream (Used for logging/progress internally)
stream = AsyncStream()

# Cache for prompt embeddings
prompt_embedding_cache = {}

# Check Hardware
free_mem_gb = get_cuda_free_memory_gb(gpu)
# Allow forcing low VRAM mode even on high VRAM GPUs (helps with very long videos)
force_low_vram = os.environ.get("FORCE_LOW_VRAM", "false").lower() == "true"
high_vram = (free_mem_gb > 60) and not force_low_vram
if force_low_vram:
    logger.info(f'Free VRAM: {free_mem_gb:.2f} GB | FORCED Low-VRAM Mode (enables VAE slicing/tiling)')
else:
    logger.info(f'Free VRAM: {free_mem_gb:.2f} GB | High-VRAM Mode: {high_vram}')

# --- 6. MODEL LOADING (Mirrors studio.py) ---
logger.info("Loading Core Models...")

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

# Configure Models
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)

# --- 7. LORA SCANNING ---
logger.info("Scanning LoRAs...")
lora_names = []
lora_dir = settings.get("lora_dir", os.path.join(os.path.dirname(__file__), 'loras'))
os.makedirs(lora_dir, exist_ok=True)

if os.path.isdir(lora_dir):
    for root, _, files in os.walk(lora_dir):
        for file in files:
            if file.endswith('.safetensors') or file.endswith('.pt'):
                lora_relative_path = os.path.relpath(os.path.join(root, file), lora_dir)
                lora_name = str(PurePath(lora_relative_path).with_suffix(''))
                lora_names.append(lora_name)
    if len(lora_names) == 1:
        lora_names.append(DUMMY_LORA_NAME)
    logger.info(f"Found LoRAs: {lora_names}")
else:
    logger.warning(f"LoRA directory not found at {lora_dir}")

# --- 8. GENERATOR INITIALIZATION ---
logger.info("Initializing Default Generator (Original)...")
current_generator = create_model_generator(
    "Original",
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    vae=vae,
    image_encoder=image_encoder,
    feature_extractor=feature_extractor,
    high_vram=high_vram,
    prompt_embedding_cache=prompt_embedding_cache,
    offline=False,
    settings=settings
)

# --- 9. QUEUE SETUP ---
job_queue = VideoJobQueue()
job_queue.set_worker_function(worker)

# --- 10. PROGRESS TRACKING ---
# Store progress updates for each job (for polling and WebSocket)
job_progress_store: Dict[str, Dict[str, Any]] = {}
progress_lock = threading.Lock()

# WebSocket connections for progress streaming
websocket_connections: Dict[str, List[WebSocket]] = defaultdict(list)

def update_job_progress(job_id: str, progress_data: Dict[str, Any]):
    """Update progress for a job and notify WebSocket clients"""
    with progress_lock:
        job_progress_store[job_id] = {
            **progress_data,
            "updated_at": time.time()
        }

# --- 11. RATE LIMITING ---
rate_limit_store: Dict[str, List[float]] = defaultdict(list)
rate_limit_lock = threading.Lock()

def check_rate_limit(api_key: str) -> bool:
    """Check if the API key has exceeded rate limits"""
    current_time = time.time()
    with rate_limit_lock:
        # Clean old entries
        rate_limit_store[api_key] = [
            t for t in rate_limit_store[api_key] 
            if current_time - t < RATE_LIMIT_WINDOW
        ]
        # Check limit
        if len(rate_limit_store[api_key]) >= RATE_LIMIT_REQUESTS:
            return False
        # Add new request
        rate_limit_store[api_key].append(current_time)
        return True

# --- 12. ENUMS AND MODELS ---

class ModelType(str, Enum):
    ORIGINAL = "Original"
    ORIGINAL_WITH_ENDFRAME = "Original with Endframe"
    F1 = "F1"
    VIDEO = "Video"
    VIDEO_F1 = "Video F1"

class LatentType(str, Enum):
    BLACK = "Black"
    WHITE = "White"
    NOISE = "Noise"
    GREEN_SCREEN = "Green Screen"

class CacheType(str, Enum):
    NONE = "None"
    TEACACHE = "TeaCache"
    MAGCACHE = "MagCache"


# --- 12.5 AI FEATURES REQUEST/RESPONSE MODELS (Phase 1) ---

class EnhancePromptRequest(BaseModel):
    """Request model for prompt enhancement"""
    prompt: str = Field(
        ...,
        description="The prompt to enhance. Can be simple text or timestamped format like '[1s: text] [3s: more text]'",
        min_length=1,
        max_length=10000
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when enhancement completes (webhook)"
    )

class EnhancePromptResponse(BaseModel):
    """Response model for prompt enhancement"""
    success: bool
    original_prompt: str
    enhanced_prompt: str
    message: str

class CaptionImageRequest(BaseModel):
    """Request model for image captioning"""
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image to caption"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL to download image from for captioning"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when captioning completes (webhook)"
    )
    
    @validator('image_base64', 'image_url', pre=True, always=True)
    def check_at_least_one_image_source(cls, v, values):
        # This validator runs for each field
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.image_base64 and not self.image_url:
            raise ValueError("Either image_base64 or image_url must be provided")

class CaptionImageResponse(BaseModel):
    """Response model for image captioning"""
    success: bool
    caption: str
    message: str

class UnloadModelRequest(BaseModel):
    """Request model for unloading AI models"""
    model: str = Field(
        ...,
        description="Model to unload: 'enhancer', 'captioner', or 'all'"
    )

class UnloadModelResponse(BaseModel):
    """Response model for unloading AI models"""
    success: bool
    message: str
    models_unloaded: List[str]


# --- 12.6 POST-PROCESSING REQUEST/RESPONSE MODELS (Phase 2) ---

class FPSMode(str, Enum):
    """Frame interpolation modes"""
    NO_INTERPOLATION = "No Interpolation"
    TWO_X = "2x Frames"
    FOUR_X = "4x Frames"

class LoopType(str, Enum):
    """Video loop types"""
    LOOP = "loop"
    PING_PONG = "ping-pong"

class ExportFormat(str, Enum):
    """Video export formats"""
    MP4 = "MP4"
    WEBM = "WebM"
    GIF = "GIF"

class UpscaleModel(str, Enum):
    """Available ESRGAN upscale models"""
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRNET_X4PLUS = "RealESRNet_x4plus"
    REALESR_GENERAL_X4V3 = "RealESR-general-x4v3"
    REALESRGAN_X4PLUS_ANIME_6B = "RealESRGAN_x4plus_anime_6B"
    REALESR_ANIMEVIDEO_V3 = "RealESR_AnimeVideo_v3"

# --- Video Analysis ---
class AnalyzeVideoRequest(BaseModel):
    """Request model for video analysis"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file to analyze")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    
    @model_validator(mode='after')
    def check_video_source(self):
        if not self.video_url and not self.video_base64:
            raise ValueError('Either video_url or video_base64 must be provided')
        return self

class AnalyzeVideoResponse(BaseModel):
    """Response model for video analysis"""
    success: bool
    analysis: Optional[str] = None
    file_size: Optional[str] = None
    duration: Optional[str] = None
    fps: Optional[str] = None
    resolution: Optional[str] = None
    frame_count: Optional[str] = None
    has_audio: Optional[str] = None
    message: str

# --- Upscale Video ---
class UpscaleVideoRequest(BaseModel):
    """Request model for video upscaling"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file to upscale")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    model: str = Field(
        default="RealESRGAN_x4plus",
        description="ESRGAN model to use for upscaling"
    )
    scale_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Target scale factor (1.0-4.0, depends on model)"
    )
    tile_size: int = Field(
        default=512,
        description="Tile size for processing (0=no tiling [risky], 256, 512 [recommended]). 512 recommended to prevent OOM."
    )
    enhance_face: bool = Field(
        default=False,
        description="Use GFPGAN to enhance faces"
    )
    denoise_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Denoise strength (only for RealESR-general-x4v3 model)"
    )
    use_streaming: bool = Field(
        default=True,
        description="Use streaming mode for low memory processing of large videos. Recommended for API use."
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Frame Interpolation ---
class InterpolateVideoRequest(BaseModel):
    """Request model for frame interpolation (RIFE)"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file to interpolate")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    fps_mode: str = Field(
        default="2x",
        description="Frame interpolation mode: '2x' or '4x'"
    )
    speed_factor: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed adjustment factor (0.25=4x slower, 4.0=4x faster)"
    )
    use_streaming: bool = Field(
        default=True,
        description="Use streaming mode for low memory processing. Recommended for API use."
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Video Filters ---
class VideoFiltersRequest(BaseModel):
    """Request model for applying video filters"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file to process")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    brightness: int = Field(default=0, ge=-100, le=100, description="Brightness adjustment (-100 to 100)")
    contrast: float = Field(default=1.0, ge=0.0, le=3.0, description="Contrast multiplier (0-3)")
    saturation: float = Field(default=1.0, ge=0.0, le=3.0, description="Saturation multiplier (0-3)")
    temperature: int = Field(default=0, ge=-100, le=100, description="Color temperature (-100=cool to 100=warm)")
    sharpen: float = Field(default=0.0, ge=0.0, le=5.0, description="Sharpen strength (0-5)")
    blur: float = Field(default=0.0, ge=0.0, le=5.0, description="Blur strength (0-5)")
    denoise: float = Field(default=0.0, ge=0.0, le=10.0, description="Denoise strength (0-10)")
    vignette: int = Field(default=0, ge=0, le=100, description="Vignette strength (0-100)")
    s_curve_contrast: int = Field(default=0, ge=0, le=100, description="S-curve contrast (0-100)")
    film_grain: int = Field(default=0, ge=0, le=50, description="Film grain strength (0-50)")
    # Optional preset support
    preset: Optional[str] = Field(default=None, description="Filter preset name (cinematic, vintage, cool, warm, dramatic)")
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Video Loop ---
class VideoLoopRequest(BaseModel):
    """Request model for creating video loops"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file to loop")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    loop_type: str = Field(
        default="loop",
        description="Loop type: 'loop' (repeat) or 'ping-pong' (forward-backward)"
    )
    num_loops: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of additional loops/repeats (1 = plays twice total)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Join Videos ---
class JoinVideosRequest(BaseModel):
    """Request model for joining/concatenating videos"""
    video_urls: List[str] = Field(
        ...,
        min_items=2,
        description="List of video URLs to join (minimum 2)"
    )
    output_name: Optional[str] = Field(
        default=None,
        description="Custom output filename (optional)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Export/Compress Video ---
class ExportVideoRequest(BaseModel):
    """Request model for exporting/compressing video"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file to export")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    format: str = Field(
        default="MP4",
        description="Output format: MP4, WebM, or GIF"
    )
    quality: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Quality (0-100, higher = better quality, larger file)"
    )
    max_width: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Maximum output width in pixels (maintains aspect ratio)"
    )
    output_name: Optional[str] = Field(
        default=None,
        description="Custom output filename (optional)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Pipeline Operation ---
class PipelineOperation(BaseModel):
    """Single operation in a processing pipeline"""
    type: str = Field(
        ...,
        description="Operation type: 'upscale', 'interpolate', 'filters', 'loop', 'export'"
    )
    params: Dict[str, Any] = Field(
        default={},
        description="Operation-specific parameters"
    )

class PipelineRequest(BaseModel):
    """Request model for running a processing pipeline"""
    video_url: Optional[str] = Field(default=None, description="URL to the input video")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    operations: List[PipelineOperation] = Field(
        ...,
        min_items=1,
        description="List of operations to perform in order"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

# --- Generic Post-Processing Response ---
class PostProcessResponse(BaseModel):
    """Generic response model for post-processing operations"""
    success: bool
    output_url: Optional[str] = None
    output_filename: Optional[str] = None
    message: str
    processing_time_seconds: Optional[float] = None


# --- Specific Post-Processing Response Models ---
class UpscaleVideoResponse(BaseModel):
    """Response model for video upscaling"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

class InterpolateVideoResponse(BaseModel):
    """Response model for video interpolation"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

class VideoFiltersResponse(BaseModel):
    """Response model for video filters"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

class VideoLoopResponse(BaseModel):
    """Response model for video loop"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

class JoinVideosResponse(BaseModel):
    """Response model for video join"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

class ExportVideoResponse(BaseModel):
    """Response model for video export"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

class PipelineResponse(BaseModel):
    """Response model for pipeline processing"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None
    operations_completed: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# PHASE 3: FRAMES STUDIO, WORKFLOW PRESETS, SYSTEM UTILITIES
# ============================================================================

# --- Frames Studio Request/Response Models ---
class ExtractFramesRequest(BaseModel):
    """Request model for extracting frames from a video"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    extraction_rate: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Extract every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

class ExtractFramesResponse(BaseModel):
    """Response model for frame extraction"""
    success: bool
    message: str
    folder_name: Optional[str] = None
    frame_count: Optional[int] = None
    frames_path: Optional[str] = None

class FrameFoldersResponse(BaseModel):
    """Response model for listing frame folders"""
    success: bool
    folders: List[str]
    message: str

class FrameInfo(BaseModel):
    """Information about a single frame"""
    filename: str
    path: str
    index: int

class ListFramesResponse(BaseModel):
    """Response model for listing frames in a folder"""
    success: bool
    folder: str
    frames: List[FrameInfo]
    total_count: int
    message: str

class DeleteFolderResponse(BaseModel):
    """Response model for deleting a frame folder"""
    success: bool
    folder: str
    message: str

class DeleteFrameResponse(BaseModel):
    """Response model for deleting a single frame"""
    success: bool
    folder: str
    frame: str
    message: str

class SaveFrameResponse(BaseModel):
    """Response model for saving a single frame"""
    success: bool
    message: str
    saved_path: Optional[str] = None
    frame_base64: Optional[str] = None

class GetFrameResponse(BaseModel):
    """Response model for getting a single frame with base64 image data"""
    success: bool
    folder: str
    filename: str
    size: int
    base64: str
    mime_type: str = "image/png"

class ReassembleFramesRequest(BaseModel):
    """Request model for reassembling frames into a video"""
    folder_name: str = Field(..., description="Name of the folder containing extracted frames")
    output_fps: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Output video frame rate"
    )
    output_name: Optional[str] = Field(
        default=None,
        description="Custom output video name (optional)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

class ReassembleFramesResponse(BaseModel):
    """Response model for frame reassembly"""
    success: bool
    message: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None

# --- Workflow Preset Request/Response Models ---
class WorkflowPresetParams(BaseModel):
    """Parameters for a workflow preset"""
    # Upscale parameters
    upscale_model: Optional[str] = Field(default="RealESRGAN_x2plus")
    upscale_factor: Optional[float] = Field(default=2.0)
    tile_size: Optional[int] = Field(default=0)
    enhance_face: Optional[bool] = Field(default=False)
    denoise_strength: Optional[float] = Field(default=0.5)
    upscale_use_streaming: Optional[bool] = Field(default=False)
    
    # Frame adjust parameters
    fps_mode: Optional[str] = Field(default="No Interpolation")
    speed_factor: Optional[float] = Field(default=1.0)
    frames_use_streaming: Optional[bool] = Field(default=False)
    
    # Loop parameters
    loop_type: Optional[str] = Field(default="loop")
    num_loops: Optional[int] = Field(default=1)
    
    # Filter parameters
    brightness: Optional[float] = Field(default=0.0)
    contrast: Optional[float] = Field(default=1.0)
    saturation: Optional[float] = Field(default=1.0)
    temperature: Optional[float] = Field(default=0.0)
    sharpen: Optional[float] = Field(default=0.0)
    blur: Optional[float] = Field(default=0.0)
    denoise: Optional[float] = Field(default=0.0)
    vignette: Optional[float] = Field(default=0.0)
    s_curve_contrast: Optional[float] = Field(default=0.0)
    film_grain_strength: Optional[float] = Field(default=0.0)
    
    # Export parameters
    export_format: Optional[str] = Field(default="MP4")
    export_quality: Optional[int] = Field(default=85)
    export_max_width: Optional[int] = Field(default=1024)

class WorkflowPresetData(BaseModel):
    """Complete workflow preset data"""
    active_steps: List[str] = Field(
        default=[],
        description="List of active pipeline steps: 'upscale', 'interpolate', 'loop', 'filters', 'export'"
    )
    params: WorkflowPresetParams = Field(default_factory=WorkflowPresetParams)

class SaveWorkflowPresetRequest(BaseModel):
    """Request model for saving a workflow preset"""
    name: str = Field(..., min_length=1, description="Name for the workflow preset")
    preset_data: WorkflowPresetData

class WorkflowPresetResponse(BaseModel):
    """Response model for workflow preset operations"""
    success: bool
    message: str
    preset_name: Optional[str] = None

class ListWorkflowPresetsResponse(BaseModel):
    """Response model for listing workflow presets"""
    success: bool
    presets: Dict[str, Any]
    message: str

# --- System Utility Request/Response Models ---
class ClearTempResponse(BaseModel):
    """Response model for clearing temporary files"""
    success: bool
    message: str
    files_deleted: Optional[int] = None
    space_freed_mb: Optional[float] = None

class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    success: bool
    message: str
    ram: Optional[Dict[str, Any]] = None
    vram: Optional[Dict[str, Any]] = None
    gpu: Optional[Dict[str, Any]] = None


# ============================================================================
# PHASE 4: FILTER PRESETS, BATCH PROCESSING, MODEL MANAGEMENT
# ============================================================================

# --- Filter Preset Request/Response Models ---
class FilterPresetSettings(BaseModel):
    """Filter settings for a preset"""
    brightness: float = Field(default=0.0)
    contrast: float = Field(default=1.0)
    saturation: float = Field(default=1.0)
    temperature: float = Field(default=0.0)
    sharpen: float = Field(default=0.0)
    blur: float = Field(default=0.0)
    denoise: float = Field(default=0.0)
    vignette: float = Field(default=0.0)
    s_curve_contrast: float = Field(default=0.0)
    film_grain_strength: float = Field(default=0.0)

class SaveFilterPresetRequest(BaseModel):
    """Request model for saving a filter preset"""
    name: str = Field(..., min_length=1, description="Name for the filter preset")
    settings: FilterPresetSettings

class FilterPresetResponse(BaseModel):
    """Response model for filter preset operations"""
    success: bool
    message: str
    preset_name: Optional[str] = None

class ListFilterPresetsResponse(BaseModel):
    """Response model for listing filter presets"""
    success: bool
    presets: Dict[str, Any]
    message: str

# --- Batch Processing Request/Response Models ---
class BatchVideoItem(BaseModel):
    """A single video in a batch"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")

class BatchProcessingRequest(BaseModel):
    """Request model for batch processing multiple videos"""
    videos: List[BatchVideoItem] = Field(
        ...,
        min_items=1,
        description="List of videos to process (at least 1)"
    )
    operations: List[PipelineOperation] = Field(
        ...,
        min_items=1,
        description="Pipeline operations to apply to each video"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results to when processing completes (webhook)"
    )

class BatchVideoResult(BaseModel):
    """Result for a single video in batch processing"""
    index: int
    success: bool
    input_source: str
    output_video_base64: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None

class BatchProcessingResponse(BaseModel):
    """Response model for batch processing"""
    success: bool
    message: str
    total_videos: int
    successful: int
    failed: int
    results: List[BatchVideoResult]

# --- Video Save Request/Response Models ---
class SaveVideoRequest(BaseModel):
    """Request model for saving a video to permanent storage"""
    video_url: Optional[str] = Field(default=None, description="URL to the video file")
    video_base64: Optional[str] = Field(default=None, description="Base64-encoded video data")
    custom_name: Optional[str] = Field(default=None, description="Custom filename (optional)")

class SaveVideoResponse(BaseModel):
    """Response model for saving a video"""
    success: bool
    message: str
    saved_path: Optional[str] = None
    filename: Optional[str] = None

# --- Autosave Mode Request/Response Models ---
class AutosaveSettingRequest(BaseModel):
    """Request model for setting autosave mode"""
    enabled: bool = Field(..., description="Whether autosave should be enabled")

class AutosaveSettingResponse(BaseModel):
    """Response model for autosave mode"""
    success: bool
    message: str
    autosave_enabled: bool

# --- Model Unload Request/Response Models ---
class UnloadMainModelResponse(BaseModel):
    """Response model for unloading the main generation model"""
    success: bool
    message: str
    model_unloaded: Optional[str] = None
    vram_freed_estimate: Optional[str] = None


# --- 13. REQUEST/RESPONSE MODELS ---

class GenerationRequest(BaseModel):
    """Complete generation request matching all Gradio parameters"""
    
    # === CORE PARAMETERS ===
    prompt: str = Field(..., description="The main prompt describing what to generate")
    negative_prompt: str = Field(
        default="low quality, worst quality, deformed, distorted, disfigured, blurry, bad anatomy",
        description="Negative prompt to avoid certain qualities"
    )
    
    # === MODEL SELECTION ===
    model_type: ModelType = Field(
        default=ModelType.ORIGINAL,
        description="Model type: Original, Original with Endframe, F1, Video, or Video F1"
    )
    
    # === IMAGE INPUT (for I2V) ===
    input_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded input image for Image-to-Video generation"
    )
    input_image_url: Optional[str] = Field(
        default=None,
        description="URL to download input image from"
    )
    
    # === END FRAME (for Endframe models) ===
    end_frame_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded end frame image"
    )
    end_frame_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strength of end frame influence (0.0-1.0)"
    )
    
    # === GENERATION SPECS ===
    seed: int = Field(
        default=-1,
        ge=-1,
        description="Random seed (-1 for random)"
    )
    steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of diffusion steps"
    )
    total_second_length: float = Field(
        default=6.0,
        ge=1.0,
        le=120.0,
        description="Total video length in seconds"
    )
    
    # === RESOLUTION ===
    resolution_w: int = Field(
        default=640,
        ge=128,
        le=1920,
        description="Video width (will be adjusted to nearest bucket)"
    )
    resolution_h: int = Field(
        default=640,
        ge=128,
        le=1920,
        description="Video height (will be adjusted to nearest bucket)"
    )
    
    # === CFG SETTINGS ===
    cfg: float = Field(
        default=1.0,
        ge=1.0,
        le=3.0,
        description="CFG Scale (1.0 for distilled, higher for more prompt following but slower)"
    )
    gs: float = Field(
        default=10.0,
        ge=1.0,
        le=32.0,
        description="Distilled CFG Scale"
    )
    rs: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="CFG Re-Scale"
    )
    
    # === LATENT SETTINGS ===
    latent_type: LatentType = Field(
        default=LatentType.BLACK,
        description="Latent background type for T2V (Black, White, Noise, Green Screen)"
    )
    latent_window_size: int = Field(
        default=9,
        ge=1,
        le=33,
        description="Latent window size (advanced, change at own risk)"
    )
    
    # === CACHE SETTINGS ===
    cache_type: CacheType = Field(
        default=CacheType.MAGCACHE,
        description="Caching strategy: None, TeaCache, or MagCache"
    )
    
    # TeaCache settings
    teacache_num_steps: int = Field(
        default=25,
        ge=1,
        le=50,
        description="TeaCache: Number of intermediate sections to keep"
    )
    teacache_rel_l1_thresh: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description="TeaCache: Relative L1 threshold (lower = faster)"
    )
    
    # MagCache settings
    magcache_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="MagCache: Error tolerance threshold (lower = faster)"
    )
    magcache_max_consecutive_skips: int = Field(
        default=2,
        ge=1,
        le=5,
        description="MagCache: Max consecutive estimated steps (higher = faster)"
    )
    magcache_retention_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="MagCache: Retention ratio (lower = faster)"
    )
    
    # === PROMPT BLENDING ===
    blend_sections: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Number of sections for prompt blending transitions"
    )
    
    # === LORA SETTINGS ===
    loras: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dictionary of LoRA names to weights, e.g. {'my_lora': 0.8}"
    )
    
    # === VIDEO MODEL SETTINGS ===
    input_video_url: Optional[str] = Field(
        default=None,
        description="URL to input video (for Video model)"
    )
    combine_with_source: bool = Field(
        default=False,
        description="Combine generated video with source video"
    )
    num_cleaned_frames: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of cleaned frames for Video model"
    )
    
    # === METADATA ===
    save_metadata: bool = Field(
        default=True,
        description="Save generation metadata to JSON"
    )
    
    # === CALLBACK/WEBHOOK SETTINGS ===
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST job completion data to (webhook). Will receive job_id, status, result_url, video_download_url, and metadata."
    )
    callback_token: Optional[str] = Field(
        default=None,
        description="Optional bearer token to include in callback Authorization header"
    )
    
    # === VALIDATORS ===
    @validator('resolution_w', 'resolution_h')
    def validate_resolution(cls, v):
        if v % 8 != 0:
            # Round to nearest multiple of 8
            v = (v // 8) * 8
        return max(128, min(1920, v))
    
    @validator('seed')
    def validate_seed(cls, v):
        if v == -1:
            return random.randint(0, 2**31 - 1)
        return v


class GenerationResponse(BaseModel):
    """Response after submitting a generation job"""
    job_id: str
    status: str
    seed: int
    resolution: Dict[str, int]
    estimated_time_seconds: Optional[float] = None
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str
    status: str
    progress_percent: Optional[float] = None
    progress_desc: Optional[str] = None
    eta_seconds: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    vram_free_gb: float
    high_vram_mode: bool
    queue_length: int
    available_loras: List[str]
    available_models: List[str]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


# --- 14. HELPER FUNCTIONS ---

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


async def download_image_from_url(url: str) -> np.ndarray:
    """Download image from URL and return as numpy array"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status}")
                
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                return np.array(image)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=400, detail="Image download timeout")
    except Exception as e:
        logger.error(f"Failed to download image from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")


def download_file_from_url(url: str, allowed_extensions: list = None) -> str:
    """Download a file from URL and return the local file path.
    
    If the URL points to a local /outputs/ path on this server, the file is
    read directly from disk to avoid self-referential HTTP requests that can
    cause deadlocks or timeouts.
    
    Args:
        url: The URL to download from
        allowed_extensions: List of allowed file extensions (e.g. ['.mp4', '.webm'])
        
    Returns:
        Path to the downloaded file
    """
    from urllib.parse import urlparse, unquote
    
    logger.info(f"[DOWNLOAD] Downloading file from URL: {url[:100]}...")
    
    try:
        # Parse URL to get filename and path
        parsed_url = urlparse(url)
        url_path = unquote(parsed_url.path)
        
        # =====================================================================
        # CHECK FOR LOCAL /outputs/ FILE - Avoid self-referential HTTP requests
        # =====================================================================
        # If the URL points to our own /outputs/ directory, read directly from disk
        # This prevents deadlocks when the API tries to download from itself
        if '/outputs/' in url_path:
            # Extract the relative path after /outputs/
            outputs_index = url_path.find('/outputs/')
            relative_path = url_path[outputs_index + len('/outputs/'):]
            local_path = os.path.join(api_output_dir, relative_path)
            
            # Normalize the path to prevent directory traversal attacks
            local_path = os.path.normpath(local_path)
            normalized_output_dir = os.path.normpath(api_output_dir)
            
            # Security check: ensure the path is within the outputs directory
            if local_path.startswith(normalized_output_dir) and os.path.exists(local_path):
                logger.info(f"[DOWNLOAD] Detected local outputs file, reading directly from disk: {local_path}")
                
                # Validate extension if needed
                _, ext = os.path.splitext(local_path)
                if allowed_extensions and ext.lower() not in [e.lower() for e in allowed_extensions]:
                    logger.warning(f"[DOWNLOAD] Local file extension {ext} not in allowed list")
                    raise HTTPException(status_code=400, detail=f"File extension {ext} not allowed")
                
                file_size = os.path.getsize(local_path)
                logger.info(f"[DOWNLOAD] Local file found: {local_path} ({file_size} bytes)")
                
                # Return the local path directly - no need to copy
                return local_path
            else:
                logger.info(f"[DOWNLOAD] Path {local_path} not found locally or outside outputs dir, will try HTTP download")
        
        # =====================================================================
        # STANDARD HTTP DOWNLOAD for external URLs
        # =====================================================================
        filename = os.path.basename(url_path) or f"downloaded_{uuid.uuid4().hex[:8]}"
        
        # Get extension
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = '.mp4'  # Default extension
        
        # Validate extension if list provided
        if allowed_extensions and ext.lower() not in [e.lower() for e in allowed_extensions]:
            logger.warning(f"[DOWNLOAD] Extension {ext} not in allowed list, defaulting to .mp4")
            ext = '.mp4'
        
        # Create temp file path
        temp_path = os.path.join(api_output_dir, f"download_{uuid.uuid4().hex[:8]}{ext}")
        
        # Download the file
        logger.info(f"[DOWNLOAD] Starting HTTP download from external URL...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Write to file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"[DOWNLOAD] File downloaded successfully: {temp_path} ({file_size} bytes)")
        
        return temp_path
        
    except requests.exceptions.Timeout:
        logger.error("[DOWNLOAD] Request timed out")
        raise HTTPException(status_code=408, detail="File download timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"[DOWNLOAD] Request error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"[DOWNLOAD] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


# ============================================================================
# BACKGROUND TASK INFRASTRUCTURE FOR LONG-RUNNING POST-PROCESSING
# ============================================================================
# This allows endpoints to return immediately when callback_url is provided,
# preventing Cloudflare/proxy timeouts (Error 524) during long operations.

import concurrent.futures
from functools import partial

# Thread pool for background post-processing tasks
_postprocess_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="postprocess_bg")

def run_postprocess_in_background(
    operation_name: str,
    callback_url: str,
    process_func,
    cleanup_func=None,
    **kwargs
):
    """
    Run a post-processing operation in a background thread.
    Sends callback when complete (success or failure).
    
    Args:
        operation_name: Name of the operation (upscale, interpolate, etc.)
        callback_url: URL to send results to
        process_func: The function to run (should return output_path or None)
        cleanup_func: Optional cleanup function to run after processing
        **kwargs: Arguments to pass to process_func (metadata is extracted and not passed)
    """
    # Extract metadata before passing to process_func (it's for the callback, not the processor)
    callback_metadata = kwargs.pop('metadata', {})
    # Copy kwargs for the background thread (to avoid closure issues)
    process_kwargs = dict(kwargs)
    
    def background_task():
        output_path = None
        output_base64 = None
        error_msg = None
        
        try:
            logger.info(f"[BG-TASK] Starting background {operation_name} processing...")
            
            # Run the actual processing (metadata is NOT passed here)
            output_path = process_func(**process_kwargs)
            
            if output_path and os.path.exists(output_path):
                logger.info(f"[BG-TASK] {operation_name} complete: {output_path}")
                
                # Read and encode output
                with open(output_path, 'rb') as f:
                    output_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                # Send success callback
                asyncio.run(send_postprocess_callback(
                    callback_url=callback_url,
                    operation_type=operation_name,
                    success=True,
                    output_path=output_path,
                    output_base64=output_base64,
                    metadata=callback_metadata
                ))
            else:
                error_msg = f"{operation_name} failed - no output produced"
                logger.error(f"[BG-TASK] {error_msg}")
                
                # Send failure callback
                asyncio.run(send_postprocess_callback(
                    callback_url=callback_url,
                    operation_type=operation_name,
                    success=False,
                    error=error_msg
                ))
                
        except Exception as e:
            error_msg = f"{operation_name} failed: {str(e)}"
            logger.error(f"[BG-TASK] {error_msg}")
            logger.error(f"[BG-TASK] Traceback: {traceback.format_exc()}")
            
            # Send failure callback
            try:
                asyncio.run(send_postprocess_callback(
                    callback_url=callback_url,
                    operation_type=operation_name,
                    success=False,
                    error=error_msg
                ))
            except Exception as cb_err:
                logger.error(f"[BG-TASK] Failed to send error callback: {cb_err}")
        
        finally:
            # Run cleanup if provided
            if cleanup_func:
                try:
                    cleanup_func()
                except Exception as clean_err:
                    logger.warning(f"[BG-TASK] Cleanup failed: {clean_err}")
    
    # Submit to thread pool
    _postprocess_executor.submit(background_task)
    logger.info(f"[BG-TASK] {operation_name} task submitted to background executor")


class AsyncProcessingResponse(BaseModel):
    """Response when processing is started in background"""
    success: bool = True
    message: str
    processing: bool = True
    callback_url: str
    note: str = "Results will be sent to your callback URL when processing completes"


def apply_bucket_resolution(width: int, height: int) -> tuple:
    """Apply bucket resolution system like Gradio does.
    
    Note: find_nearest_bucket expects (h, w) and returns (bucket_h, bucket_w).
    We pass (height, width) and use average of both as the resolution parameter,
    matching how Gradio handles it when no input image is provided.
    """
    try:
        # Use average of width and height as resolution, matching Gradio's behavior
        resolution = (width + height) // 2
        # find_nearest_bucket expects (h, w) order and returns (bucket_h, bucket_w)
        bucket_h, bucket_w = find_nearest_bucket(height, width, resolution=resolution)
        logger.info(f"Resolution adjusted: {width}x{height} -> {bucket_w}x{bucket_h} (using resolution bucket {resolution})")
        return bucket_w, bucket_h
    except Exception as e:
        logger.warning(f"Bucket resolution failed, using original: {e}")
        return width, height


def create_latent_image(width: int, height: int, latent_type: LatentType) -> np.ndarray:
    """Create a latent background image based on type"""
    if latent_type == LatentType.WHITE:
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    elif latent_type == LatentType.NOISE:
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif latent_type == LatentType.GREEN_SCREEN:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 1] = 177  # Green channel
        img[:, :, 2] = 64   # Blue channel (in RGB it's actually the blue)
        return img
    else:  # BLACK
        return np.zeros((height, width, 3), dtype=np.uint8)


def estimate_generation_time(req: GenerationRequest) -> float:
    """Estimate generation time in seconds based on parameters"""
    # Base time per step (rough estimate for RTX 4090)
    base_time_per_step = 1.5  # seconds
    
    # Adjust for resolution
    pixels = req.resolution_w * req.resolution_h
    resolution_factor = pixels / (640 * 640)
    
    # Adjust for duration
    duration_factor = req.total_second_length / 6.0
    
    # Adjust for CFG (doubles time if > 1)
    cfg_factor = 2.0 if req.cfg > 1.0 else 1.0
    
    # Adjust for cache
    cache_factor = 0.7 if req.cache_type != CacheType.NONE else 1.0
    
    total_time = req.steps * base_time_per_step * resolution_factor * duration_factor * cfg_factor * cache_factor
    
    return round(total_time, 1)


def validate_api_key(x_api_key: str = Header(None)) -> str:
    """Validate API key from header"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key. Provide X-API-Key header.")
    
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not check_rate_limit(x_api_key):
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    
    return x_api_key


# --- 15. API DEFINITION ---

app = FastAPI(
    title="FramePack Studio API",
    description="Production API for FramePack Studio video generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve outputs statically
app.mount("/outputs", StaticFiles(directory=api_output_dir), name="outputs")


# --- 16. API ENDPOINTS ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint - no authentication required.
    Returns system status and available resources.
    """
    current_free_mem = get_cuda_free_memory_gb(gpu)
    
    # Count queue
    jobs = job_queue.get_all_jobs()
    pending_count = sum(1 for j in jobs if j.status == JobStatus.PENDING)
    
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        vram_free_gb=round(current_free_mem, 2),
        high_vram_mode=high_vram,
        queue_length=pending_count,
        available_loras=[l for l in lora_names if l != DUMMY_LORA_NAME],
        available_models=[m.value for m in ModelType]
    )


@app.get("/loras", tags=["System"])
async def list_loras(x_api_key: str = Header(None)):
    """List available LoRAs"""
    validate_api_key(x_api_key)
    return {
        "loras": [l for l in lora_names if l != DUMMY_LORA_NAME],
        "lora_dir": lora_dir
    }


# --- 16.5 AI FEATURES ENDPOINTS (Phase 1) ---

@app.post("/enhance-prompt", response_model=EnhancePromptResponse, tags=["AI Features"])
async def enhance_prompt_endpoint(req: EnhancePromptRequest, x_api_key: str = Header(None)):
    """
    Enhance a simple prompt into a detailed video generation prompt using AI.
    
    Uses the IBM Granite 3.3-2b-instruct model to expand and improve prompts
    with better visual descriptions, motion details, and camera work.
    
    Supports both simple prompts and timestamped formats like:
    - Simple: "A cat playing"
    - Timestamped: "[1s: A cat sits] [3s: The cat jumps] [5s: The cat runs]"
    
    Note: First call may take longer as the model loads into memory.
    
    Optionally provide callback_url to receive results via webhook.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[AI] Enhance prompt request received. Prompt length: {len(req.prompt)} chars")
    logger.info(f"[AI] Original prompt preview: {req.prompt[:100]}...")
    
    try:
        # Call the enhance_prompt function from llm_enhancer.py
        enhanced = enhance_prompt(req.prompt)
        
        logger.info(f"[AI] Prompt enhanced successfully. Enhanced length: {len(enhanced)} chars")
        logger.info(f"[AI] Enhanced prompt preview: {enhanced[:100]}...")
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation_type="enhance_prompt",
                success=True,
                metadata={
                    "original_prompt": req.prompt,
                    "enhanced_prompt": enhanced,
                    "original_length": len(req.prompt),
                    "enhanced_length": len(enhanced)
                }
            )
        
        return EnhancePromptResponse(
            success=True,
            original_prompt=req.prompt,
            enhanced_prompt=enhanced,
            message="Prompt enhanced successfully"
        )
        
    except Exception as e:
        error_msg = f"Failed to enhance prompt: {str(e)}"
        logger.error(f"[AI] {error_msg}")
        logger.error(f"[AI] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation_type="enhance_prompt",
                success=False,
                error=error_msg,
                metadata={"original_prompt": req.prompt}
            )
        
        # Return original prompt on failure with error message
        return EnhancePromptResponse(
            success=False,
            original_prompt=req.prompt,
            enhanced_prompt=req.prompt,  # Return original on failure
            message=error_msg
        )


@app.post("/caption-image", response_model=CaptionImageResponse, tags=["AI Features"])
async def caption_image_endpoint(req: CaptionImageRequest, x_api_key: str = Header(None)):
    """
    Generate a detailed caption/description from an image using AI.
    
    Uses Microsoft's Florence-2-large model to analyze the image and
    produce a detailed description suitable for use as a video generation prompt.
    
    Provide either:
    - image_base64: Base64 encoded image data
    - image_url: URL to download image from
    
    Note: First call may take longer as the model loads into memory.
    
    Optionally provide callback_url to receive results via webhook.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[AI] Caption image request received")
    
    image_source = "base64" if req.image_base64 else (req.image_url[:50] if req.image_url else "none")
    
    try:
        # Get the image as numpy array
        if req.image_base64:
            logger.info("[AI] Decoding base64 image for captioning...")
            image_np = decode_base64_image(req.image_base64)
            logger.info(f"[AI] Image decoded successfully. Shape: {image_np.shape}")
        elif req.image_url:
            logger.info(f"[AI] Downloading image from URL: {req.image_url[:50]}...")
            image_np = await download_image_from_url(req.image_url)
            logger.info(f"[AI] Image downloaded successfully. Shape: {image_np.shape}")
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either image_base64 or image_url must be provided"
            )
        
        # Call the caption_image function from llm_captioner.py
        logger.info("[AI] Generating caption...")
        caption = caption_image(image_np)
        
        logger.info(f"[AI] Caption generated successfully. Length: {len(caption)} chars")
        logger.info(f"[AI] Caption preview: {caption[:100]}...")
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation_type="caption_image",
                success=True,
                metadata={
                    "caption": caption,
                    "caption_length": len(caption),
                    "image_source": image_source,
                    "image_shape": list(image_np.shape) if image_np is not None else None
                }
            )
        
        return CaptionImageResponse(
            success=True,
            caption=caption,
            message="Image captioned successfully"
        )
        
    except HTTPException:
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation_type="caption_image",
                success=False,
                error="HTTP error during captioning",
                metadata={"image_source": image_source}
            )
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error_msg = f"Failed to caption image: {str(e)}"
        logger.error(f"[AI] {error_msg}")
        logger.error(f"[AI] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation_type="caption_image",
                success=False,
                error=error_msg,
                metadata={"image_source": image_source}
            )
        
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/ai/unload", response_model=UnloadModelResponse, tags=["AI Features"])
async def unload_ai_models(req: UnloadModelRequest, x_api_key: str = Header(None)):
    """
    Unload AI models from memory to free up VRAM/RAM.
    
    Available models to unload:
    - 'enhancer': Unload the prompt enhancement model (Granite 3.3-2b)
    - 'captioner': Unload the image captioning model (Florence-2)
    - 'all': Unload all AI models
    
    Use this when you need to free up memory for video generation
    or when AI features are no longer needed.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[AI] Unload request received for model: {req.model}")
    
    models_unloaded = []
    
    try:
        if req.model.lower() in ['enhancer', 'all']:
            logger.info("[AI] Unloading prompt enhancer model...")
            unload_enhancing_model()
            models_unloaded.append('enhancer')
            logger.info("[AI] Prompt enhancer model unloaded")
        
        if req.model.lower() in ['captioner', 'all']:
            logger.info("[AI] Unloading image captioner model...")
            unload_captioning_model()
            models_unloaded.append('captioner')
            logger.info("[AI] Image captioner model unloaded")
        
        if not models_unloaded:
            return UnloadModelResponse(
                success=False,
                message=f"Unknown model: {req.model}. Use 'enhancer', 'captioner', or 'all'",
                models_unloaded=[]
            )
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return UnloadModelResponse(
            success=True,
            message=f"Successfully unloaded {len(models_unloaded)} model(s)",
            models_unloaded=models_unloaded
        )
        
    except Exception as e:
        error_msg = f"Failed to unload models: {str(e)}"
        logger.error(f"[AI] {error_msg}")
        logger.error(f"[AI] Traceback: {traceback.format_exc()}")
        
        return UnloadModelResponse(
            success=False,
            message=error_msg,
            models_unloaded=models_unloaded  # Return what was unloaded before error
        )


@app.get("/ai/status", tags=["AI Features"])
async def get_ai_status(x_api_key: str = Header(None)):
    """
    Check the status of AI models (whether they are loaded in memory).
    
    Returns information about:
    - enhancer_loaded: Whether the prompt enhancement model is in memory
    - captioner_loaded: Whether the image captioning model is in memory
    """
    validate_api_key(x_api_key)
    
    # Check if models are loaded by inspecting the module globals
    from modules import llm_enhancer, llm_captioner
    
    enhancer_loaded = llm_enhancer.model is not None
    captioner_loaded = llm_captioner.model is not None
    
    return {
        "enhancer": {
            "loaded": enhancer_loaded,
            "model_name": "ibm-granite/granite-3.3-2b-instruct" if enhancer_loaded else None
        },
        "captioner": {
            "loaded": captioner_loaded,
            "model_name": "microsoft/Florence-2-large" if captioner_loaded else None
        }
    }


# ============================================================================
# POST-PROCESSING ENDPOINTS (Toolbox Features)
# ============================================================================

@app.get("/postprocess/models", tags=["Post-Processing"])
async def get_upscale_models(x_api_key: str = Header(None)):
    """
    Get a list of available upscaling models.
    
    Returns all Real-ESRGAN models available for video upscaling with descriptions.
    """
    validate_api_key(x_api_key)
    
    logger.info("[POST-PROCESS] Fetching available upscale models")
    
    models = {
        "RealESRGAN_x2plus": {
            "name": "RealESRGAN x2plus",
            "scale": 2,
            "description": "2x upscaling - Fast, general purpose"
        },
        "RealESRGAN_x4plus": {
            "name": "RealESRGAN x4plus", 
            "scale": 4,
            "description": "4x upscaling - Sharp, detailed output"
        },
        "RealESRNet_x4plus": {
            "name": "RealESRNet x4plus",
            "scale": 4,
            "description": "4x upscaling - Smoother, natural output"
        },
        "RealESR-general-x4v3": {
            "name": "RealESR-general-x4v3",
            "scale": 4,
            "description": "4x upscaling - With denoise support"
        },
        "RealESRGAN_x4plus_anime_6B": {
            "name": "RealESRGAN x4plus Anime 6B",
            "scale": 4,
            "description": "4x upscaling - Optimized for anime"
        },
        "RealESR_AnimeVideo_v3": {
            "name": "RealESR AnimeVideo v3",
            "scale": 4,
            "description": "4x upscaling - Anime video specialized"
        }
    }
    
    return {"models": models}


@app.get("/postprocess/presets", tags=["Post-Processing"])
async def get_filter_presets(x_api_key: str = Header(None)):
    """
    Get a list of available video filter presets.
    
    Returns all predefined filter presets that can be applied to videos.
    """
    validate_api_key(x_api_key)
    
    logger.info("[POST-PROCESS] Fetching available filter presets")
    
    # Return the presets from toolbox_app
    presets = tb_filter_presets_data if tb_filter_presets_data else {}
    
    return {"presets": presets}


@app.post("/postprocess/analyze", response_model=AnalyzeVideoResponse, tags=["Post-Processing"])
async def analyze_video(req: AnalyzeVideoRequest, x_api_key: str = Header(None)):
    """
    Analyze a video file to get its properties.
    
    Returns video metadata including:
    - Duration, FPS, resolution
    - File size, codec information
    - Frame count
    
    Provide either video_base64 or video_url (not both).
    """
    validate_api_key(x_api_key)
    
    logger.info("[POST-PROCESS] Video analysis request received")
    
    # Validate input - must have exactly one source
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for analysis")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    if has_base64 and has_url:
        logger.warning("[POST-PROCESS] Both video sources provided, using video_url")
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL: {req.video_url[:100]}...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"analyze_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        logger.info(f"[POST-PROCESS] Analyzing video: {temp_video_path}")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Call the toolbox processor analyze method
        result = tb_processor.tb_analyze_video_input(temp_video_path)
        
        if result is None:
            logger.error("[POST-PROCESS] Analysis returned no data")
            raise HTTPException(status_code=500, detail="Failed to analyze video")
        
        logger.info(f"[POST-PROCESS] Analysis complete")
        
        # tb_analyze_video_input returns a formatted string with all video info
        return AnalyzeVideoResponse(
            success=True,
            message="Video analyzed successfully",
            analysis=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to analyze video: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp file if we created one from base64
        if temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/upscale", response_model=Union[UpscaleVideoResponse, AsyncProcessingResponse], tags=["Post-Processing"])
async def upscale_video(req: UpscaleVideoRequest, x_api_key: str = Header(None)):
    """
    Upscale a video using Real-ESRGAN models.
    
    Available models:
    - RealESRGAN_x2plus: 2x upscaling (fast, general purpose)
    - RealESRGAN_x4plus: 4x upscaling (sharp output)
    - RealESRNet_x4plus: 4x upscaling (smoother output)
    - RealESR-general-x4v3: 4x with denoise support
    - RealESRGAN_x4plus_anime_6B: 4x for anime
    - RealESR_AnimeVideo_v3: 4x anime video specialized
    
    Provide either video_base64 or video_url (not both).
    
    **ASYNC MODE**: When callback_url is provided, processing runs in background
    and results are sent to your callback URL. This prevents timeout errors for
    long operations.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Upscale request: model={req.model}, scale={req.scale_factor}")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for upscaling")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    # Validate model
    valid_models = ["RealESRGAN_x2plus", "RealESRGAN_x4plus", "RealESRNet_x4plus", 
                    "RealESR-general-x4v3", "RealESRGAN_x4plus_anime_6B", "RealESR_AnimeVideo_v3"]
    if req.model not in valid_models:
        logger.error(f"[POST-PROCESS] Invalid model: {req.model}")
        raise HTTPException(status_code=400, detail=f"Invalid model. Valid models: {valid_models}")
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL for upscaling...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64 for upscaling...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"upscale_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        logger.info(f"[POST-PROCESS] Starting upscale: {temp_video_path}")
        logger.info(f"[POST-PROCESS] Upscale params: model={req.model}, scale={req.scale_factor}, tile={req.tile_size}, face={req.enhance_face}, denoise={req.denoise_strength}, streaming={req.use_streaming}")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # ================================================================
        # ASYNC BACKGROUND MODE: When callback_url is provided, run in 
        # background to prevent Cloudflare/proxy timeouts (Error 524)
        # ================================================================
        if req.callback_url:
            logger.info(f"[POST-PROCESS] Callback URL provided - running upscale in BACKGROUND mode")
            logger.info(f"[POST-PROCESS] Callback URL: {req.callback_url}")
            
            # FORCE safe values to prevent OOM crash in background processing
            safe_tile_size = req.tile_size
            if safe_tile_size == 0 or safe_tile_size > 512:
                logger.warning(f"[POST-PROCESS] ⚠️ tile_size={req.tile_size} is risky, FORCING to 512 to prevent OOM")
                safe_tile_size = 512
            
            safe_streaming = True  # ALWAYS force streaming in background mode
            if not req.use_streaming:
                logger.warning(f"[POST-PROCESS] ⚠️ use_streaming=False is risky, FORCING to True to prevent OOM crash")
            
            # Define cleanup function
            cleanup_path = temp_video_path if has_base64 else None
            def cleanup():
                if cleanup_path and os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                        logger.debug(f"[BG-CLEANUP] Cleaned up temp file: {cleanup_path}")
                    except Exception as e:
                        logger.warning(f"[BG-CLEANUP] Failed to clean up: {e}")
            
            # Submit to background executor with FORCED safe values
            run_postprocess_in_background(
                operation_name="upscale",
                callback_url=req.callback_url,
                process_func=tb_processor.tb_upscale_video,
                cleanup_func=cleanup,
                video_path=temp_video_path,
                model_key=req.model,
                output_scale_factor_ui=req.scale_factor,
                tile_size=safe_tile_size,
                enhance_face=req.enhance_face,
                denoise_strength_ui=req.denoise_strength,
                use_streaming=safe_streaming,
                progress=DummyProgress(),
                metadata={
                    "model": req.model,
                    "scale_factor": req.scale_factor,
                    "tile_size": safe_tile_size,
                    "enhance_face": req.enhance_face,
                    "denoise_strength": req.denoise_strength,
                    "use_streaming": safe_streaming
                }
            )
            
            # Return immediately - no timeout!
            return AsyncProcessingResponse(
                success=True,
                message="Upscale processing started in background",
                processing=True,
                callback_url=req.callback_url,
                note="Results will be sent to your callback URL when processing completes. This may take several minutes for large videos."
            )
        
        # ================================================================
        # SYNCHRONOUS MODE: No callback URL - process and return directly
        # ================================================================
        logger.info("[POST-PROCESS] No callback URL - running upscale in SYNCHRONOUS mode")
        
        # Call the toolbox processor upscale method
        output_path = tb_processor.tb_upscale_video(
            video_path=temp_video_path,
            model_key=req.model,
            output_scale_factor_ui=req.scale_factor,
            tile_size=req.tile_size,
            enhance_face=req.enhance_face,
            denoise_strength_ui=req.denoise_strength,
            use_streaming=req.use_streaming,
            progress=DummyProgress()
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[POST-PROCESS] Upscale failed - no output produced")
            raise HTTPException(status_code=500, detail="Upscale failed - check server logs for details")
        
        logger.info(f"[POST-PROCESS] Upscale complete: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return UpscaleVideoResponse(
            success=True,
            message="Video upscaled successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to upscale video: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input file if we created one from base64 and running synchronously
        # (Background mode handles its own cleanup)
        if not req.callback_url and temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/interpolate", response_model=Union[InterpolateVideoResponse, AsyncProcessingResponse], tags=["Post-Processing"])
async def interpolate_video(req: InterpolateVideoRequest, x_api_key: str = Header(None)):
    """
    Interpolate video frames using RIFE to increase smoothness.
    
    FPS modes:
    - "2x": Double the frame rate
    - "4x": Quadruple the frame rate
    
    Speed factor: Adjust playback speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
    
    Provide either video_base64 or video_url (not both).
    
    **ASYNC MODE**: When callback_url is provided, processing runs in background
    and results are sent to your callback URL. This prevents timeout errors for
    long operations.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Interpolation request: fps_mode={req.fps_mode}, speed={req.speed_factor}")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for interpolation")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    # Validate and normalize fps_mode - accept multiple formats for flexibility
    # Map API formats to what the toolbox processor expects
    fps_mode_mapping = {
        "2x": "2x Frames",
        "4x": "4x Frames",
        "2x Frames": "2x Frames",
        "4x Frames": "4x Frames",
        "No Interpolation": "No Interpolation",
        "none": "No Interpolation"
    }
    normalized_fps_mode = fps_mode_mapping.get(req.fps_mode)
    if normalized_fps_mode is None:
        logger.error(f"[POST-PROCESS] Invalid fps_mode: {req.fps_mode}")
        raise HTTPException(status_code=400, detail=f"fps_mode must be one of: {list(fps_mode_mapping.keys())}")
    
    logger.info(f"[POST-PROCESS] FPS mode normalized: '{req.fps_mode}' -> '{normalized_fps_mode}'")
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL for interpolation...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64 for interpolation...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"interpolate_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        logger.info(f"[POST-PROCESS] Starting interpolation: {temp_video_path}")
        logger.info(f"[POST-PROCESS] Interpolate params: fps_mode={normalized_fps_mode}, speed={req.speed_factor}, streaming={req.use_streaming}")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # ================================================================
        # ASYNC BACKGROUND MODE: When callback_url is provided, run in 
        # background to prevent Cloudflare/proxy timeouts (Error 524)
        # ================================================================
        if req.callback_url:
            logger.info(f"[POST-PROCESS] Callback URL provided - running interpolation in BACKGROUND mode")
            logger.info(f"[POST-PROCESS] Callback URL: {req.callback_url}")
            
            # FORCE streaming mode to prevent OOM crash in background processing
            safe_streaming = True  # ALWAYS force streaming in background mode
            if not req.use_streaming:
                logger.warning(f"[POST-PROCESS] ⚠️ use_streaming=False is risky, FORCING to True to prevent OOM crash")
            
            # Define cleanup function
            cleanup_path = temp_video_path if has_base64 else None
            def cleanup():
                if cleanup_path and os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                        logger.debug(f"[BG-CLEANUP] Cleaned up temp file: {cleanup_path}")
                    except Exception as e:
                        logger.warning(f"[BG-CLEANUP] Failed to clean up: {e}")
            
            # Submit to background executor with FORCED safe values
            run_postprocess_in_background(
                operation_name="interpolate",
                callback_url=req.callback_url,
                process_func=tb_processor.tb_process_frames,
                cleanup_func=cleanup,
                video_path=temp_video_path,
                target_fps_mode=normalized_fps_mode,
                speed_factor=req.speed_factor,
                use_streaming=safe_streaming,
                progress=DummyProgress(),
                metadata={
                    "fps_mode": req.fps_mode,
                    "speed_factor": req.speed_factor,
                    "use_streaming": safe_streaming
                }
            )
            
            # Return immediately - no timeout!
            return AsyncProcessingResponse(
                success=True,
                message="Interpolation processing started in background",
                processing=True,
                callback_url=req.callback_url,
                note="Results will be sent to your callback URL when processing completes. This may take several minutes."
            )
        
        # ================================================================
        # SYNCHRONOUS MODE: No callback URL - process and return directly
        # ================================================================
        logger.info("[POST-PROCESS] No callback URL - running interpolation in SYNCHRONOUS mode")
        
        # Call the toolbox processor interpolation method
        output_path = tb_processor.tb_process_frames(
            video_path=temp_video_path,
            target_fps_mode=normalized_fps_mode,
            speed_factor=req.speed_factor,
            use_streaming=req.use_streaming,
            progress=DummyProgress()
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[POST-PROCESS] Interpolation failed - no output produced")
            raise HTTPException(status_code=500, detail="Interpolation failed - check server logs for details")
        
        logger.info(f"[POST-PROCESS] Interpolation complete: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return InterpolateVideoResponse(
            success=True,
            message="Video interpolated successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to interpolate video: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input file if running synchronously
        # (Background mode handles its own cleanup)
        if not req.callback_url and temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/filters", response_model=VideoFiltersResponse, tags=["Post-Processing"])
async def apply_video_filters(req: VideoFiltersRequest, x_api_key: str = Header(None)):
    """
    Apply visual filters to a video using FFmpeg.
    
    Filter parameters (all default to 0.0 for no change):
    - brightness: -1.0 to 1.0 (darker to brighter)
    - contrast: -1.0 to 1.0 (less to more contrast)
    - saturation: -1.0 to 1.0 (desaturated to oversaturated)
    - temperature: -1.0 to 1.0 (cooler to warmer)
    - sharpen: 0.0 to 1.0 (no sharpen to max sharpen)
    - blur: 0.0 to 1.0 (no blur to max blur)
    - denoise: 0.0 to 1.0 (no denoise to max denoise)
    - vignette: 0.0 to 1.0 (no vignette to max vignette)
    - s_curve_contrast: 0.0 to 1.0 (cinematic contrast curve)
    - film_grain: 0.0 to 1.0 (film grain effect)
    
    Or use a preset name to apply predefined filter settings.
    
    Provide either video_base64 or video_url (not both).
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Filters request: preset={req.preset}")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for filters")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL for filters...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64 for filters...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"filters_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        logger.info(f"[POST-PROCESS] Applying filters to: {temp_video_path}")
        logger.info(f"[POST-PROCESS] Filter params: brightness={req.brightness}, contrast={req.contrast}, saturation={req.saturation}, temperature={req.temperature}")
        logger.info(f"[POST-PROCESS] Filter params: sharpen={req.sharpen}, blur={req.blur}, denoise={req.denoise}, vignette={req.vignette}")
        logger.info(f"[POST-PROCESS] Filter params: s_curve={req.s_curve_contrast}, film_grain={req.film_grain}, preset={req.preset}")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Call the toolbox processor filters method
        # Method signature: tb_apply_filters(video_path, brightness, contrast, saturation, temperature, sharpen, blur, denoise, vignette, s_curve_contrast, film_grain_strength)
        output_path = tb_processor.tb_apply_filters(
            video_path=temp_video_path,
            brightness=req.brightness,
            contrast=req.contrast,
            saturation=req.saturation,
            temperature=req.temperature,
            sharpen=req.sharpen,
            blur=req.blur,
            denoise=req.denoise,
            vignette=req.vignette,
            s_curve_contrast=req.s_curve_contrast,
            film_grain_strength=req.film_grain,
            progress=DummyProgress()
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[POST-PROCESS] Filters failed - no output produced")
            raise HTTPException(status_code=500, detail="Filters failed - check server logs for details")
        
        logger.info(f"[POST-PROCESS] Filters applied: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="filters",
                success=True,
                output_path=output_path,
                output_video_base64=output_base64,
                message="Filters applied successfully",
                metadata={
                    "preset": req.preset,
                    "brightness": req.brightness,
                    "contrast": req.contrast,
                    "saturation": req.saturation,
                    "temperature": req.temperature,
                    "sharpen": req.sharpen,
                    "blur": req.blur,
                    "denoise": req.denoise,
                    "vignette": req.vignette,
                    "s_curve_contrast": req.s_curve_contrast,
                    "film_grain": req.film_grain
                }
            )
        
        return VideoFiltersResponse(
            success=True,
            message="Filters applied successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to apply filters: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="filters",
                success=False,
                error=error_msg,
                message="Filters failed"
            )
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input file
        if temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/loop", response_model=VideoLoopResponse, tags=["Post-Processing"])
async def create_video_loop(req: VideoLoopRequest, x_api_key: str = Header(None)):
    """
    Create a looping video.
    
    Loop types:
    - "loop": Simple loop - video plays forward repeatedly
    - "ping-pong": Video plays forward then backward (seamless loop)
    
    num_loops: Number of times to loop (1-10)
    
    Provide either video_base64 or video_url (not both).
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Loop request: type={req.loop_type}, num_loops={req.num_loops}")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for loop")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    # Validate loop_type - also support 'none' for no-op (matches Gradio)
    valid_loop_types = ["loop", "ping-pong", "none"]
    if req.loop_type not in valid_loop_types:
        logger.error(f"[POST-PROCESS] Invalid loop_type: {req.loop_type}")
        raise HTTPException(status_code=400, detail=f"loop_type must be one of: {valid_loop_types}")
    
    # Handle 'none' loop type as no-op - return original video
    if req.loop_type == "none":
        logger.info("[POST-PROCESS] Loop type is 'none' - returning original video without modification")
        # Still need to get the video for response
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL for loop...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64 for loop...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"loop_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        logger.info(f"[POST-PROCESS] Creating loop: {temp_video_path}")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Call the toolbox processor loop method
        # Method signature: tb_create_loop(video_path, loop_type, num_loops)
        output_path = tb_processor.tb_create_loop(
            video_path=temp_video_path,
            loop_type=req.loop_type,
            num_loops=req.num_loops,
            progress=DummyProgress()
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[POST-PROCESS] Loop failed - no output produced")
            raise HTTPException(status_code=500, detail="Loop creation failed - check server logs for details")
        
        logger.info(f"[POST-PROCESS] Loop created: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="loop",
                success=True,
                output_path=output_path,
                output_video_base64=output_base64,
                message="Loop created successfully",
                metadata={
                    "loop_type": req.loop_type,
                    "num_loops": req.num_loops
                }
            )
        
        return VideoLoopResponse(
            success=True,
            message="Loop created successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to create loop: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="loop",
                success=False,
                error=error_msg,
                message="Loop creation failed"
            )
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input file
        if temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/join", response_model=JoinVideosResponse, tags=["Post-Processing"])
async def join_videos(req: JoinVideosRequest, x_api_key: str = Header(None)):
    """
    Join multiple videos into a single video.
    
    Provide a list of video URLs to concatenate in order.
    Videos should have compatible resolutions and codecs for best results.
    
    Optional: Specify output_name for the resulting file.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Join request: {len(req.video_urls)} videos")
    
    # Validate input
    if not req.video_urls or len(req.video_urls) < 2:
        logger.error("[POST-PROCESS] Need at least 2 videos to join")
        raise HTTPException(status_code=400, detail="At least 2 video URLs are required")
    
    temp_video_paths = []
    
    try:
        # Download all videos
        for i, url in enumerate(req.video_urls):
            logger.info(f"[POST-PROCESS] Downloading video {i+1}/{len(req.video_urls)}...")
            video_path = download_file_from_url(url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
            temp_video_paths.append(video_path)
        
        logger.info(f"[POST-PROCESS] Joining {len(temp_video_paths)} videos")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Call the toolbox processor join method
        # Method signature: tb_join_videos(video_paths, output_base_name_override)
        output_path = tb_processor.tb_join_videos(
            video_paths=temp_video_paths,
            output_base_name_override=req.output_name
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[POST-PROCESS] Join failed - no output produced")
            raise HTTPException(status_code=500, detail="Join failed - check server logs for details")
        
        logger.info(f"[POST-PROCESS] Videos joined: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="join",
                success=True,
                output_path=output_path,
                output_video_base64=output_base64,
                message="Videos joined successfully",
                metadata={
                    "num_videos": len(req.video_urls),
                    "output_name": req.output_name
                }
            )
        
        return JoinVideosResponse(
            success=True,
            message="Videos joined successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to join videos: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="join",
                success=False,
                error=error_msg,
                message="Join videos failed"
            )
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input files
        for path in temp_video_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"[POST-PROCESS] Cleaned up temp file: {path}")
                except Exception as e:
                    logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/export", response_model=ExportVideoResponse, tags=["Post-Processing"])
async def export_video(req: ExportVideoRequest, x_api_key: str = Header(None)):
    """
    Export/convert a video to a different format with quality settings.
    
    Formats:
    - "MP4": Standard MP4 (H.264)
    - "WebM": WebM format (VP9)
    - "GIF": Animated GIF
    
    Quality: 1-100 (higher = better quality, larger file)
    max_width: Optional maximum width (preserves aspect ratio)
    
    Provide either video_base64 or video_url (not both).
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Export request: format={req.format}, quality={req.quality}")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for export")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    # Validate format
    if req.format not in ["MP4", "WebM", "GIF"]:
        logger.error(f"[POST-PROCESS] Invalid format: {req.format}")
        raise HTTPException(status_code=400, detail="format must be 'MP4', 'WebM', or 'GIF'")
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL for export...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.gif'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64 for export...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"export_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        logger.info(f"[POST-PROCESS] Exporting: {temp_video_path} -> {req.format}")
        logger.info(f"[POST-PROCESS] Export params: quality={req.quality}, max_width={req.max_width}")
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Call the toolbox processor export method
        # Method signature: tb_export_video(video_path, export_format, quality_slider, max_width, output_base_name_override)
        output_path = tb_processor.tb_export_video(
            video_path=temp_video_path,
            export_format=req.format,
            quality_slider=req.quality,
            max_width=req.max_width or 1920,  # Default max width
            output_base_name_override=req.output_name,
            progress=DummyProgress()
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[POST-PROCESS] Export failed - no output produced")
            raise HTTPException(status_code=500, detail="Export failed - check server logs for details")
        
        logger.info(f"[POST-PROCESS] Export complete: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="export",
                success=True,
                output_path=output_path,
                output_video_base64=output_base64,
                message="Video exported successfully",
                metadata={
                    "format": req.format,
                    "quality": req.quality,
                    "max_width": req.max_width,
                    "output_name": req.output_name
                }
            )
        
        return ExportVideoResponse(
            success=True,
            message="Video exported successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to export video: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="export",
                success=False,
                error=error_msg,
                message="Export failed"
            )
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input file
        if temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")


@app.post("/postprocess/pipeline", response_model=Union[PipelineResponse, AsyncProcessingResponse], tags=["Post-Processing"])
async def run_pipeline(req: PipelineRequest, x_api_key: str = Header(None)):
    """
    Run a pipeline of multiple post-processing operations on a video.
    
    Operations are executed in order. Each operation can be:
    - "upscale": Upscale with Real-ESRGAN
    - "interpolate": Frame interpolation with RIFE
    - "filters": Apply visual filters
    - "loop": Create loop
    - "export": Export to format
    
    Example operations array:
    [
        {"type": "upscale", "params": {"model": "RealESRGAN_x2plus"}},
        {"type": "interpolate", "params": {"fps_mode": "2x"}},
        {"type": "export", "params": {"format": "MP4", "quality": 90}}
    ]
    
    Provide either video_base64 or video_url (not both).
    
    **ASYNC MODE**: When callback_url is provided, processing runs in background
    and results are sent to your callback URL. This is HIGHLY RECOMMENDED for
    pipelines as they can take many minutes.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[POST-PROCESS] Pipeline request: {len(req.operations)} operations")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[POST-PROCESS] No video source provided for pipeline")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    if not req.operations or len(req.operations) == 0:
        logger.error("[POST-PROCESS] No operations provided for pipeline")
        raise HTTPException(status_code=400, detail="At least one operation is required")
    
    # Validate operation types
    valid_ops = ["upscale", "interpolate", "filters", "loop", "export"]
    for op in req.operations:
        if op.type not in valid_ops:
            logger.error(f"[POST-PROCESS] Invalid operation type: {op.type}")
            raise HTTPException(status_code=400, detail=f"Invalid operation type: {op.type}. Valid types: {valid_ops}")
    
    temp_video_path = None
    
    try:
        # Get video file path
        if has_url:
            logger.info(f"[POST-PROCESS] Downloading video from URL for pipeline...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        else:
            logger.info("[POST-PROCESS] Decoding video from base64 for pipeline...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"pipeline_input_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        # Check if toolbox processor is available
        if tb_processor is None:
            logger.error("[POST-PROCESS] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # ================================================================
        # ASYNC BACKGROUND MODE: When callback_url is provided, run in 
        # background to prevent Cloudflare/proxy timeouts (Error 524)
        # ================================================================
        if req.callback_url:
            logger.info(f"[POST-PROCESS] Callback URL provided - running pipeline in BACKGROUND mode")
            logger.info(f"[POST-PROCESS] Callback URL: {req.callback_url}")
            
            # Define the pipeline process function
            def run_pipeline_operations(input_path, operations, tb_proc, has_b64):
                """Run all pipeline operations and return final output path"""
                current_video_path = input_path
                intermediate_files = []
                results = []
                
                try:
                    for i, op in enumerate(operations):
                        logger.info(f"[BG-PIPELINE] Step {i+1}/{len(operations)}: {op.type}")
                        logger.info(f"[BG-PIPELINE] Step {i+1} params: {op.params}")
                        
                        output_path = None
                        
                        if op.type == "upscale":
                            params = op.params or {}
                            # FORCE streaming mode and reasonable tile size to prevent OOM in background processing
                            tile_size = params.get("tile_size", 512)
                            if tile_size == 0 or tile_size > 512:
                                old_tile_size = tile_size
                                tile_size = 512  # Use 512 to prevent OOM
                                logger.warning(f"[BG-PIPELINE] ⚠️ tile_size was {old_tile_size}, FORCING to 512 to prevent OOM")
                            
                            # ALWAYS force streaming mode for background processing - in-memory mode causes 100% RAM crash
                            requested_streaming = params.get("use_streaming", True)
                            use_streaming = True  # FORCED - ignore request
                            if not requested_streaming:
                                logger.warning(f"[BG-PIPELINE] ⚠️ use_streaming was False, FORCING to True to prevent OOM crash")
                            
                            logger.info(f"[BG-PIPELINE] Starting upscale with model={params.get('model', 'RealESRGAN_x2plus')}, tile_size={tile_size}, streaming={use_streaming}")
                            
                            output_path = tb_proc.tb_upscale_video(
                                video_path=current_video_path,
                                model_key=params.get("model", "RealESRGAN_x2plus"),
                                output_scale_factor_ui=params.get("scale_factor", 2.0),
                                tile_size=tile_size,
                                enhance_face=params.get("enhance_face", False),
                                denoise_strength_ui=params.get("denoise_strength", 0.5),
                                use_streaming=use_streaming,
                                progress=DummyProgress()
                            )
                            logger.info(f"[BG-PIPELINE] Upscale returned: {output_path}")
                        
                        elif op.type == "interpolate":
                            params = op.params or {}
                            # ALWAYS force streaming mode for background processing
                            requested_streaming = params.get("use_streaming", True)
                            use_streaming = True  # FORCED - ignore request
                            if not requested_streaming:
                                logger.warning(f"[BG-PIPELINE] ⚠️ use_streaming was False for interpolate, FORCING to True to prevent OOM crash")
                            
                            logger.info(f"[BG-PIPELINE] Starting interpolate with fps_mode={params.get('fps_mode', '2x')}, streaming={use_streaming}")
                            output_path = tb_proc.tb_process_frames(
                                video_path=current_video_path,
                                target_fps_mode=params.get("fps_mode", "2x"),
                                speed_factor=params.get("speed_factor", 1.0),
                                use_streaming=use_streaming,
                                progress=DummyProgress()
                            )
                            logger.info(f"[BG-PIPELINE] Interpolate returned: {output_path}")
                        
                        elif op.type == "filters":
                            params = op.params or {}
                            logger.info(f"[BG-PIPELINE] Starting filters with params: brightness={params.get('brightness', 0.0)}, contrast={params.get('contrast', 1.0)}")
                            output_path = tb_proc.tb_apply_filters(
                                video_path=current_video_path,
                                brightness=params.get("brightness", 0.0),
                                contrast=params.get("contrast", 1.0),
                                saturation=params.get("saturation", 1.0),
                                temperature=params.get("temperature", 0.0),
                                sharpen=params.get("sharpen", 0.0),
                                blur=params.get("blur", 0.0),
                                denoise=params.get("denoise", 0.0),
                                vignette=params.get("vignette", 0.0),
                                s_curve_contrast=params.get("s_curve_contrast", 0.0),
                                film_grain_strength=params.get("film_grain", 0.0),
                                progress=DummyProgress()
                            )
                            logger.info(f"[BG-PIPELINE] Filters returned: {output_path}")
                        
                        elif op.type == "loop":
                            params = op.params or {}
                            logger.info(f"[BG-PIPELINE] Starting loop with type={params.get('loop_type', 'loop')}, num_loops={params.get('num_loops', 2)}")
                            output_path = tb_proc.tb_create_loop(
                                video_path=current_video_path,
                                loop_type=params.get("loop_type", "loop"),
                                num_loops=params.get("num_loops", 2),
                                progress=DummyProgress()
                            )
                            logger.info(f"[BG-PIPELINE] Loop returned: {output_path}")
                        
                        elif op.type == "export":
                            params = op.params or {}
                            logger.info(f"[BG-PIPELINE] Starting export with format={params.get('format', 'MP4')}, quality={params.get('quality', 85)}")
                            output_path = tb_proc.tb_export_video(
                                video_path=current_video_path,
                                export_format=params.get("format", "MP4"),
                                quality_slider=params.get("quality", 85),
                                max_width=params.get("max_width") or 1920,
                                output_base_name_override=params.get("output_name"),
                                progress=DummyProgress()
                            )
                            logger.info(f"[BG-PIPELINE] Export returned: {output_path}")
                        
                        # Check result
                        if output_path is None or not os.path.exists(output_path):
                            error_msg = f"Pipeline step {i+1} ({op.type}) failed - no output produced"
                            logger.error(f"[BG-PIPELINE] {error_msg}")
                            results.append({"step": i+1, "type": op.type, "success": False, "message": "Operation failed"})
                            raise Exception(error_msg)
                        
                        results.append({"step": i+1, "type": op.type, "success": True, "message": "Operation completed"})
                        
                        # Track intermediate files for cleanup
                        if current_video_path != input_path and current_video_path != output_path:
                            intermediate_files.append(current_video_path)
                        
                        current_video_path = output_path
                        logger.info(f"[BG-PIPELINE] Step {i+1} complete: {output_path}")
                    
                    logger.info(f"[BG-PIPELINE] All steps complete, returning: {current_video_path}")
                    return current_video_path
                
                except Exception as e:
                    logger.error(f"[BG-PIPELINE] Pipeline failed with error: {str(e)}")
                    logger.error(f"[BG-PIPELINE] Traceback: {traceback.format_exc()}")
                    raise
                    
                finally:
                    # Clean up intermediate files
                    for path in intermediate_files:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                                logger.debug(f"[BG-PIPELINE] Cleaned up intermediate: {path}")
                            except Exception as e:
                                logger.warning(f"[BG-PIPELINE] Cleanup failed: {e}")
                    
                    # Clean up input if from base64
                    if has_b64 and input_path and os.path.exists(input_path):
                        try:
                            os.remove(input_path)
                            logger.debug(f"[BG-PIPELINE] Cleaned up input: {input_path}")
                        except Exception as e:
                            logger.warning(f"[BG-PIPELINE] Input cleanup failed: {e}")
            
            # Submit to background executor
            run_postprocess_in_background(
                operation_name="pipeline",
                callback_url=req.callback_url,
                process_func=run_pipeline_operations,
                cleanup_func=None,  # Cleanup handled inside the function
                input_path=temp_video_path,
                operations=req.operations,
                tb_proc=tb_processor,
                has_b64=has_base64,
                metadata={
                    "operations_count": len(req.operations),
                    "operations": [{"type": op.type, "params": op.params} for op in req.operations]
                }
            )
            
            # Return immediately - no timeout!
            return AsyncProcessingResponse(
                success=True,
                message=f"Pipeline with {len(req.operations)} operations started in background",
                processing=True,
                callback_url=req.callback_url,
                note="Results will be sent to your callback URL when all operations complete. This may take 10+ minutes for complex pipelines."
            )
        
        # ================================================================
        # SYNCHRONOUS MODE: No callback URL - process and return directly
        # (WARNING: May timeout for long pipelines!)
        # ================================================================
        logger.info("[POST-PROCESS] No callback URL - running pipeline in SYNCHRONOUS mode (may timeout!)")
        logger.warning("[POST-PROCESS] Pipeline without callback_url may timeout! Consider providing callback_url.")
        
        current_video_path = temp_video_path
        intermediate_files = []
        results = []
        
        # Execute each operation in sequence
        for i, op in enumerate(req.operations):
            logger.info(f"[POST-PROCESS] Pipeline step {i+1}/{len(req.operations)}: {op.type}")
            
            output_path = None
            
            if op.type == "upscale":
                params = op.params or {}
                output_path = tb_processor.tb_upscale_video(
                    video_path=current_video_path,
                    model_key=params.get("model", "RealESRGAN_x2plus"),
                    output_scale_factor_ui=params.get("scale_factor", 2.0),
                    tile_size=params.get("tile_size", 512),
                    enhance_face=params.get("enhance_face", False),
                    denoise_strength_ui=params.get("denoise_strength", 0.5),
                    use_streaming=params.get("use_streaming", True),
                    progress=DummyProgress()
                )
            
            elif op.type == "interpolate":
                params = op.params or {}
                output_path = tb_processor.tb_process_frames(
                    video_path=current_video_path,
                    target_fps_mode=params.get("fps_mode", "2x"),
                    speed_factor=params.get("speed_factor", 1.0),
                    use_streaming=params.get("use_streaming", True),
                    progress=DummyProgress()
                )
            
            elif op.type == "filters":
                params = op.params or {}
                output_path = tb_processor.tb_apply_filters(
                    video_path=current_video_path,
                    brightness=params.get("brightness", 0.0),
                    contrast=params.get("contrast", 1.0),
                    saturation=params.get("saturation", 1.0),
                    temperature=params.get("temperature", 0.0),
                    sharpen=params.get("sharpen", 0.0),
                    blur=params.get("blur", 0.0),
                    denoise=params.get("denoise", 0.0),
                    vignette=params.get("vignette", 0.0),
                    s_curve_contrast=params.get("s_curve_contrast", 0.0),
                    film_grain_strength=params.get("film_grain", 0.0),
                    progress=DummyProgress()
                )
            
            elif op.type == "loop":
                params = op.params or {}
                output_path = tb_processor.tb_create_loop(
                    video_path=current_video_path,
                    loop_type=params.get("loop_type", "loop"),
                    num_loops=params.get("num_loops", 2),
                    progress=DummyProgress()
                )
            
            elif op.type == "export":
                params = op.params or {}
                output_path = tb_processor.tb_export_video(
                    video_path=current_video_path,
                    export_format=params.get("format", "MP4"),
                    quality_slider=params.get("quality", 85),
                    max_width=params.get("max_width") or 1920,
                    output_base_name_override=params.get("output_name"),
                    progress=DummyProgress()
                )
            
            # Check result
            if output_path is None or not os.path.exists(output_path):
                logger.error(f"[POST-PROCESS] Pipeline step {i+1} failed")
                results.append({"step": i+1, "type": op.type, "success": False, "message": "Operation failed"})
                raise HTTPException(status_code=500, detail=f"Pipeline step {i+1} ({op.type}) failed")
            
            results.append({"step": i+1, "type": op.type, "success": True, "message": "Operation completed"})
            
            # Track intermediate files for cleanup (but not the final output)
            if current_video_path != temp_video_path and current_video_path != output_path:
                intermediate_files.append(current_video_path)
            
            current_video_path = output_path
            logger.info(f"[POST-PROCESS] Pipeline step {i+1} complete: {output_path}")
        
        logger.info(f"[POST-PROCESS] Pipeline complete: {current_video_path}")
        
        # Read final output and convert to base64
        with open(current_video_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return PipelineResponse(
            success=True,
            message=f"Pipeline completed successfully ({len(req.operations)} operations)",
            output_video_base64=output_base64,
            output_path=current_video_path,
            operations_completed=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(f"[POST-PROCESS] {error_msg}")
        logger.error(f"[POST-PROCESS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp input file (only if synchronous mode)
        if not req.callback_url:
            if temp_video_path and has_base64 and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    logger.debug(f"[POST-PROCESS] Cleaned up temp file: {temp_video_path}")
                except Exception as e:
                    logger.warning(f"[POST-PROCESS] Failed to clean up temp file: {e}")
            
            # Clean up intermediate files (if they exist in local scope)
            if 'intermediate_files' in locals():
                for path in intermediate_files:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.debug(f"[POST-PROCESS] Cleaned up intermediate file: {path}")
                        except Exception as e:
                            logger.warning(f"[POST-PROCESS] Failed to clean up intermediate file: {e}")


# ============================================================================
# PHASE 3: FRAMES STUDIO ENDPOINTS
# ============================================================================

@app.post("/frames/extract", response_model=ExtractFramesResponse, tags=["Frames Studio"])
async def extract_frames(req: ExtractFramesRequest, x_api_key: str = Header(None)):
    """
    Extract frames from a video for frame-by-frame editing.
    
    Use this to:
    - Extract all frames (extraction_rate=1)
    - Extract every Nth frame to reduce frame count
    
    After extraction, use /frames/folders to find your extracted folder,
    then /frames/{folder} to list and manage individual frames.
    
    Provide either video_base64 or video_url (not both).
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] Extract frames request: rate={req.extraction_rate}")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        logger.error("[FRAMES] No video source provided")
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    temp_video_path = None
    
    try:
        # Get video to a temp file
        if has_base64:
            logger.info("[FRAMES] Decoding base64 video for extraction...")
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"temp_extract_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
            logger.info(f"[FRAMES] Temp video saved: {temp_video_path}")
        else:
            logger.info(f"[FRAMES] Downloading video from URL for extraction...")
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
            logger.info(f"[FRAMES] Video downloaded: {temp_video_path}")
        
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Call the toolbox processor extract method
        # Method signature: tb_extract_frames(video_path, extraction_rate, progress)
        logger.info(f"[FRAMES] Starting extraction with rate={req.extraction_rate}...")
        output_folder = tb_processor.tb_extract_frames(
            video_path=temp_video_path,
            extraction_rate=req.extraction_rate,
            progress=DummyProgress()
        )
        
        if output_folder is None or not os.path.exists(output_folder):
            logger.error("[FRAMES] Extraction failed - no output folder")
            raise HTTPException(status_code=500, detail="Frame extraction failed - check server logs")
        
        # Count frames in the output folder
        frame_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        frame_count = len(frame_files)
        folder_name = os.path.basename(output_folder)
        
        logger.info(f"[FRAMES] Extraction complete: {frame_count} frames in {folder_name}")
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="extract_frames",
                success=True,
                output_path=output_folder,
                message=f"Successfully extracted {frame_count} frames",
                metadata={
                    "extraction_rate": req.extraction_rate,
                    "folder_name": folder_name,
                    "frame_count": frame_count
                }
            )
        
        return ExtractFramesResponse(
            success=True,
            message=f"Successfully extracted {frame_count} frames",
            folder_name=folder_name,
            frame_count=frame_count,
            frames_path=output_folder
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to extract frames: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="extract_frames",
                success=False,
                error=error_msg,
                message="Frame extraction failed"
            )
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp video file
        if temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.debug(f"[FRAMES] Cleaned up temp file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"[FRAMES] Failed to clean up temp file: {e}")


@app.get("/frames/folders", response_model=FrameFoldersResponse, tags=["Frames Studio"])
async def list_frame_folders(x_api_key: str = Header(None)):
    """
    List all extracted frame folders.
    
    Returns a list of folder names that can be used with other /frames/ endpoints.
    Each folder represents a previous frame extraction operation.
    """
    validate_api_key(x_api_key)
    
    logger.info("[FRAMES] List folders request")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Get list of extracted frame folders
        folders = tb_processor.tb_get_extracted_frame_folders()
        
        logger.info(f"[FRAMES] Found {len(folders)} folders")
        
        return FrameFoldersResponse(
            success=True,
            folders=folders,
            message=f"Found {len(folders)} extracted frame folder(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to list folders: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/frames/{folder}", response_model=ListFramesResponse, tags=["Frames Studio"])
async def list_frames_in_folder(folder: str, x_api_key: str = Header(None)):
    """
    List all frames in a specific extracted folder.
    
    Returns frame information including filename, path, and index.
    Use this to browse frames before deletion or saving operations.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] List frames request for folder: {folder}")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Get frames from the folder
        frame_paths = tb_processor.tb_get_frames_from_folder(folder)
        
        if not frame_paths:
            logger.warning(f"[FRAMES] No frames found in folder: {folder}")
            return ListFramesResponse(
                success=True,
                folder=folder,
                frames=[],
                total_count=0,
                message=f"No frames found in folder '{folder}'"
            )
        
        # Build frame info list
        frames = []
        for i, path in enumerate(frame_paths):
            frames.append(FrameInfo(
                filename=os.path.basename(path),
                path=path,
                index=i
            ))
        
        logger.info(f"[FRAMES] Found {len(frames)} frames in {folder}")
        
        return ListFramesResponse(
            success=True,
            folder=folder,
            frames=frames,
            total_count=len(frames),
            message=f"Found {len(frames)} frame(s) in folder '{folder}'"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to list frames: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete("/frames/{folder}", response_model=DeleteFolderResponse, tags=["Frames Studio"])
async def delete_frame_folder(folder: str, x_api_key: str = Header(None)):
    """
    Delete an entire extracted frames folder and all its contents.
    
    This is a destructive operation - all frames in the folder will be permanently deleted.
    Use with caution!
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] Delete folder request: {folder}")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Delete the folder
        success = tb_processor.tb_delete_extracted_frames_folder(folder)
        
        if success:
            logger.info(f"[FRAMES] Successfully deleted folder: {folder}")
            return DeleteFolderResponse(
                success=True,
                folder=folder,
                message=f"Successfully deleted folder '{folder}' and all its contents"
            )
        else:
            logger.error(f"[FRAMES] Failed to delete folder: {folder}")
            return DeleteFolderResponse(
                success=False,
                folder=folder,
                message=f"Failed to delete folder '{folder}' - check server logs"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to delete folder: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete("/frames/{folder}/{frame}", response_model=DeleteFrameResponse, tags=["Frames Studio"])
async def delete_single_frame(folder: str, frame: str, x_api_key: str = Header(None)):
    """
    Delete a single frame from an extracted frames folder.
    
    Use this to remove bad or glitchy frames before reassembling the video.
    The frame parameter should be the filename (e.g., "frame_000123.png").
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] Delete frame request: {folder}/{frame}")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Build the full path to the frame
        frame_path = os.path.join(tb_processor.extracted_frames_target_path, folder, frame)
        
        if not os.path.exists(frame_path):
            logger.error(f"[FRAMES] Frame not found: {frame_path}")
            raise HTTPException(status_code=404, detail=f"Frame '{frame}' not found in folder '{folder}'")
        
        # Delete the frame
        result = tb_processor.tb_delete_single_frame(frame_path)
        
        if "✅" in result or "Deleted" in result:
            logger.info(f"[FRAMES] Successfully deleted frame: {frame}")
            return DeleteFrameResponse(
                success=True,
                folder=folder,
                frame=frame,
                message=f"Successfully deleted frame '{frame}'"
            )
        else:
            logger.error(f"[FRAMES] Failed to delete frame: {result}")
            return DeleteFrameResponse(
                success=False,
                folder=folder,
                frame=frame,
                message=result
            )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to delete frame: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/frames/{folder}/{frame}", response_model=GetFrameResponse, tags=["Frames Studio"])
async def get_single_frame(folder: str, frame: str, x_api_key: str = Header(None)):
    """
    Get a single frame image with base64 data for preview.
    
    Returns the frame image encoded as base64 for displaying in the UI.
    Use this to show frame previews before selecting frames to delete.
    
    The frame parameter should be the filename (e.g., "frame_000123.png").
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] Get frame request: {folder}/{frame}")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Build the full path to the frame
        frame_path = os.path.join(tb_processor.extracted_frames_target_path, folder, frame)
        
        if not os.path.exists(frame_path):
            logger.error(f"[FRAMES] Frame not found: {frame_path}")
            raise HTTPException(status_code=404, detail=f"Frame '{frame}' not found in folder '{folder}'")
        
        # Determine MIME type based on extension
        ext = os.path.splitext(frame)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(ext, 'image/png')
        
        # Read the frame and encode as base64
        with open(frame_path, 'rb') as f:
            frame_data = f.read()
        
        frame_base64 = base64.b64encode(frame_data).decode('utf-8')
        frame_size = len(frame_data)
        
        logger.info(f"[FRAMES] Successfully read frame: {frame} ({frame_size} bytes)")
        
        return GetFrameResponse(
            success=True,
            folder=folder,
            filename=frame,
            size=frame_size,
            base64=frame_base64,
            mime_type=mime_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get frame: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/frames/{folder}/save/{frame}", response_model=SaveFrameResponse, tags=["Frames Studio"])
async def save_single_frame(folder: str, frame: str, x_api_key: str = Header(None)):
    """
    Save a single frame as a high-quality image to permanent storage.
    
    This copies the frame to the saved_videos directory with a timestamped filename.
    Useful for extracting a good frame to use as an image prompt for generation.
    
    Returns both the saved path and the frame as base64.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] Save frame request: {folder}/{frame}")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Build the full path to the frame
        frame_path = os.path.join(tb_processor.extracted_frames_target_path, folder, frame)
        
        if not os.path.exists(frame_path):
            logger.error(f"[FRAMES] Frame not found: {frame_path}")
            raise HTTPException(status_code=404, detail=f"Frame '{frame}' not found in folder '{folder}'")
        
        # Save the frame
        saved_path = tb_processor.tb_save_single_frame(frame_path)
        
        if saved_path is None:
            logger.error(f"[FRAMES] Failed to save frame: {frame}")
            raise HTTPException(status_code=500, detail="Failed to save frame - check server logs")
        
        # Read frame and convert to base64
        with open(saved_path, 'rb') as f:
            frame_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        logger.info(f"[FRAMES] Successfully saved frame to: {saved_path}")
        
        return SaveFrameResponse(
            success=True,
            message=f"Successfully saved frame '{frame}'",
            saved_path=saved_path,
            frame_base64=frame_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to save frame: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/frames/reassemble", response_model=ReassembleFramesResponse, tags=["Frames Studio"])
async def reassemble_frames(req: ReassembleFramesRequest, x_api_key: str = Header(None)):
    """
    Reassemble extracted frames back into a video.
    
    After editing frames (deleting bad ones), use this to create a new video
    from the remaining frames in a folder.
    
    The folder_name should match one from /frames/folders.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FRAMES] Reassemble request: folder={req.folder_name}, fps={req.output_fps}")
    
    try:
        if tb_processor is None:
            logger.error("[FRAMES] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Frames processing module not available")
        
        # Build the full path to the frames folder
        frames_path = os.path.join(tb_processor.extracted_frames_target_path, req.folder_name)
        
        if not os.path.exists(frames_path) or not os.path.isdir(frames_path):
            logger.error(f"[FRAMES] Folder not found: {frames_path}")
            raise HTTPException(status_code=404, detail=f"Folder '{req.folder_name}' not found")
        
        # Reassemble frames to video
        # Method signature: tb_reassemble_frames_to_video(frames_source, output_fps, output_base_name_override)
        logger.info(f"[FRAMES] Starting reassembly from {frames_path}...")
        output_path = tb_processor.tb_reassemble_frames_to_video(
            frames_source=frames_path,
            output_fps=req.output_fps,
            output_base_name_override=req.output_name,
            progress=DummyProgress()
        )
        
        if output_path is None or not os.path.exists(output_path):
            logger.error("[FRAMES] Reassembly failed - no output produced")
            raise HTTPException(status_code=500, detail="Frame reassembly failed - check server logs")
        
        logger.info(f"[FRAMES] Reassembly complete: {output_path}")
        
        # Read output and convert to base64
        with open(output_path, 'rb') as f:
            output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="reassemble_frames",
                success=True,
                output_path=output_path,
                output_video_base64=output_base64,
                message="Frames reassembled successfully",
                metadata={
                    "folder_name": req.folder_name,
                    "output_fps": req.output_fps,
                    "output_name": req.output_name
                }
            )
        
        return ReassembleFramesResponse(
            success=True,
            message="Frames reassembled successfully",
            output_video_base64=output_base64,
            output_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to reassemble frames: {str(e)}"
        logger.error(f"[FRAMES] {error_msg}")
        logger.error(f"[FRAMES] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="reassemble_frames",
                success=False,
                error=error_msg,
                message="Frame reassembly failed"
            )
        
        raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# PHASE 3: WORKFLOW PRESET ENDPOINTS
# ============================================================================

@app.get("/workflow/presets", response_model=ListWorkflowPresetsResponse, tags=["Workflow Presets"])
async def list_workflow_presets(x_api_key: str = Header(None)):
    """
    List all available workflow presets.
    
    Workflow presets save complete pipeline configurations including:
    - Active pipeline steps
    - All parameter values for each step
    
    Use these to quickly apply saved processing workflows.
    """
    validate_api_key(x_api_key)
    
    logger.info("[WORKFLOW] List presets request")
    
    try:
        # Refresh presets from file
        _initialize_workflow_presets()
        
        logger.info(f"[WORKFLOW] Found {len(tb_workflow_presets_data)} presets")
        
        return ListWorkflowPresetsResponse(
            success=True,
            presets=tb_workflow_presets_data,
            message=f"Found {len(tb_workflow_presets_data)} workflow preset(s)"
        )
        
    except Exception as e:
        error_msg = f"Failed to list presets: {str(e)}"
        logger.error(f"[WORKFLOW] {error_msg}")
        logger.error(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/workflow/presets", response_model=WorkflowPresetResponse, tags=["Workflow Presets"])
async def save_workflow_preset(req: SaveWorkflowPresetRequest, x_api_key: str = Header(None)):
    """
    Save a new workflow preset or update an existing one.
    
    Workflow presets include:
    - active_steps: List of steps to enable (e.g., ["upscale", "interpolate", "export"])
    - params: All parameter values for each processing step
    
    Use GET /workflow/presets to see the expected parameter structure.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[WORKFLOW] Save preset request: {req.name}")
    
    try:
        global tb_workflow_presets_data
        
        clean_name = req.name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Preset name cannot be empty")
        
        # Convert Pydantic model to dict
        preset_data = {
            "active_steps": req.preset_data.active_steps,
            "params": req.preset_data.params.model_dump()
        }
        
        preset_existed = clean_name in tb_workflow_presets_data
        tb_workflow_presets_data[clean_name] = preset_data
        
        # Save to file
        with open(TB_WORKFLOW_PRESETS_FILE, 'w') as f:
            json.dump(tb_workflow_presets_data, f, indent=4)
        
        action = "updated" if preset_existed else "saved"
        logger.info(f"[WORKFLOW] Preset '{clean_name}' {action} successfully")
        
        return WorkflowPresetResponse(
            success=True,
            message=f"Workflow preset '{clean_name}' {action} successfully",
            preset_name=clean_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to save preset: {str(e)}"
        logger.error(f"[WORKFLOW] {error_msg}")
        logger.error(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete("/workflow/presets/{name}", response_model=WorkflowPresetResponse, tags=["Workflow Presets"])
async def delete_workflow_preset(name: str, x_api_key: str = Header(None)):
    """
    Delete a workflow preset by name.
    
    The "None" preset cannot be deleted as it represents default values.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[WORKFLOW] Delete preset request: {name}")
    
    try:
        global tb_workflow_presets_data
        
        clean_name = name.strip()
        
        if clean_name == "None":
            logger.warning("[WORKFLOW] Cannot delete 'None' preset")
            raise HTTPException(status_code=400, detail="Cannot delete the 'None' preset - it represents default values")
        
        if clean_name not in tb_workflow_presets_data:
            logger.warning(f"[WORKFLOW] Preset not found: {clean_name}")
            raise HTTPException(status_code=404, detail=f"Preset '{clean_name}' not found")
        
        # Remove the preset
        del tb_workflow_presets_data[clean_name]
        
        # Save to file
        with open(TB_WORKFLOW_PRESETS_FILE, 'w') as f:
            json.dump(tb_workflow_presets_data, f, indent=4)
        
        logger.info(f"[WORKFLOW] Preset '{clean_name}' deleted successfully")
        
        return WorkflowPresetResponse(
            success=True,
            message=f"Workflow preset '{clean_name}' deleted successfully",
            preset_name=clean_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to delete preset: {str(e)}"
        logger.error(f"[WORKFLOW] {error_msg}")
        logger.error(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# PHASE 3: SYSTEM UTILITY ENDPOINTS
# ============================================================================

@app.post("/system/clear-temp", response_model=ClearTempResponse, tags=["System Utilities"])
async def clear_temporary_files(x_api_key: str = Header(None)):
    """
    Clear temporary files from the toolbox processing directories.
    
    This removes:
    - Temporary video files from post-processing temp folder
    - Gradio temp folder contents (matches Gradio UI behavior)
    - API temp files (prefixed with temp_)
    - Intermediate processing files
    - Extracted frames that haven't been saved
    
    Use this to free up disk space after processing.
    """
    validate_api_key(x_api_key)
    
    logger.info("[SYSTEM] Clear temp files request")
    
    try:
        files_deleted = 0
        space_freed = 0
        
        # Helper function to clean a directory and track stats
        def clean_directory(dir_path: str, description: str) -> tuple:
            """Clean a directory and return (files_deleted, space_freed)"""
            local_files = 0
            local_space = 0
            
            if not dir_path or not os.path.exists(dir_path):
                logger.debug(f"[SYSTEM] {description}: Path not found or not set - {dir_path}")
                return 0, 0
            
            try:
                items = os.listdir(dir_path)
                if not items:
                    logger.debug(f"[SYSTEM] {description}: Already empty")
                    return 0, 0
                
                for item in items:
                    item_path = os.path.join(dir_path, item)
                    try:
                        if os.path.isfile(item_path):
                            local_space += os.path.getsize(item_path)
                            os.remove(item_path)
                            local_files += 1
                            logger.debug(f"[SYSTEM] Deleted temp file: {item_path}")
                        elif os.path.isdir(item_path):
                            # Calculate folder size before deletion
                            for root, dirs, files in os.walk(item_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    try:
                                        local_space += os.path.getsize(file_path)
                                        local_files += 1
                                    except OSError:
                                        pass
                            shutil.rmtree(item_path)
                            logger.debug(f"[SYSTEM] Deleted temp folder: {item_path}")
                    except Exception as e:
                        logger.warning(f"[SYSTEM] Failed to delete {item_path}: {e}")
                
                logger.info(f"[SYSTEM] {description}: Cleaned {local_files} items, freed {local_space / (1024 * 1024):.2f} MB")
                
            except Exception as e:
                logger.warning(f"[SYSTEM] Error cleaning {description}: {e}")
            
            return local_files, local_space
        
        # 1. Clean Post-processing Temp Folder (matches Gradio UI)
        if tb_processor is not None:
            postproc_temp_dir = tb_processor._base_temp_output_dir
            deleted, freed = clean_directory(postproc_temp_dir, "Post-processing temp folder")
            files_deleted += deleted
            space_freed += freed
            
            # 2. Clean Gradio Temp Folder (matches Gradio UI - this was missing!)
            gradio_temp_dir = settings.get("gradio_temp_dir")
            if gradio_temp_dir:
                deleted, freed = clean_directory(gradio_temp_dir, "Gradio temp folder")
                files_deleted += deleted
                space_freed += freed
        
        # 3. Also clean API temp files (temp_* prefix files only)
        if os.path.exists(api_output_dir):
            api_deleted = 0
            api_freed = 0
            for item in os.listdir(api_output_dir):
                if item.startswith("temp_"):
                    item_path = os.path.join(api_output_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            size = os.path.getsize(item_path)
                            os.remove(item_path)
                            api_deleted += 1
                            api_freed += size
                            logger.debug(f"[SYSTEM] Deleted API temp file: {item_path}")
                    except Exception as e:
                        logger.warning(f"[SYSTEM] Failed to delete {item_path}: {e}")
            
            if api_deleted > 0:
                logger.info(f"[SYSTEM] API temp files: Cleaned {api_deleted} items, freed {api_freed / (1024 * 1024):.2f} MB")
            files_deleted += api_deleted
            space_freed += api_freed
        
        space_freed_mb = space_freed / (1024 * 1024)
        
        logger.info(f"[SYSTEM] Clear temp complete: {files_deleted} files, freed {space_freed_mb:.2f} MB")
        
        return ClearTempResponse(
            success=True,
            message=f"Cleared {files_deleted} temporary file(s), freed {space_freed_mb:.2f} MB",
            files_deleted=files_deleted,
            space_freed_mb=round(space_freed_mb, 2)
        )
        
    except Exception as e:
        error_msg = f"Failed to clear temp files: {str(e)}"
        logger.error(f"[SYSTEM] {error_msg}")
        logger.error(f"[SYSTEM] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/system/status", response_model=SystemStatusResponse, tags=["System Utilities"])
async def get_system_status(x_api_key: str = Header(None)):
    """
    Get current system status including RAM, VRAM, and GPU information.
    
    Returns:
    - RAM: Total, used, available, percentage
    - VRAM: Total, used, available, percentage (GPU memory)
    - GPU: Name, utilization, temperature
    
    Useful for monitoring resources before/after heavy operations.
    """
    validate_api_key(x_api_key)
    
    logger.info("[SYSTEM] Status request")
    
    try:
        # Get system info from SystemMonitor
        system_info = SystemMonitor.get_system_info()
        
        logger.info(f"[SYSTEM] Status retrieved successfully")
        
        return SystemStatusResponse(
            success=True,
            message="System status retrieved successfully",
            ram=system_info.get("ram"),
            vram=system_info.get("vram"),
            gpu=system_info.get("gpu")
        )
        
    except Exception as e:
        error_msg = f"Failed to get system status: {str(e)}"
        logger.error(f"[SYSTEM] {error_msg}")
        logger.error(f"[SYSTEM] Traceback: {traceback.format_exc()}")
        
        # Return partial info on error
        return SystemStatusResponse(
            success=False,
            message=error_msg,
            ram=None,
            vram=None,
            gpu=None
        )


@app.post("/postprocess/pipeline/from-preset", response_model=PipelineResponse, tags=["Post-Processing"])
async def run_pipeline_from_preset(
    preset_name: str,
    video_url: Optional[str] = None,
    video_base64: Optional[str] = None,
    x_api_key: str = Header(None)
):
    """
    Run a processing pipeline using a saved workflow preset.
    
    This is a convenience endpoint that:
    1. Loads the specified workflow preset
    2. Builds operations from the preset's active_steps and params
    3. Executes the pipeline
    
    Provide either video_base64 or video_url (not both).
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[PIPELINE] Run from preset request: {preset_name}")
    
    try:
        # Refresh and get preset
        _initialize_workflow_presets()
        
        if preset_name not in tb_workflow_presets_data:
            raise HTTPException(status_code=404, detail=f"Workflow preset '{preset_name}' not found")
        
        preset = tb_workflow_presets_data[preset_name]
        active_steps = preset.get("active_steps", [])
        params = preset.get("params", {})
        
        if not active_steps:
            raise HTTPException(status_code=400, detail="Preset has no active steps to execute")
        
        # Build operations from preset
        operations = []
        
        # Map step names to operation types and build params
        step_mapping = {
            "Upscale": "upscale",
            "Frame Adjust": "interpolate", 
            "Loop": "loop",
            "Filters": "filters",
            "Export": "export"
        }
        
        for step in active_steps:
            op_type = step_mapping.get(step)
            if not op_type:
                logger.warning(f"[PIPELINE] Unknown step type: {step}")
                continue
            
            op_params = {}
            
            if op_type == "upscale":
                op_params = {
                    "model": params.get("upscale_model", "RealESRGAN_x2plus"),
                    "scale_factor": params.get("upscale_factor", 2.0),
                    "tile_size": params.get("tile_size", 0),
                    "enhance_face": params.get("enhance_face", False),
                    "denoise_strength": params.get("denoise_strength", 0.5),
                    "use_streaming": params.get("upscale_use_streaming", False)
                }
            elif op_type == "interpolate":
                op_params = {
                    "fps_mode": params.get("fps_mode", "No Interpolation"),
                    "speed_factor": params.get("speed_factor", 1.0),
                    "use_streaming": params.get("frames_use_streaming", False)
                }
            elif op_type == "loop":
                op_params = {
                    "loop_type": params.get("loop_type", "loop"),
                    "num_loops": params.get("num_loops", 1)
                }
            elif op_type == "filters":
                op_params = {
                    "brightness": params.get("brightness", 0.0),
                    "contrast": params.get("contrast", 1.0),
                    "saturation": params.get("saturation", 1.0),
                    "temperature": params.get("temperature", 0.0),
                    "sharpen": params.get("sharpen", 0.0),
                    "blur": params.get("blur", 0.0),
                    "denoise": params.get("denoise", 0.0),
                    "vignette": params.get("vignette", 0.0),
                    "s_curve_contrast": params.get("s_curve_contrast", 0.0),
                    "film_grain": params.get("film_grain_strength", 0.0)
                }
            elif op_type == "export":
                op_params = {
                    "format": params.get("export_format", "MP4"),
                    "quality": params.get("export_quality", 85),
                    "max_width": params.get("export_max_width", 1024)
                }
            
            operations.append(PipelineOperation(type=op_type, params=op_params))
        
        if not operations:
            raise HTTPException(status_code=400, detail="No valid operations could be built from preset")
        
        logger.info(f"[PIPELINE] Built {len(operations)} operations from preset")
        
        # Create pipeline request and call the existing pipeline endpoint
        pipeline_req = PipelineRequest(
            video_url=video_url,
            video_base64=video_base64,
            operations=operations
        )
        
        # Call the pipeline endpoint directly
        return await run_pipeline(pipeline_req, x_api_key)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to run pipeline from preset: {str(e)}"
        logger.error(f"[PIPELINE] {error_msg}")
        logger.error(f"[PIPELINE] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# PHASE 4: FILTER PRESETS, BATCH PROCESSING, MODEL MANAGEMENT
# ============================================================================

@app.get("/filters/presets", response_model=ListFilterPresetsResponse, tags=["Filter Presets"])
async def list_filter_presets(x_api_key: str = Header(None)):
    """
    List all available filter presets.
    
    Filter presets save filter slider values for quick application.
    Built-in presets include: none, cinematic, vintage, cool, warm, dramatic.
    User presets are also listed.
    """
    validate_api_key(x_api_key)
    
    logger.info("[FILTERS] List presets request")
    
    try:
        # Refresh presets from file
        _initialize_filter_presets()
        
        logger.info(f"[FILTERS] Found {len(tb_filter_presets_data)} filter presets")
        
        return ListFilterPresetsResponse(
            success=True,
            presets=tb_filter_presets_data,
            message=f"Found {len(tb_filter_presets_data)} filter preset(s)"
        )
        
    except Exception as e:
        error_msg = f"Failed to list filter presets: {str(e)}"
        logger.error(f"[FILTERS] {error_msg}")
        logger.error(f"[FILTERS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/filters/presets", response_model=FilterPresetResponse, tags=["Filter Presets"])
async def save_filter_preset(req: SaveFilterPresetRequest, x_api_key: str = Header(None)):
    """
    Save a new filter preset or update an existing one.
    
    Filter presets store values for all filter sliders:
    brightness, contrast, saturation, temperature, sharpen, blur, 
    denoise, vignette, s_curve_contrast, film_grain_strength.
    
    Note: The 'none' preset cannot be overwritten.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FILTERS] Save preset request: {req.name}")
    
    try:
        global tb_filter_presets_data
        
        clean_name = req.name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Preset name cannot be empty")
        
        if clean_name.lower() == "none":
            raise HTTPException(status_code=400, detail="'none' is a protected preset and cannot be overwritten")
        
        # Convert settings to dict
        preset_values = req.settings.model_dump()
        
        preset_existed = clean_name in tb_filter_presets_data
        tb_filter_presets_data[clean_name] = preset_values
        
        # Save to file
        with open(TB_BUILT_IN_PRESETS_FILE, 'w') as f:
            json.dump(tb_filter_presets_data, f, indent=4)
        
        action = "updated" if preset_existed else "saved"
        logger.info(f"[FILTERS] Preset '{clean_name}' {action} successfully")
        
        return FilterPresetResponse(
            success=True,
            message=f"Filter preset '{clean_name}' {action} successfully",
            preset_name=clean_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to save filter preset: {str(e)}"
        logger.error(f"[FILTERS] {error_msg}")
        logger.error(f"[FILTERS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete("/filters/presets/{name}", response_model=FilterPresetResponse, tags=["Filter Presets"])
async def delete_filter_preset(name: str, x_api_key: str = Header(None)):
    """
    Delete a filter preset by name.
    
    Note: The 'none' preset cannot be deleted.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[FILTERS] Delete preset request: {name}")
    
    try:
        global tb_filter_presets_data
        
        clean_name = name.strip()
        
        if clean_name.lower() == "none":
            raise HTTPException(status_code=400, detail="'none' preset cannot be deleted")
        
        if clean_name not in tb_filter_presets_data:
            raise HTTPException(status_code=404, detail=f"Filter preset '{clean_name}' not found")
        
        # Remove the preset
        del tb_filter_presets_data[clean_name]
        
        # Save to file
        with open(TB_BUILT_IN_PRESETS_FILE, 'w') as f:
            json.dump(tb_filter_presets_data, f, indent=4)
        
        logger.info(f"[FILTERS] Preset '{clean_name}' deleted successfully")
        
        return FilterPresetResponse(
            success=True,
            message=f"Filter preset '{clean_name}' deleted successfully",
            preset_name=clean_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to delete filter preset: {str(e)}"
        logger.error(f"[FILTERS] {error_msg}")
        logger.error(f"[FILTERS] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/postprocess/batch", response_model=BatchProcessingResponse, tags=["Post-Processing"])
async def batch_process_videos(req: BatchProcessingRequest, x_api_key: str = Header(None)):
    """
    Process multiple videos through the same pipeline.
    
    Each video in the batch will have the same operations applied.
    Results are returned for each video individually.
    
    This is useful for processing a collection of videos with the same settings.
    
    Note: This operation can be resource-intensive for large batches.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[BATCH] Batch processing request: {len(req.videos)} videos, {len(req.operations)} operations")
    
    results = []
    successful = 0
    failed = 0
    
    try:
        if tb_processor is None:
            logger.error("[BATCH] Toolbox processor not initialized")
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Validate operation types
        valid_ops = ["upscale", "interpolate", "filters", "loop", "export"]
        for op in req.operations:
            if op.type not in valid_ops:
                raise HTTPException(status_code=400, detail=f"Invalid operation type: {op.type}")
        
        # Process each video
        for i, video_item in enumerate(req.videos):
            logger.info(f"[BATCH] Processing video {i+1}/{len(req.videos)}")
            
            temp_video_path = None
            current_video_path = None
            intermediate_files = []
            
            try:
                # Get video source identifier
                has_base64 = video_item.video_base64 is not None and len(video_item.video_base64) > 0
                has_url = video_item.video_url is not None and len(video_item.video_url) > 0
                
                if not has_base64 and not has_url:
                    results.append(BatchVideoResult(
                        index=i,
                        success=False,
                        input_source="none",
                        error="No video source provided"
                    ))
                    failed += 1
                    continue
                
                input_source = "base64" if has_base64 else video_item.video_url[:50]
                
                # Get video to temp file
                if has_base64:
                    video_bytes = base64.b64decode(video_item.video_base64)
                    temp_video_path = os.path.join(api_output_dir, f"temp_batch_{uuid.uuid4().hex[:8]}.mp4")
                    with open(temp_video_path, 'wb') as f:
                        f.write(video_bytes)
                else:
                    temp_video_path = download_file_from_url(video_item.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
                
                current_video_path = temp_video_path
                
                # Apply each operation
                for op in req.operations:
                    output_path = None
                    
                    if op.type == "upscale":
                        params = op.params or {}
                        output_path = tb_processor.tb_upscale_video(
                            video_path=current_video_path,
                            model_key=params.get("model", "RealESRGAN_x2plus"),
                            output_scale_factor_ui=params.get("scale_factor", 2.0),
                            tile_size=params.get("tile_size", 512),
                            enhance_face=params.get("enhance_face", False),
                            denoise_strength_ui=params.get("denoise_strength", 0.5),
                            use_streaming=params.get("use_streaming", True),
                            progress=DummyProgress()
                        )
                    elif op.type == "interpolate":
                        params = op.params or {}
                        # Normalize fps_mode like the main interpolate endpoint does
                        raw_fps_mode = params.get("fps_mode", "2x")
                        fps_mode_mapping = {
                            "2x": "2x Frames",
                            "4x": "4x Frames",
                            "2x Frames": "2x Frames",
                            "4x Frames": "4x Frames",
                            "No Interpolation": "No Interpolation",
                            "none": "No Interpolation"
                        }
                        normalized_fps_mode = fps_mode_mapping.get(raw_fps_mode, "2x Frames")
                        output_path = tb_processor.tb_process_frames(
                            video_path=current_video_path,
                            target_fps_mode=normalized_fps_mode,
                            speed_factor=params.get("speed_factor", 1.0),
                            use_streaming=params.get("use_streaming", True),
                            progress=DummyProgress()
                        )
                    elif op.type == "filters":
                        params = op.params or {}
                        output_path = tb_processor.tb_apply_filters(
                            video_path=current_video_path,
                            brightness=params.get("brightness", 0.0),
                            contrast=params.get("contrast", 1.0),
                            saturation=params.get("saturation", 1.0),
                            temperature=params.get("temperature", 0.0),
                            sharpen=params.get("sharpen", 0.0),
                            blur=params.get("blur", 0.0),
                            denoise=params.get("denoise", 0.0),
                            vignette=params.get("vignette", 0.0),
                            s_curve_contrast=params.get("s_curve_contrast", 0.0),
                            film_grain_strength=params.get("film_grain", 0.0),
                            progress=DummyProgress()
                        )
                    elif op.type == "loop":
                        params = op.params or {}
                        output_path = tb_processor.tb_create_loop(
                            video_path=current_video_path,
                            loop_type=params.get("loop_type", "loop"),
                            num_loops=params.get("num_loops", 2),
                            progress=DummyProgress()
                        )
                    elif op.type == "export":
                        params = op.params or {}
                        output_path = tb_processor.tb_export_video(
                            video_path=current_video_path,
                            export_format=params.get("format", "MP4"),
                            quality_slider=params.get("quality", 85),
                            max_width=params.get("max_width") or 1920,
                            output_base_name_override=params.get("output_name"),
                            progress=DummyProgress()
                        )
                    
                    if output_path is None or not os.path.exists(output_path):
                        raise Exception(f"Operation {op.type} failed")
                    
                    # Track intermediate files
                    if current_video_path != temp_video_path and current_video_path != output_path:
                        intermediate_files.append(current_video_path)
                    
                    current_video_path = output_path
                
                # Read final output
                with open(current_video_path, 'rb') as f:
                    output_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                results.append(BatchVideoResult(
                    index=i,
                    success=True,
                    input_source=input_source,
                    output_video_base64=output_base64,
                    output_path=current_video_path
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"[BATCH] Video {i+1} failed: {str(e)}")
                results.append(BatchVideoResult(
                    index=i,
                    success=False,
                    input_source=input_source if 'input_source' in locals() else "unknown",
                    error=str(e)
                ))
                failed += 1
            
            finally:
                # Clean up temp and intermediate files
                if temp_video_path and has_base64 and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
                
                for path in intermediate_files:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
        
        logger.info(f"[BATCH] Batch complete: {successful}/{len(req.videos)} successful")
        
        # Send callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="batch",
                success=failed == 0,
                message=f"Batch processing complete: {successful} successful, {failed} failed",
                metadata={
                    "total_videos": len(req.videos),
                    "successful": successful,
                    "failed": failed,
                    "operations_count": len(req.operations),
                    "results_summary": [
                        {
                            "index": r.index,
                            "success": r.success,
                            "output_path": r.output_path,
                            "error": r.error
                        }
                        for r in results
                    ]
                }
            )
        
        return BatchProcessingResponse(
            success=failed == 0,
            message=f"Batch processing complete: {successful} successful, {failed} failed",
            total_videos=len(req.videos),
            successful=successful,
            failed=failed,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        logger.error(f"[BATCH] {error_msg}")
        logger.error(f"[BATCH] Traceback: {traceback.format_exc()}")
        
        # Send error callback if URL provided
        if req.callback_url:
            await send_postprocess_callback(
                callback_url=req.callback_url,
                operation="batch",
                success=False,
                error=error_msg,
                message="Batch processing failed",
                metadata={
                    "total_videos": len(req.videos) if req.videos else 0,
                    "operations_count": len(req.operations) if req.operations else 0
                }
            )
        
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/postprocess/save", response_model=SaveVideoResponse, tags=["Post-Processing"])
async def save_video_to_storage(req: SaveVideoRequest, x_api_key: str = Header(None)):
    """
    Save a video to permanent storage.
    
    Copies a video file to the permanent 'saved_videos' directory.
    Useful when autosave is disabled and you want to keep a processed video.
    
    Provide either video_base64 or video_url.
    """
    validate_api_key(x_api_key)
    
    logger.info("[SAVE] Save video to permanent storage request")
    
    # Validate input
    has_base64 = req.video_base64 is not None and len(req.video_base64) > 0
    has_url = req.video_url is not None and len(req.video_url) > 0
    
    if not has_base64 and not has_url:
        raise HTTPException(status_code=400, detail="Either video_base64 or video_url is required")
    
    temp_video_path = None
    
    try:
        if tb_processor is None:
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Get video to temp file
        if has_base64:
            video_bytes = base64.b64decode(req.video_base64)
            temp_video_path = os.path.join(api_output_dir, f"temp_save_{uuid.uuid4().hex[:8]}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        else:
            temp_video_path = download_file_from_url(req.video_url, ['.mp4', '.webm', '.avi', '.mov', '.mkv'])
        
        # If custom name provided, rename the file first
        if req.custom_name:
            custom_filename = req.custom_name.strip()
            if not custom_filename.lower().endswith(('.mp4', '.webm', '.gif')):
                custom_filename += os.path.splitext(temp_video_path)[1]
            new_temp_path = os.path.join(os.path.dirname(temp_video_path), custom_filename)
            os.rename(temp_video_path, new_temp_path)
            temp_video_path = new_temp_path
        
        # Copy to permanent storage
        saved_path = tb_processor.tb_copy_video_to_permanent_storage(temp_video_path)
        
        if saved_path is None:
            raise HTTPException(status_code=500, detail="Failed to save video to permanent storage")
        
        filename = os.path.basename(saved_path)
        
        logger.info(f"[SAVE] Video saved to: {saved_path}")
        
        return SaveVideoResponse(
            success=True,
            message="Video saved to permanent storage successfully",
            saved_path=saved_path,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to save video: {str(e)}"
        logger.error(f"[SAVE] {error_msg}")
        logger.error(f"[SAVE] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up temp file if we created one from base64
        if temp_video_path and has_base64 and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass


@app.get("/postprocess/autosave", response_model=AutosaveSettingResponse, tags=["Post-Processing"])
async def get_autosave_setting(x_api_key: str = Header(None)):
    """
    Get the current autosave setting.
    
    When autosave is enabled, processed videos are automatically 
    saved to the permanent 'saved_videos' directory.
    """
    validate_api_key(x_api_key)
    
    logger.info("[AUTOSAVE] Get autosave setting request")
    
    try:
        if tb_processor is None:
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Check if autosave is enabled by comparing output directories
        is_autosave_enabled = tb_processor.toolbox_video_output_dir == tb_processor._base_permanent_save_dir
        
        return AutosaveSettingResponse(
            success=True,
            message=f"Autosave is {'enabled' if is_autosave_enabled else 'disabled'}",
            autosave_enabled=is_autosave_enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to get autosave setting: {str(e)}"
        logger.error(f"[AUTOSAVE] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/postprocess/autosave", response_model=AutosaveSettingResponse, tags=["Post-Processing"])
async def set_autosave_setting(req: AutosaveSettingRequest, x_api_key: str = Header(None)):
    """
    Set the autosave mode.
    
    When enabled, all processed videos will be automatically saved to
    the permanent 'saved_videos' directory.
    
    When disabled, videos are saved to the temp directory and may be 
    cleaned up. Use /postprocess/save to manually save specific videos.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"[AUTOSAVE] Set autosave request: {req.enabled}")
    
    try:
        if tb_processor is None:
            raise HTTPException(status_code=500, detail="Post-processing module not available")
        
        # Persist setting to settings file (like Gradio's tb_handle_autosave_toggle)
        settings.set("toolbox_autosave_enabled", req.enabled)
        logger.info(f"[AUTOSAVE] Persisted autosave setting to settings file: {req.enabled}")
        
        # Apply the setting to the processor
        tb_processor.set_autosave_mode(req.enabled, silent=False)
        
        logger.info(f"[AUTOSAVE] Autosave {'enabled' if req.enabled else 'disabled'}")
        
        return AutosaveSettingResponse(
            success=True,
            message=f"Autosave {'enabled' if req.enabled else 'disabled'} successfully",
            autosave_enabled=req.enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to set autosave: {str(e)}"
        logger.error(f"[AUTOSAVE] {error_msg}")
        logger.error(f"[AUTOSAVE] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/model/unload", response_model=UnloadMainModelResponse, tags=["Model Management"])
async def unload_main_model(x_api_key: str = Header(None)):
    """
    Unload the main video generation transformer model to free VRAM.
    
    Use this before running heavy post-processing operations (like 4K upscaling)
    to maximize available GPU memory.
    
    The model will be reloaded automatically when the next generation job starts.
    
    Note: Cannot unload while a generation job is running.
    """
    validate_api_key(x_api_key)
    
    logger.info("[MODEL] Unload main model request")
    
    try:
        import gc
        
        # Check if there's a current generator to unload
        generator_to_unload = None
        model_name = "Unknown"
        
        # Try to find the current generator in the main module
        if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
            generator_to_unload = sys.modules['__main__'].current_generator
            logger.info("[MODEL] Found generator in __main__")
        elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'):
            generator_to_unload = sys.modules['studio'].current_generator
            logger.info("[MODEL] Found generator in studio module")
        
        if generator_to_unload is None:
            logger.info("[MODEL] No active generator found to unload")
            return UnloadMainModelResponse(
                success=True,
                message="No model currently loaded to unload",
                model_unloaded=None,
                vram_freed_estimate=None
            )
        
        # Check if a job is currently running
        job_queue = None
        if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'job_queue'):
            job_queue = sys.modules['__main__'].job_queue
        elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'job_queue'):
            job_queue = sys.modules['studio'].job_queue
        
        if job_queue is not None:
            current_job = getattr(job_queue, 'current_job', None)
            if current_job is not None:
                job_status = getattr(current_job, 'status', None)
                if job_status is not None and hasattr(job_status, 'value') and job_status.value == 'running':
                    raise HTTPException(
                        status_code=409,
                        detail="Cannot unload model: A video generation job is currently running"
                    )
        
        # Get model name before unloading
        try:
            if hasattr(generator_to_unload, 'get_model_name') and callable(generator_to_unload.get_model_name):
                model_name = generator_to_unload.get_model_name()
            elif hasattr(generator_to_unload, 'transformer') and generator_to_unload.transformer is not None:
                model_name = generator_to_unload.transformer.__class__.__name__
            else:
                model_name = generator_to_unload.__class__.__name__
        except:
            pass
        
        # Unload the model (matching Gradio's tb_handle_delete_studio_transformer)
        logger.info(f"[MODEL] Unloading model: {model_name}")
        
        # Step 1: Unload LoRAs first if available
        if hasattr(generator_to_unload, 'unload_loras') and callable(generator_to_unload.unload_loras):
            logger.info("[MODEL] Unloading LoRAs from transformer...")
            try:
                generator_to_unload.unload_loras()
                logger.info("[MODEL] LoRAs unloaded successfully")
            except Exception as e_lora:
                logger.warning(f"[MODEL] LoRA unload failed (non-critical): {e_lora}")
        
        # Step 2: Move transformer to CPU before deletion (safer memory cleanup)
        if hasattr(generator_to_unload, 'transformer') and generator_to_unload.transformer is not None:
            transformer_ref = generator_to_unload.transformer
            transformer_name = transformer_ref.__class__.__name__
            
            # Move to CPU first for cleaner VRAM release
            if hasattr(transformer_ref, 'to') and callable(transformer_ref.to):
                try:
                    logger.info(f"[MODEL] Moving transformer ({transformer_name}) to CPU...")
                    cpu_device = torch.device('cpu')
                    transformer_ref.to(cpu_device)
                    logger.info(f"[MODEL] Transformer moved to CPU successfully")
                except Exception as e_cpu:
                    logger.warning(f"[MODEL] Failed to move to CPU (non-critical): {e_cpu}")
            
            # Now delete the transformer
            logger.info(f"[MODEL] Deleting transformer reference...")
            generator_to_unload.transformer = None
            del transformer_ref
            logger.info(f"[MODEL] Transformer deleted")
        
        # Clear from module
        if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
            sys.modules['__main__'].current_generator = None
        if 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'):
            sys.modules['studio'].current_generator = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"[MODEL] Model '{model_name}' unloaded successfully")
        
        return UnloadMainModelResponse(
            success=True,
            message=f"Model '{model_name}' unloaded successfully. VRAM freed.",
            model_unloaded=model_name,
            vram_freed_estimate="~12-24 GB (varies by model)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to unload model: {str(e)}"
        logger.error(f"[MODEL] {error_msg}")
        logger.error(f"[MODEL] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/model/status", tags=["Model Management"])
async def get_model_status(x_api_key: str = Header(None)):
    """
    Get the status of the main video generation model.
    
    Returns whether a model is currently loaded and its name.
    """
    validate_api_key(x_api_key)
    
    logger.info("[MODEL] Get model status request")
    
    try:
        generator = None
        model_name = None
        model_loaded = False
        
        # Try to find the current generator
        if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
            generator = sys.modules['__main__'].current_generator
        elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'):
            generator = sys.modules['studio'].current_generator
        
        if generator is not None:
            model_loaded = True
            try:
                if hasattr(generator, 'get_model_name') and callable(generator.get_model_name):
                    model_name = generator.get_model_name()
                elif hasattr(generator, 'transformer') and generator.transformer is not None:
                    model_name = generator.transformer.__class__.__name__
                else:
                    model_name = generator.__class__.__name__
            except:
                model_name = "Unknown"
        
        return {
            "model_loaded": model_loaded,
            "model_name": model_name,
            "vram_usage": get_cuda_free_memory_gb() if torch.cuda.is_available() else None
        }
        
    except Exception as e:
        error_msg = f"Failed to get model status: {str(e)}"
        logger.error(f"[MODEL] {error_msg}")
        return {
            "model_loaded": False,
            "model_name": None,
            "error": error_msg
        }


@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_video(req: GenerationRequest, x_api_key: str = Header(None)):
    """
    Submit a video generation job.
    
    Supports:
    - Text-to-Video (T2V): Just provide a prompt
    - Image-to-Video (I2V): Provide input_image_base64 or input_image_url
    - End Frame: Provide end_frame_image_base64 (for Endframe models)
    
    Returns immediately with job_id for status polling.
    """
    validate_api_key(x_api_key)
    
    logger.info(f"New generation request: model={req.model_type}, prompt={req.prompt[:50]}...")
    
    # NOTE: We pass raw resolution values to the worker, just like Gradio does.
    # The pipeline's preprocess_inputs() handles bucket calculation based on:
    # - If has_input_image: uses image dimensions + resolutionW as bucket tier
    # - If no image: uses resolutionH, resolutionW + average as bucket tier
    # This ensures consistent behavior between API and Gradio.
    
    # Log what bucket will likely be used (for informational purposes only)
    estimated_bucket_h, estimated_bucket_w = apply_bucket_resolution(req.resolution_w, req.resolution_h)
    logger.info(f"Resolution requested: {req.resolution_w}x{req.resolution_h} -> estimated bucket: {estimated_bucket_w}x{estimated_bucket_h}")
    
    # Process input image if provided
    input_image = None
    has_input_image = False
    input_image_path = None
    
    if req.input_image_base64:
        input_image = decode_base64_image(req.input_image_base64)
        has_input_image = True
        logger.info(f"Received base64 input image: {input_image.shape}")
    elif req.input_image_url:
        input_image = await download_image_from_url(req.input_image_url)
        has_input_image = True
        logger.info(f"Downloaded input image from URL: {input_image.shape}")
    
    # If no input image, create latent background for T2V using RAW resolution
    # The pipeline will resize this to the correct bucket size
    if input_image is None:
        input_image = create_latent_image(req.resolution_w, req.resolution_h, req.latent_type)
        logger.info(f"Created {req.latent_type.value} latent image for T2V: {req.resolution_w}x{req.resolution_h}")
    
    # Process end frame image if provided
    end_frame_image = None
    end_frame_image_path = None
    
    if req.end_frame_image_base64:
        end_frame_image = decode_base64_image(req.end_frame_image_base64)
        logger.info(f"Received end frame image: {end_frame_image.shape}")
    
    # Validate model type for end frame
    if end_frame_image is not None and req.model_type not in [ModelType.ORIGINAL_WITH_ENDFRAME, ModelType.VIDEO]:
        logger.warning(f"End frame provided but model type is {req.model_type}. End frame will be ignored.")
    
    # Process LoRA selection
    selected_loras = []
    lora_values_list = []
    
    if req.loras:
        for lora_name, weight in req.loras.items():
            if lora_name in lora_names:
                selected_loras.append(lora_name)
                lora_values_list.append(weight)
            else:
                logger.warning(f"LoRA '{lora_name}' not found, skipping")
    
    # Determine cache settings
    use_teacache = req.cache_type == CacheType.TEACACHE
    use_magcache = req.cache_type == CacheType.MAGCACHE
    
    # Build job parameters matching worker.py signature exactly
    job_params = {
        # Model type
        'model_type': req.model_type.value,
        
        # Image inputs
        'input_image': input_image.copy() if input_image is not None else None,
        'has_input_image': has_input_image,
        'latent_type': req.latent_type.value,
        
        # End frame
        'end_frame_image': end_frame_image.copy() if end_frame_image is not None else None,
        'end_frame_strength': req.end_frame_strength,
        
        # Prompts
        'prompt_text': req.prompt,
        'n_prompt': req.negative_prompt,
        
        # Generation specs
        'seed': req.seed,
        'total_second_length': req.total_second_length,
        'latent_window_size': req.latent_window_size,
        'steps': req.steps,
        
        # CFG settings
        'cfg': req.cfg,
        'gs': req.gs,
        'rs': req.rs,
        
        # Resolution (raw values - pipeline handles bucket calculation, matching Gradio behavior)
        'resolutionW': req.resolution_w,
        'resolutionH': req.resolution_h,
        
        # Cache settings - TeaCache
        'use_teacache': use_teacache,
        'teacache_num_steps': req.teacache_num_steps,
        'teacache_rel_l1_thresh': req.teacache_rel_l1_thresh,
        
        # Cache settings - MagCache
        'use_magcache': use_magcache,
        'magcache_threshold': req.magcache_threshold,
        'magcache_max_consecutive_skips': req.magcache_max_consecutive_skips,
        'magcache_retention_ratio': req.magcache_retention_ratio,
        
        # Prompt blending
        'blend_sections': req.blend_sections,
        
        # LoRA
        'selected_loras': selected_loras,
        'lora_values': lora_values_list,
        'lora_loaded_names': lora_names,
        
        # Paths
        'output_dir': settings.get("output_dir"),
        'metadata_dir': settings.get("metadata_dir"),
        'input_files_dir': settings.get("input_files_dir"),
        'input_image_path': input_image_path,
        'end_frame_image_path': end_frame_image_path,
        
        # Video model specifics
        'input_video': req.input_video_url,
        'combine_with_source': req.combine_with_source,
        'num_cleaned_frames': req.num_cleaned_frames,
        
        # Metadata
        'save_metadata_checked': req.save_metadata,
        
        # Callback/Webhook settings
        'callback_url': req.callback_url,
        'callback_token': req.callback_token,
    }
    
    # Add to queue
    job_id = job_queue.add_job(job_params)
    logger.info(f"Job {job_id} added to queue with seed {req.seed}")
    
    # Initialize progress tracking
    with progress_lock:
        job_progress_store[job_id] = {
            "progress_percent": 0,
            "progress_desc": "Queued",
            "updated_at": time.time()
        }
    
    # Estimate generation time
    estimated_time = estimate_generation_time(req)
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        seed=req.seed,
        resolution={"width": estimated_bucket_w, "height": estimated_bucket_h},
        estimated_time_seconds=estimated_time,
        message=f"Job queued successfully. Requested: {req.resolution_w}x{req.resolution_h}, estimated bucket: {estimated_bucket_w}x{estimated_bucket_h}. Time: ~{estimated_time}s"
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Generation"])
async def get_job_status(job_id: str, x_api_key: str = Header(None)):
    """
    Check the status of a generation job.
    
    Returns current status, progress, and result URL when complete.
    """
    validate_api_key(x_api_key)
    
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    response = JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )
    
    # Get progress from store
    with progress_lock:
        if job_id in job_progress_store:
            progress_data = job_progress_store[job_id]
            response.progress_percent = progress_data.get("progress_percent")
            response.progress_desc = progress_data.get("progress_desc")
            response.eta_seconds = progress_data.get("eta_seconds")
    
    # Get progress from job object if available
    if job.progress_data:
        response.progress_desc = job.progress_data.get("desc", response.progress_desc)
    
    # Get result URL if completed
    if job.status == JobStatus.COMPLETED and job.result:
        filename = os.path.basename(job.result)
        response.result_url = f"/outputs/{filename}"
        response.progress_percent = 100
        response.progress_desc = "Completed"
    
    # Get error if failed
    if job.status == JobStatus.FAILED:
        response.error = job.error
    
    return response


@app.post("/cancel/{job_id}", tags=["Generation"])
async def cancel_job(job_id: str, x_api_key: str = Header(None)):
    """
    Cancel a pending or running job.
    """
    validate_api_key(x_api_key)
    
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
    
    if job.status == JobStatus.CANCELLED:
        return {"job_id": job_id, "status": "already_cancelled"}
    
    # Send cancel signal
    with job_queue.lock:
        if job_queue.current_job and job_queue.current_job.id == job_id:
            if job_queue.current_job.stream:
                job_queue.current_job.stream.input_queue.push('end')
            job_queue.current_job.status = JobStatus.CANCELLED
            job_queue.current_job.completed_at = time.time()
            logger.info(f"Cancelled running job {job_id}")
        elif job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            logger.info(f"Cancelled pending job {job_id}")
    
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/queue", tags=["Queue"])
async def get_queue_status(x_api_key: str = Header(None)):
    """
    Get current queue status and all jobs.
    """
    validate_api_key(x_api_key)
    
    jobs = job_queue.get_all_jobs()
    
    job_list = []
    for job in jobs:
        job_info = {
            "id": job.id,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "model_type": job.params.get('model_type'),
            "prompt": job.params.get('prompt_text', '')[:100] + "..." if len(job.params.get('prompt_text', '')) > 100 else job.params.get('prompt_text', ''),
        }
        
        if job.status == JobStatus.PENDING:
            job_info["queue_position"] = job_queue.get_queue_position(job.id)
        
        if job.status == JobStatus.COMPLETED and job.result:
            job_info["result_url"] = f"/outputs/{os.path.basename(job.result)}"
        
        if job.error:
            job_info["error"] = job.error
        
        job_list.append(job_info)
    
    # Count by status
    pending = sum(1 for j in jobs if j.status == JobStatus.PENDING)
    running = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
    completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
    
    return {
        "summary": {
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": len(jobs)
        },
        "jobs": job_list
    }


@app.delete("/queue/completed", tags=["Queue"])
async def clear_completed_jobs(x_api_key: str = Header(None)):
    """
    Clear all completed and cancelled jobs from the queue.
    """
    validate_api_key(x_api_key)
    
    removed_count = job_queue.clear_completed_jobs()
    logger.info(f"Cleared {removed_count} completed jobs")
    
    return {"removed_count": removed_count, "message": f"Removed {removed_count} completed/cancelled jobs"}


# --- 17. WEBSOCKET FOR PROGRESS STREAMING ---

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time progress streaming.
    
    Connect to receive progress updates for a specific job.
    """
    await websocket.accept()
    
    # Validate job exists
    job = job_queue.get_job(job_id)
    if not job:
        await websocket.send_json({"error": f"Job {job_id} not found"})
        await websocket.close()
        return
    
    # Add to connections
    websocket_connections[job_id].append(websocket)
    logger.info(f"WebSocket connected for job {job_id}")
    
    try:
        last_update = None
        while True:
            # Check job status
            job = job_queue.get_job(job_id)
            if not job:
                await websocket.send_json({"status": "not_found"})
                break
            
            # Build progress update
            update = {
                "job_id": job_id,
                "status": job.status.value,
                "timestamp": time.time()
            }
            
            # Add progress data
            if job.progress_data:
                update["progress_desc"] = job.progress_data.get("desc", "")
                # Note: preview images are too large for WebSocket, skip them
            
            # Add result if completed
            if job.status == JobStatus.COMPLETED and job.result:
                update["result_url"] = f"/outputs/{os.path.basename(job.result)}"
                await websocket.send_json(update)
                break
            
            if job.status == JobStatus.FAILED:
                update["error"] = job.error
                await websocket.send_json(update)
                break
            
            if job.status == JobStatus.CANCELLED:
                await websocket.send_json(update)
                break
            
            # Only send if changed
            if update != last_update:
                await websocket.send_json(update)
                last_update = update
            
            await asyncio.sleep(0.5)  # Poll every 500ms
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        # Remove from connections
        if websocket in websocket_connections[job_id]:
            websocket_connections[job_id].remove(websocket)


# --- 18. SSE ENDPOINT FOR PROGRESS ---

@app.get("/stream/progress/{job_id}", tags=["Generation"])
async def stream_progress(job_id: str, x_api_key: str = Header(None)):
    """
    Server-Sent Events endpoint for progress streaming.
    
    Alternative to WebSocket for simpler clients.
    """
    validate_api_key(x_api_key)
    
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    async def event_generator():
        last_update = None
        while True:
            job = job_queue.get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break
            
            update = {
                "job_id": job_id,
                "status": job.status.value,
                "timestamp": time.time()
            }
            
            if job.progress_data:
                update["progress_desc"] = job.progress_data.get("desc", "")
            
            if job.status == JobStatus.COMPLETED and job.result:
                update["result_url"] = f"/outputs/{os.path.basename(job.result)}"
                yield f"data: {json.dumps(update)}\n\n"
                break
            
            if job.status in [JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.error:
                    update["error"] = job.error
                yield f"data: {json.dumps(update)}\n\n"
                break
            
            if update != last_update:
                yield f"data: {json.dumps(update)}\n\n"
                last_update = update
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# --- 19. CALLBACK/WEBHOOK SYSTEM ---

async def send_job_callback(job_id: str, job: Any, base_url: str = "http://localhost:8000"):
    """
    Send a callback/webhook POST request when a job completes.
    This notifies external services (like Supabase Edge Functions) about job completion.
    """
    callback_url = job.params.get('callback_url')
    if not callback_url:
        logger.debug(f"No callback_url for job {job_id}, skipping callback")
        return
    
    logger.info(f"Sending callback for job {job_id} to {callback_url}")
    
    try:
        # Build callback payload
        payload = {
            "job_id": job_id,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": job.error,
        }
        
        # Add result info if job completed successfully
        if job.status == JobStatus.COMPLETED and job.result:
            filename = os.path.basename(job.result)
            payload["result_url"] = f"/outputs/{filename}"
            payload["video_download_url"] = f"{base_url}/outputs/{filename}"
            payload["video_filename"] = filename
            payload["video_local_path"] = job.result
            
            # Add file size if available
            try:
                payload["video_file_size"] = os.path.getsize(job.result)
            except:
                pass
        
        # Add generation metadata
        payload["metadata"] = {
            "model_type": job.params.get('model_type'),
            "prompt": job.params.get('prompt_text'),
            "negative_prompt": job.params.get('n_prompt'),
            "seed": job.params.get('seed'),
            "steps": job.params.get('steps'),
            "cfg": job.params.get('cfg'),
            "gs": job.params.get('gs'),
            "resolution_w": job.params.get('resolutionW'),
            "resolution_h": job.params.get('resolutionH'),
            "total_second_length": job.params.get('total_second_length'),
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Job-ID": job_id,
        }
        
        # Add authorization token if provided
        callback_token = job.params.get('callback_token')
        if callback_token:
            headers["Authorization"] = f"Bearer {callback_token}"
        
        # Send the callback
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()
                if response.status >= 200 and response.status < 300:
                    logger.info(f"Callback successful for job {job_id}: {response.status}")
                    logger.debug(f"Callback response: {response_text}")
                else:
                    logger.warning(f"Callback failed for job {job_id}: {response.status} - {response_text}")
                    
    except asyncio.TimeoutError:
        logger.error(f"Callback timeout for job {job_id} to {callback_url}")
    except Exception as e:
        logger.error(f"Callback error for job {job_id}: {str(e)}")


async def send_postprocess_callback(
    callback_url: str,
    operation_type: str = None,
    success: bool = True,
    output_path: Optional[str] = None,
    output_base64: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    # Alternative parameter names for backward compatibility
    operation: str = None,
    output_video_base64: Optional[str] = None,
    message: Optional[str] = None
):
    """
    Send a callback/webhook POST request when a post-processing operation completes.
    Used by all Phase 1-4 endpoints that support callbacks.
    
    Args:
        callback_url: URL to POST results to
        operation_type: Type of operation (upscale, interpolate, filters, etc.) - also accepts 'operation' alias
        success: Whether the operation succeeded
        output_path: Local path to the output file
        output_base64: Base64-encoded output (optional, can be large) - also accepts 'output_video_base64' alias
        error: Error message if operation failed
        metadata: Additional operation-specific metadata
        message: Optional message (will be added to metadata if provided)
    """
    # Handle alternative parameter names for backward compatibility
    if operation_type is None and operation is not None:
        operation_type = operation
    if output_base64 is None and output_video_base64 is not None:
        output_base64 = output_video_base64
    
    # Add message to metadata if provided
    if message is not None:
        if metadata is None:
            metadata = {}
        metadata["message"] = message
    
    if not callback_url:
        return
    
    if not operation_type:
        logger.error("[CALLBACK] Cannot send callback without operation_type")
        return
    
    logger.info(f"[CALLBACK] Sending post-process callback for '{operation_type}' to {callback_url}")
    
    try:
        # Build callback payload
        payload = {
            "type": "postprocess",
            "operation": operation_type,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        
        if success and output_path:
            filename = os.path.basename(output_path)
            payload["output_filename"] = filename
            payload["output_path"] = output_path
            
            # Calculate the correct relative URL path from the outputs directory
            # The output_path might be in a subdirectory like outputs/postprocessed_output/temp_processing/
            try:
                # Get absolute paths for comparison
                abs_output_path = os.path.abspath(output_path)
                abs_output_dir = os.path.abspath(api_output_dir)
                
                # Calculate relative path from outputs directory
                if abs_output_path.startswith(abs_output_dir):
                    relative_path = os.path.relpath(abs_output_path, abs_output_dir)
                    # Convert Windows backslashes to forward slashes for URLs
                    relative_path = relative_path.replace("\\", "/")
                    payload["output_url"] = f"{BASE_URL}/outputs/{relative_path}"
                else:
                    # File is outside outputs directory, just use filename
                    payload["output_url"] = f"{BASE_URL}/outputs/{filename}"
                    
                logger.debug(f"[CALLBACK] output_path: {output_path}")
                logger.debug(f"[CALLBACK] output_url: {payload['output_url']}")
            except Exception as path_err:
                logger.warning(f"[CALLBACK] Could not calculate relative path: {path_err}, using filename only")
                payload["output_url"] = f"{BASE_URL}/outputs/{filename}"
            
            # Add file size if available
            try:
                payload["output_file_size"] = os.path.getsize(output_path)
            except:
                pass
            
            # Include base64 output if provided (can be large!)
            if output_base64:
                payload["output_base64"] = output_base64
        
        if error:
            payload["error"] = error
        
        if metadata:
            payload["metadata"] = metadata
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Operation-Type": operation_type,
        }
        
        # Send the callback
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for potentially large payloads
            ) as response:
                response_text = await response.text()
                if response.status >= 200 and response.status < 300:
                    logger.info(f"[CALLBACK] Post-process callback successful for '{operation_type}': {response.status}")
                    logger.debug(f"[CALLBACK] Response: {response_text[:200]}...")
                else:
                    logger.warning(f"[CALLBACK] Post-process callback failed for '{operation_type}': {response.status} - {response_text}")
                    
    except asyncio.TimeoutError:
        logger.error(f"[CALLBACK] Timeout sending callback for '{operation_type}' to {callback_url}")
    except Exception as e:
        logger.error(f"[CALLBACK] Error sending callback for '{operation_type}': {str(e)}")


# Background task to check for completed jobs and send callbacks
async def process_job_callbacks():
    """
    Background task that monitors for completed jobs and sends callbacks.
    This runs periodically to catch any jobs that completed.
    """
    processed_callbacks = set()  # Track jobs we've already sent callbacks for
    
    while True:
        try:
            jobs = job_queue.get_all_jobs()
            for job in jobs:
                # Only process completed/failed jobs with callback_url that we haven't processed yet
                if job.id not in processed_callbacks:
                    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and job.params.get('callback_url'):
                        await send_job_callback(job.id, job, base_url=BASE_URL)
                        processed_callbacks.add(job.id)
            
            # Clean up old entries from processed_callbacks (keep only recent ones)
            if len(processed_callbacks) > 1000:
                # Get current job IDs
                current_job_ids = {job.id for job in jobs}
                # Remove IDs that are no longer in the queue
                processed_callbacks = processed_callbacks.intersection(current_job_ids)
                
        except Exception as e:
            logger.error(f"Error in callback processor: {e}")
        
        await asyncio.sleep(2)  # Check every 2 seconds


# --- 20. DELETE/CLEANUP ENDPOINTS ---

@app.delete("/outputs/{filename}", tags=["Cleanup"])
async def delete_output_file(filename: str, x_api_key: str = Header(None)):
    """
    Delete a video file from the outputs directory.
    Use this after successfully uploading to external storage (e.g., Supabase).
    
    This endpoint is typically called by your webhook handler after it has:
    1. Received the callback notification
    2. Downloaded and uploaded the video to Supabase Storage
    3. Updated the database record
    """
    validate_api_key(x_api_key)
    
    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(api_output_dir, safe_filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {safe_filename} not found")
    
    # Verify the file is within our outputs directory (security check)
    real_path = os.path.realpath(file_path)
    real_output_dir = os.path.realpath(api_output_dir)
    if not real_path.startswith(real_output_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        os.remove(file_path)
        logger.info(f"Deleted output file: {safe_filename}")
        
        # Also try to delete associated metadata file if it exists
        metadata_file = file_path.replace('.mp4', '.json')
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            logger.info(f"Deleted metadata file: {os.path.basename(metadata_file)}")
        
        return {
            "success": True,
            "message": f"File {safe_filename} deleted successfully",
            "deleted_files": [safe_filename]
        }
    except Exception as e:
        logger.error(f"Error deleting file {safe_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.post("/cleanup/job/{job_id}", tags=["Cleanup"])
async def cleanup_job_files(job_id: str, x_api_key: str = Header(None)):
    """
    Clean up all files associated with a specific job.
    Deletes the output video and any intermediate files.
    """
    validate_api_key(x_api_key)
    
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    deleted_files = []
    errors = []
    
    # Delete main result file
    if job.result and os.path.exists(job.result):
        try:
            os.remove(job.result)
            deleted_files.append(os.path.basename(job.result))
            logger.info(f"Deleted job result: {job.result}")
        except Exception as e:
            errors.append(f"Error deleting {job.result}: {str(e)}")
    
    # Find and delete any intermediate files for this job
    try:
        for filename in os.listdir(api_output_dir):
            if filename.startswith(f"{job_id}_") or filename.startswith(job_id):
                file_path = os.path.join(api_output_dir, filename)
                try:
                    os.remove(file_path)
                    deleted_files.append(filename)
                    logger.info(f"Deleted intermediate file: {filename}")
                except Exception as e:
                    errors.append(f"Error deleting {filename}: {str(e)}")
    except Exception as e:
        errors.append(f"Error scanning output directory: {str(e)}")
    
    return {
        "success": len(errors) == 0,
        "job_id": job_id,
        "deleted_files": deleted_files,
        "errors": errors if errors else None
    }


# --- 21. STARTUP EVENT ---

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 50)
    logger.info("FramePack Studio API Starting")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
    logger.info(f"VRAM: {free_mem_gb:.2f} GB free")
    logger.info(f"High VRAM Mode: {high_vram}")
    logger.info(f"Output Directory: {api_output_dir}")
    logger.info(f"LoRAs Available: {len([l for l in lora_names if l != DUMMY_LORA_NAME])}")
    logger.info("=" * 50)
    
    # Start the callback processor background task
    asyncio.create_task(process_job_callbacks())
    logger.info("Callback processor started")


# --- 22. MAIN ---

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("FramePack Studio API Server")
    print("=" * 50)
    print(f"Docs: http://0.0.0.0:8000/docs")
    print(f"Health: http://0.0.0.0:8000/health")
    print("=" * 50 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
