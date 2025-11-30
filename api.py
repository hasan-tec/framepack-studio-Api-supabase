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
from pathlib import PurePath, Path
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import threading
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
high_vram = free_mem_gb > 60
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


def apply_bucket_resolution(width: int, height: int) -> tuple:
    """Apply bucket resolution system like Gradio does"""
    try:
        bucket_width, bucket_height = find_nearest_bucket(width, height, resolution=640)
        logger.info(f"Resolution adjusted: {width}x{height} -> {bucket_width}x{bucket_height}")
        return bucket_width, bucket_height
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
    
    # Apply bucket resolution
    bucket_w, bucket_h = apply_bucket_resolution(req.resolution_w, req.resolution_h)
    
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
    
    # If no input image, create latent background for T2V
    if input_image is None:
        input_image = create_latent_image(bucket_w, bucket_h, req.latent_type)
        logger.info(f"Created {req.latent_type.value} latent image for T2V")
    
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
        
        # Resolution (bucket adjusted)
        'resolutionW': bucket_w,
        'resolutionH': bucket_h,
        
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
        resolution={"width": bucket_w, "height": bucket_h},
        estimated_time_seconds=estimated_time,
        message=f"Job queued successfully. Estimated time: {estimated_time}s"
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


# --- 19. STARTUP EVENT ---

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


# --- 20. MAIN ---

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
