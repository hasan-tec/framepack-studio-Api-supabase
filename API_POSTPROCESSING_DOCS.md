# FramePack Studio API - Post-Processing & Utilities Documentation

> **Complete API Reference for Phases 1-4: AI Features, Post-Processing, Frames Studio, and System Utilities**

---

## Table of Contents

1. [Authentication](#authentication)
2. [Webhook Callbacks](#webhook-callbacks)
3. [Phase 1: AI Features](#phase-1-ai-features)
   - [Prompt Enhancement](#post-enhance-prompt)
   - [Image Captioning](#post-caption-image)
4. [Phase 2: Core Post-Processing](#phase-2-core-post-processing)
   - [Video Upscaling](#post-postprocessupscale)
   - [Frame Interpolation](#post-postprocessinterpolate)
   - [Video Filters](#post-postprocessfilters)
   - [Video Looping](#post-postprocessloop)
   - [Video Joining](#post-postprocessjoin)
   - [Video Export](#post-postprocessexport)
   - [Processing Pipeline](#post-postprocesspipeline)
   - [Video Analysis](#post-postprocessanalyze)
   - [List Upscale Models](#get-postprocessmodels)
5. [Phase 3: Frames Studio & Workflow Presets](#phase-3-frames-studio--workflow-presets)
   - [Extract Frames](#post-framesextract)
   - [List Frame Folders](#get-framesfolders)
   - [List Frames in Folder](#get-framesfolder)
   - [Delete Frame Folder](#delete-framesfolder)
   - [Delete Single Frame](#delete-framesfolderframe)
   - [Save Single Frame](#post-framesfoldersaveframe)
   - [Reassemble Frames](#post-framesreassemble)
   - [List Workflow Presets](#get-workflowpresets)
   - [Save Workflow Preset](#post-workflowpresets)
   - [Delete Workflow Preset](#delete-workflowpresetsname)
   - [Run Pipeline from Preset](#post-postprocesspipelinefrom-preset)
   - [Clear Temp Files](#post-systemclear-temp)
   - [System Status](#get-systemstatus)
6. [Phase 4: Filter Presets, Batch Processing & Model Management](#phase-4-filter-presets-batch-processing--model-management)
   - [List Filter Presets](#get-filterspresets)
   - [Save Filter Preset](#post-filterspresets)
   - [Delete Filter Preset](#delete-filterspresetsname)
   - [Batch Process Videos](#post-postprocessbatch)
   - [Save Video](#post-postprocesssave)
   - [Get Autosave Setting](#get-postprocessautosave)
   - [Set Autosave Setting](#post-postprocessautosave)
   - [Unload Main Model](#post-modelunload)
   - [Model Status](#get-modelstatus)

---

## Authentication

All endpoints require API key authentication via the `X-API-Key` header.

```bash
curl -X POST "http://localhost:8000/enhance-prompt" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat"}'
```

---

## Webhook Callbacks

All post-processing endpoints support optional webhook callbacks. When a `callback_url` is provided, the API will POST the results to your specified URL when the operation completes (success or failure).

### Callback Parameter

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `callback_url` | string | ❌ | URL to receive webhook notification when processing completes |

### Callback Payload Format

When the operation completes, a JSON payload is sent to your callback URL:

```json
{
  "type": "postprocess",
  "operation": "upscale",
  "success": true,
  "timestamp": "2024-12-04T14:30:22.123456",
  "output_path": "/outputs/postprocessed_output/video_upscaled_241204.mp4",
  "output_filename": "video_upscaled_241204.mp4",
  "output_url": "http://localhost:8000/outputs/video_upscaled_241204.mp4",
  "output_file_size": 15234567,
  "output_base64": "AAAAIGZ0eXBpc29t...",
  "metadata": {
    "model": "RealESRGAN_x4plus",
    "scale_factor": 2.0,
    "enhance_face": true
  }
}
```

### Callback Payload Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"postprocess"` |
| `operation` | string | Operation name (e.g., `upscale`, `interpolate`, `filters`, etc.) |
| `success` | boolean | `true` or `false` |
| `timestamp` | string | ISO format timestamp |
| `output_path` | string | Local path to output file (on success) |
| `output_filename` | string | Output filename (on success) |
| `output_url` | string | URL to download the output file (on success) |
| `output_file_size` | int | Output file size in bytes (on success) |
| `output_base64` | string | Base64-encoded output video (on success, if provided) |
| `error` | string | Error details (only on failure) |
| `metadata` | object | Operation-specific parameters used |

### Error Callback Example

```json
{
  "type": "postprocess",
  "operation": "upscale",
  "success": false,
  "timestamp": "2024-12-04T14:30:22.123456",
  "error": "Model failed to load: insufficient VRAM",
  "metadata": null
}
```

### Supported Callback Endpoints

The following endpoints support `callback_url`:

| Endpoint | Operation Name |
|----------|----------------|
| `POST /postprocess/upscale` | `upscale` |
| `POST /postprocess/interpolate` | `interpolate` |
| `POST /postprocess/filters` | `filters` |
| `POST /postprocess/loop` | `loop` |
| `POST /postprocess/join` | `join` |
| `POST /postprocess/export` | `export` |
| `POST /postprocess/pipeline` | `pipeline` |
| `POST /postprocess/batch` | `batch` |
| `POST /frames/extract` | `extract_frames` |
| `POST /frames/reassemble` | `reassemble_frames` |

### Example with Callback

```json
{
  "video_url": "https://example.com/video.mp4",
  "model": "RealESRGAN_x4plus",
  "callback_url": "https://myapp.com/webhooks/video-processed"
}
```

---

## Async Background Processing Mode (Prevents Timeouts)

**⚠️ IMPORTANT FOR PRODUCTION:** When using a reverse proxy like Cloudflare, Nginx, or RunPod's proxy, you may encounter **Error 524 (timeout)** for long-running operations. This happens because proxies typically timeout after 60-100 seconds, while video processing can take 5-30+ minutes.

### Solution: Use Callback URL

When you provide a `callback_url`, the API switches to **async background processing mode**:

1. **Immediate Response**: API returns immediately (within seconds) with a "processing started" message
2. **Background Processing**: The operation runs in a background thread
3. **Callback Delivery**: Results are sent to your callback URL when complete

### Async Response Format

When `callback_url` is provided, you'll receive an immediate response like this:

```json
{
  "success": true,
  "message": "Upscale processing started in background",
  "processing": true,
  "callback_url": "https://myapp.com/webhooks/video-processed",
  "note": "Results will be sent to your callback URL when processing completes. This may take several minutes for large videos."
}
```

### Which Endpoints Support Async Mode?

The following long-running endpoints support async background processing when `callback_url` is provided:

| Endpoint | Typical Duration | Async Recommended |
|----------|------------------|-------------------|
| `/postprocess/upscale` | 3-30 minutes | ✅ **Highly Recommended** |
| `/postprocess/interpolate` | 2-20 minutes | ✅ **Highly Recommended** |
| `/postprocess/pipeline` | 5-60+ minutes | ✅ **Required for pipelines** |
| `/postprocess/filters` | 30s-5 minutes | Optional |
| `/postprocess/loop` | 30s-2 minutes | Optional |
| `/postprocess/export` | 30s-5 minutes | Optional |
| `/postprocess/join` | 30s-5 minutes | Optional |

### Best Practices for Production

1. **Always provide `callback_url`** for upscale, interpolate, and pipeline operations
2. **Implement a webhook endpoint** in your application to receive results
3. **Store job IDs** to correlate requests with callbacks
4. **Handle both success and failure callbacks** appropriately

### Example: Proper Production Usage

```bash
# This will NOT timeout - returns immediately, sends results to callback
curl -X POST "http://your-api.com/postprocess/upscale" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "model": "RealESRGAN_x4plus",
    "scale_factor": 2.0,
    "callback_url": "https://myapp.com/webhooks/video-processed"
  }'

# Immediate response:
# {"success": true, "processing": true, "message": "Upscale processing started in background", ...}

# Later, your callback URL receives the full result with the processed video
```

---

# Phase 1: AI Features

## POST `/enhance-prompt`

Enhance a simple prompt into a detailed, cinematic video prompt using IBM Granite LLM.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | ✅ | The simple prompt to enhance |

### Example Request

```json
{
  "prompt": "a cat playing with yarn"
}
```

### Example Response

```json
{
  "success": true,
  "original_prompt": "a cat playing with yarn",
  "enhanced_prompt": "A fluffy orange tabby cat playfully batting at a ball of bright red yarn on a sunlit hardwood floor. The cat's movements are graceful and curious, with soft natural lighting streaming through a nearby window creating warm highlights on its fur. The yarn unravels in gentle loops as the cat pounces and tumbles, capturing a moment of pure feline joy and mischief.",
  "message": "Prompt enhanced successfully"
}
```

---

## POST `/caption-image`

Generate a descriptive caption from an image using Microsoft Florence-2-large model.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_base64` | string | ❌* | Base64-encoded image |
| `image_url` | string | ❌* | URL to download image from |

*One of `image_base64` or `image_url` is required.

### Example Request

```json
{
  "image_url": "https://example.com/cat.jpg"
}
```

### Example Response

```json
{
  "success": true,
  "caption": "A fluffy orange cat sitting on a windowsill looking outside at birds",
  "message": "Image captioned successfully"
}
```

---

## POST `/ai/unload`

Unload AI models from memory to free up VRAM/RAM.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | ✅ | Model to unload: `"enhancer"`, `"captioner"`, or `"all"` |

### Available Models

| Model | Description |
|-------|-------------|
| `enhancer` | Unload the prompt enhancement model (Granite 3.3-2b) |
| `captioner` | Unload the image captioning model (Florence-2) |
| `all` | Unload all AI models |

### Example Request

```json
{
  "model": "all"
}
```

### Example Response

```json
{
  "success": true,
  "message": "Successfully unloaded 2 model(s)",
  "models_unloaded": ["enhancer", "captioner"]
}
```

---

## GET `/ai/status`

Check the status of AI models (whether they are loaded in memory).

### Example Response

```json
{
  "enhancer": {
    "loaded": true,
    "model_name": "ibm-granite/granite-3.3-2b-instruct"
  },
  "captioner": {
    "loaded": false,
    "model_name": null
  }
}
```

---

# Phase 2: Core Post-Processing

## POST `/postprocess/upscale`

Upscale a video using ESRGAN models (RealESRGAN, anime models, etc.).

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_base64` | string | ❌* | - | Base64-encoded video |
| `video_url` | string | ❌* | - | URL to download video from |
| `model` | string | ❌ | `"RealESRGAN_x4plus"` | Upscale model to use |
| `scale_factor` | float | ❌ | `2.0` | Target scale factor (1.0 - 4.0) |
| `tile_size` | int | ❌ | `0` | Tile size for processing (0=auto, 256, 512). Smaller uses less VRAM. |
| `enhance_face` | bool | ❌ | `false` | Enable GFPGAN face enhancement |
| `denoise_strength` | float | ❌ | `0.5` | Denoise strength (0.0-1.0, only for RealESR-general-x4v3) |
| `use_streaming` | bool | ❌ | `false` | Use streaming mode for low memory processing of large videos |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### Available Models

| Model | Scale | Description |
|-------|-------|-------------|
| `RealESRGAN_x2plus` | 2x | General purpose, balanced |
| `RealESRGAN_x4plus` | 4x | General purpose, sharp |
| `RealESRNet_x4plus` | 4x | Smoother results |
| `RealESR-general-x4v3` | 4x | With denoise slider |
| `RealESRGAN_x4plus_anime_6B` | 4x | Anime optimized |
| `RealESR_AnimeVideo_v3` | 4x | Anime video specialized |

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "model": "RealESRGAN_x4plus",
  "scale_factor": 2.0,
  "enhance_face": true
}
```

### Example Response

```json
{
  "success": true,
  "message": "Video upscaled successfully",
  "output_video_base64": "AAAAIGZ0eXBpc29t...",
  "output_path": "/outputs/postprocessed_output/temp_processing/video_upscaled_x2_241204_143022.mp4"
}
```

---

## POST `/postprocess/interpolate`

Interpolate frames to increase video frame rate using RIFE.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_base64` | string | ❌* | - | Base64-encoded video |
| `video_url` | string | ❌* | - | URL to download video from |
| `fps_mode` | string | ❌ | `"2x"` | Interpolation mode |
| `speed_factor` | float | ❌ | `1.0` | Speed adjustment (0.25-4.0) |
| `use_streaming` | bool | ❌ | `false` | Use streaming mode for low memory processing |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### FPS Modes

| Mode | Description |
|------|-------------|
| `No Interpolation` | Speed adjustment only |
| `2x` | Double the frame rate |
| `4x` | Quadruple the frame rate |

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "fps_mode": "2x",
  "speed_factor": 0.5
}
```

---

## POST `/postprocess/filters`

Apply visual filters to a video (brightness, contrast, saturation, etc.).

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_base64` | string | ❌* | - | Base64-encoded video |
| `video_url` | string | ❌* | - | URL to download video from |
| `brightness` | int | ❌ | `0` | Brightness adjustment (-100 to 100) |
| `contrast` | float | ❌ | `1.0` | Contrast multiplier (0.0 to 3.0) |
| `saturation` | float | ❌ | `1.0` | Saturation multiplier (0.0 to 3.0) |
| `temperature` | int | ❌ | `0` | Color temperature (-100 to 100) |
| `sharpen` | float | ❌ | `0.0` | Sharpening amount (0.0 to 5.0) |
| `blur` | float | ❌ | `0.0` | Blur amount (0.0 to 5.0) |
| `denoise` | float | ❌ | `0.0` | Denoise strength (0.0 to 10.0) |
| `vignette` | int | ❌ | `0` | Vignette intensity (0 to 100) |
| `s_curve_contrast` | int | ❌ | `0` | S-curve contrast (0 to 100) |
| `film_grain` | int | ❌ | `0` | Film grain strength (0 to 50) |
| `preset` | string | ❌ | `null` | Filter preset name (cinematic, vintage, cool, warm, dramatic) |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "brightness": 10,
  "contrast": 1.3,
  "saturation": 0.9,
  "vignette": 15
}
```

---

## POST `/postprocess/loop`

Create looped versions of a video.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_base64` | string | ❌* | - | Base64-encoded video |
| `video_url` | string | ❌* | - | URL to download video from |
| `loop_type` | string | ❌ | `"loop"` | Type of loop (`loop` or `ping-pong`) |
| `num_loops` | int | ❌ | `1` | Number of additional loops/repeats (1-10). Value of 1 = plays twice total. |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### Loop Types

| Type | Description |
|------|-------------|
| `loop` | Simple repeat |
| `ping-pong` | Forward then reverse (seamless) |

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "loop_type": "ping-pong",
  "num_loops": 3
}
```

---

## POST `/postprocess/join`

Join/concatenate multiple videos into one.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_urls` | array | ✅ | - | List of video URLs to join (minimum 2) |
| `output_name` | string | ❌ | `null` | Custom output filename |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

> **Note:** Unlike other endpoints, join only supports URLs, not base64 input.

### Example Request

```json
{
  "video_urls": [
    "https://example.com/video1.mp4",
    "https://example.com/video2.mp4",
    "https://example.com/video3.mp4"
  ],
  "output_name": "combined_video"
}
```

---

## POST `/postprocess/export`

Export/convert a video to different formats with quality control.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_base64` | string | ❌* | - | Base64-encoded video |
| `video_url` | string | ❌* | - | URL to download video from |
| `format` | string | ❌ | `"MP4"` | Output format (MP4, WebM, or GIF) |
| `quality` | int | ❌ | `85` | Quality (0-100, higher = better quality, larger file) |
| `max_width` | int | ❌ | `1024` | Maximum width in pixels (256-4096, maintains aspect ratio) |
| `output_name` | string | ❌ | `null` | Custom output filename |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### Supported Formats

| Format | Description |
|--------|-------------|
| `MP4` | H.264 video, widely compatible |
| `WebM` | VP9 video, web optimized |
| `GIF` | Animated GIF |

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "format": "GIF",
  "quality": 80,
  "max_width": 480
}
```

---

## POST `/postprocess/pipeline`

Run multiple post-processing operations in sequence.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_base64` | string | ❌* | Base64-encoded video |
| `video_url` | string | ❌* | URL to download video from |
| `operations` | array | ✅ | Array of operation objects |
| `callback_url` | string | ❌ | Webhook URL to receive completion notification |

Each operation object:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | ✅ | Operation type |
| `params` | object | ❌ | Operation parameters |

### Operation Types

- `upscale` - Video upscaling
- `interpolate` - Frame interpolation
- `filters` - Video filters
- `loop` - Video looping
- `export` - Format conversion

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "operations": [
    {
      "type": "upscale",
      "params": {
        "model": "RealESRGAN_x2plus",
        "scale_factor": 2.0
      }
    },
    {
      "type": "filters",
      "params": {
        "brightness": 5,
        "contrast": 1.2
      }
    },
    {
      "type": "export",
      "params": {
        "format": "MP4",
        "quality": 90
      }
    }
  ]
}
```

---

## POST `/postprocess/analyze`

Analyze a video and get detailed metadata.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_base64` | string | ❌* | Base64-encoded video |
| `video_url` | string | ❌* | URL to download video from |

### Example Response

```json
{
  "success": true,
  "analysis": {
    "filename": "video.mp4",
    "file_size": "15.2 MB",
    "duration": "4.5 seconds",
    "resolution": "1280x720",
    "fps": 30.0,
    "codec": "h264",
    "bitrate": "28.1 Mbps",
    "has_audio": true,
    "audio_codec": "aac",
    "total_frames": 135
  }
}
```

---

## GET `/postprocess/models`

List all available upscale models.

### Example Response

```json
{
  "success": true,
  "models": {
    "RealESRGAN_x2plus": {
      "scale": 2,
      "description": "General purpose 2x upscaler, balanced quality"
    },
    "RealESRGAN_x4plus": {
      "scale": 4,
      "description": "General purpose 4x upscaler, sharp results"
    },
    "RealESRGAN_x4plus_anime_6B": {
      "scale": 4,
      "description": "Optimized for anime content"
    }
  }
}
```

---

# Phase 3: Frames Studio & Workflow Presets

## POST `/frames/extract`

Extract frames from a video for frame-by-frame editing.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_base64` | string | ❌* | - | Base64-encoded video |
| `video_url` | string | ❌* | - | URL to download video from |
| `extraction_rate` | int | ❌ | `1` | Extract every Nth frame (1 = all frames) |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### Example Request

```json
{
  "video_url": "https://example.com/video.mp4",
  "extraction_rate": 2
}
```

### Example Response

```json
{
  "success": true,
  "message": "Successfully extracted 67 frames",
  "folder_name": "video_extracted_every_2_241204_143522",
  "frame_count": 67,
  "frames_path": "/outputs/postprocessed_output/frames/extracted_frames/video_extracted_every_2_241204_143522"
}
```

---

## GET `/frames/folders`

List all extracted frame folders.

### Example Response

```json
{
  "success": true,
  "folders": [
    "video1_extracted_every_1_241204_140000",
    "video2_extracted_every_2_241204_141500"
  ],
  "message": "Found 2 extracted frame folder(s)"
}
```

---

## GET `/frames/{folder}`

List all frames in a specific folder.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `folder` | string | Folder name from `/frames/folders` |

### Example Response

```json
{
  "success": true,
  "folder": "video_extracted_every_1_241204_140000",
  "frames": [
    { "filename": "frame_000000.png", "path": "/full/path/frame_000000.png", "index": 0 },
    { "filename": "frame_000001.png", "path": "/full/path/frame_000001.png", "index": 1 },
    { "filename": "frame_000002.png", "path": "/full/path/frame_000002.png", "index": 2 }
  ],
  "total_count": 135,
  "message": "Found 135 frame(s) in folder"
}
```

---

## DELETE `/frames/{folder}`

Delete an entire extracted frames folder.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `folder` | string | Folder name to delete |

### Example Response

```json
{
  "success": true,
  "folder": "video_extracted_every_1_241204_140000",
  "message": "Successfully deleted folder and all its contents"
}
```

---

## DELETE `/frames/{folder}/{frame}`

Delete a single frame from a folder.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `folder` | string | Folder name |
| `frame` | string | Frame filename (e.g., `frame_000042.png`) |

### Example Response

```json
{
  "success": true,
  "folder": "video_extracted_every_1_241204_140000",
  "frame": "frame_000042.png",
  "message": "Successfully deleted frame 'frame_000042.png'"
}
```

---

## POST `/frames/{folder}/save/{frame}`

Save a single frame as a high-quality image to permanent storage.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `folder` | string | Folder name |
| `frame` | string | Frame filename |

### Example Response

```json
{
  "success": true,
  "message": "Successfully saved frame 'frame_000042.png'",
  "saved_path": "/outputs/postprocessed_output/saved_videos/saved_frame_video_241204_143522_frame_000042.png",
  "frame_base64": "iVBORw0KGgoAAAANSUhEUg..."
}
```

---

## POST `/frames/reassemble`

Reassemble edited frames back into a video.

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `folder_name` | string | ✅ | - | Folder containing frames |
| `output_fps` | int | ❌ | `30` | Output video FPS |
| `output_name` | string | ❌ | `null` | Custom output name |
| `callback_url` | string | ❌ | - | Webhook URL to receive completion notification |

### Example Request

```json
{
  "folder_name": "video_extracted_every_1_241204_140000",
  "output_fps": 24,
  "output_name": "edited_video"
}
```

### Example Response

```json
{
  "success": true,
  "message": "Frames reassembled successfully",
  "output_video_base64": "AAAAIGZ0eXBpc29t...",
  "output_path": "/outputs/postprocessed_output/frames/reassembled_videos/edited_video_24fps_reassembled_241204_144000.mp4"
}
```

---

## GET `/workflow/presets`

List all saved workflow presets.

### Example Response

```json
{
  "success": true,
  "presets": {
    "None": {
      "active_steps": [],
      "params": {
        "upscale_model": "RealESRGAN_x2plus",
        "upscale_factor": 2.0,
        "brightness": 0,
        "contrast": 1
      }
    },
    "4K Upscale + Cinematic": {
      "active_steps": ["Upscale", "Filters", "Export"],
      "params": {
        "upscale_model": "RealESRGAN_x4plus",
        "upscale_factor": 4.0,
        "brightness": -5,
        "contrast": 1.3,
        "vignette": 10
      }
    }
  },
  "message": "Found 2 workflow preset(s)"
}
```

---

## POST `/workflow/presets`

Save a new workflow preset.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Preset name |
| `preset_data` | object | ✅ | Preset configuration |

### preset_data Object

| Field | Type | Description |
|-------|------|-------------|
| `active_steps` | array | Steps to enable: `["Upscale", "Frame Adjust", "Filters", "Loop", "Export"]` |
| `params` | object | All parameter values |

### Example Request

```json
{
  "name": "My Custom Workflow",
  "preset_data": {
    "active_steps": ["Upscale", "Filters"],
    "params": {
      "upscale_model": "RealESRGAN_x2plus",
      "upscale_factor": 2.0,
      "tile_size": 512,
      "enhance_face": false,
      "brightness": 5,
      "contrast": 1.2,
      "saturation": 1.0
    }
  }
}
```

---

## DELETE `/workflow/presets/{name}`

Delete a workflow preset.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Preset name to delete |

> **Note:** The "None" preset cannot be deleted.

---

## POST `/postprocess/pipeline/from-preset`

Run a pipeline using a saved workflow preset.

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `preset_name` | string | ✅ | Name of the workflow preset |
| `video_url` | string | ❌* | Video URL |
| `video_base64` | string | ❌* | Base64 video |

### Example Request

```bash
curl -X POST "http://localhost:8000/postprocess/pipeline/from-preset?preset_name=4K%20Upscale%20%2B%20Cinematic&video_url=https://example.com/video.mp4" \
  -H "X-API-Key: YOUR_KEY"
```

---

## POST `/system/clear-temp`

Clear temporary processing files to free disk space.

### Example Response

```json
{
  "success": true,
  "message": "Cleared 15 temporary file(s), freed 234.50 MB",
  "files_deleted": 15,
  "space_freed_mb": 234.50
}
```

---

## GET `/system/status`

Get current system resource status.

### Example Response

```json
{
  "success": true,
  "message": "System status retrieved successfully",
  "ram": {
    "used_gb": 12.5,
    "total_gb": 32.0,
    "percent": 39.1
  },
  "vram": {
    "used_gb": 8.2,
    "total_gb": 24.0,
    "percent": 34.2
  },
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "utilization": 15,
    "temperature": 45
  }
}
```

---

# Phase 4: Filter Presets, Batch Processing & Model Management

## GET `/filters/presets`

List all filter presets (built-in + user-created).

### Example Response

```json
{
  "success": true,
  "presets": {
    "none": {
      "brightness": 0, "contrast": 1, "saturation": 1, "temperature": 0,
      "sharpen": 0, "blur": 0, "denoise": 0, "vignette": 0,
      "s_curve_contrast": 0, "film_grain_strength": 0
    },
    "cinematic": {
      "brightness": -5, "contrast": 1.3, "saturation": 0.9, "temperature": 20,
      "vignette": 10, "sharpen": 1.2, "s_curve_contrast": 15, "film_grain_strength": 5
    },
    "vintage": {
      "brightness": 5, "contrast": 1.1, "saturation": 0.7, "temperature": 15,
      "vignette": 30, "blur": 0.5, "s_curve_contrast": 10, "film_grain_strength": 10
    },
    "cool": { ... },
    "warm": { ... },
    "dramatic": { ... }
  },
  "message": "Found 6 filter preset(s)"
}
```

---

## POST `/filters/presets`

Save a new filter preset.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Preset name (cannot be "none") |
| `settings` | object | ✅ | Filter settings |

### settings Object

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `brightness` | float | 0 | -100 to 100 | Brightness adjustment |
| `contrast` | float | 1 | 0 to 3 | Contrast multiplier |
| `saturation` | float | 1 | 0 to 3 | Saturation multiplier |
| `temperature` | float | 0 | -100 to 100 | Color temperature |
| `sharpen` | float | 0 | 0 to 5 | Sharpening |
| `blur` | float | 0 | 0 to 5 | Blur amount |
| `denoise` | float | 0 | 0 to 10 | Denoise strength |
| `vignette` | float | 0 | 0 to 100 | Vignette intensity |
| `s_curve_contrast` | float | 0 | 0 to 100 | S-curve contrast |
| `film_grain_strength` | float | 0 | 0 to 50 | Film grain |

### Example Request

```json
{
  "name": "my_custom_look",
  "settings": {
    "brightness": 0,
    "contrast": 1.4,
    "saturation": 0.85,
    "temperature": 10,
    "vignette": 20,
    "s_curve_contrast": 12
  }
}
```

---

## DELETE `/filters/presets/{name}`

Delete a user-created filter preset.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Preset name to delete |

> **Note:** The "none" preset cannot be deleted.

---

## POST `/postprocess/batch`

Process multiple videos through the same pipeline.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `videos` | array | ✅ | Array of video sources |
| `operations` | array | ✅ | Operations to apply to each video |
| `callback_url` | string | ❌ | Webhook URL to receive completion notification |

### Example Request

```json
{
  "videos": [
    { "video_url": "https://example.com/video1.mp4" },
    { "video_url": "https://example.com/video2.mp4" },
    { "video_base64": "AAAAIGZ0eXBpc29t..." }
  ],
  "operations": [
    {
      "type": "upscale",
      "params": { "model": "RealESRGAN_x2plus", "scale_factor": 2.0 }
    },
    {
      "type": "filters",
      "params": { "brightness": 5, "contrast": 1.2 }
    }
  ]
}
```

### Example Response

```json
{
  "success": true,
  "message": "Batch processing complete: 3 successful, 0 failed",
  "total_videos": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "index": 0,
      "success": true,
      "input_source": "https://example.com/video1.mp4",
      "output_video_base64": "AAAAIGZ0eXBpc29t...",
      "output_path": "/outputs/video1_processed.mp4"
    },
    {
      "index": 1,
      "success": true,
      "input_source": "https://example.com/video2.mp4",
      "output_video_base64": "AAAAIGZ0eXBpc29t...",
      "output_path": "/outputs/video2_processed.mp4"
    },
    {
      "index": 2,
      "success": true,
      "input_source": "base64",
      "output_video_base64": "AAAAIGZ0eXBpc29t...",
      "output_path": "/outputs/video3_processed.mp4"
    }
  ]
}
```

---

## POST `/postprocess/save`

Save a video to permanent storage.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video_base64` | string | ❌* | Base64-encoded video |
| `video_url` | string | ❌* | URL to download video from |
| `custom_name` | string | ❌ | Custom filename |

### Example Response

```json
{
  "success": true,
  "message": "Video saved to permanent storage successfully",
  "saved_path": "/outputs/postprocessed_output/saved_videos/my_video.mp4",
  "filename": "my_video.mp4"
}
```

---

## GET `/postprocess/autosave`

Get the current autosave setting.

### Example Response

```json
{
  "success": true,
  "message": "Autosave is enabled",
  "autosave_enabled": true
}
```

---

## POST `/postprocess/autosave`

Set the autosave mode.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | ✅ | Enable or disable autosave |

### Example Request

```json
{
  "enabled": true
}
```

### Example Response

```json
{
  "success": true,
  "message": "Autosave enabled successfully",
  "autosave_enabled": true
}
```

> **Note:** When enabled, all processed videos are automatically saved to the permanent `saved_videos` directory. This setting persists across API restarts.

---

## POST `/model/unload`

Unload the main video generation transformer to free VRAM.

Use this before heavy post-processing operations (like 4K upscaling) to maximize available GPU memory.

### Example Response

```json
{
  "success": true,
  "message": "Model 'HunyuanVideoTransformer3DModel' unloaded successfully. VRAM freed.",
  "model_unloaded": "HunyuanVideoTransformer3DModel",
  "vram_freed_estimate": "~12-24 GB (varies by model)"
}
```

### Error Response (Job Running)

```json
{
  "detail": "Cannot unload model: A video generation job is currently running"
}
```
HTTP Status: `409 Conflict`

---

## GET `/model/status`

Get the status of the main video generation model.

### Example Response

```json
{
  "model_loaded": true,
  "model_name": "HunyuanVideoTransformer3DModel",
  "vram_usage": 14.5
}
```

---

# Error Responses

All endpoints may return these error responses:

### 400 Bad Request
```json
{
  "detail": "Either video_base64 or video_url is required"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid or missing API key"
}
```

### 404 Not Found
```json
{
  "detail": "Preset 'my_preset' not found"
}
```

### 409 Conflict
```json
{
  "detail": "Cannot unload model: A video generation job is currently running"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Post-processing module not available"
}
```

---

# Quick Reference

## All Endpoints Summary

| Phase | Method | Endpoint | Description |
|-------|--------|----------|-------------|
| 1 | POST | `/enhance-prompt` | Enhance prompt with LLM |
| 1 | POST | `/caption-image` | Caption image with LLM |
| 1 | POST | `/ai/unload` | Unload AI models (enhancer/captioner) |
| 1 | GET | `/ai/status` | Check AI model status |
| 2 | POST | `/postprocess/upscale` | Upscale video |
| 2 | POST | `/postprocess/interpolate` | Frame interpolation |
| 2 | POST | `/postprocess/filters` | Apply video filters |
| 2 | POST | `/postprocess/loop` | Create video loops |
| 2 | POST | `/postprocess/join` | Join multiple videos |
| 2 | POST | `/postprocess/export` | Export/convert video |
| 2 | POST | `/postprocess/pipeline` | Run multiple operations |
| 2 | POST | `/postprocess/analyze` | Analyze video metadata |
| 2 | GET | `/postprocess/models` | List upscale models |
| 3 | POST | `/frames/extract` | Extract frames from video |
| 3 | GET | `/frames/folders` | List frame folders |
| 3 | GET | `/frames/{folder}` | List frames in folder |
| 3 | DELETE | `/frames/{folder}` | Delete frame folder |
| 3 | DELETE | `/frames/{folder}/{frame}` | Delete single frame |
| 3 | POST | `/frames/{folder}/save/{frame}` | Save single frame |
| 3 | POST | `/frames/reassemble` | Reassemble frames to video |
| 3 | GET | `/workflow/presets` | List workflow presets |
| 3 | POST | `/workflow/presets` | Save workflow preset |
| 3 | DELETE | `/workflow/presets/{name}` | Delete workflow preset |
| 3 | POST | `/postprocess/pipeline/from-preset` | Run pipeline from preset |
| 3 | POST | `/system/clear-temp` | Clear temp files |
| 3 | GET | `/system/status` | Get system status |
| 4 | GET | `/filters/presets` | List filter presets |
| 4 | POST | `/filters/presets` | Save filter preset |
| 4 | DELETE | `/filters/presets/{name}` | Delete filter preset |
| 4 | POST | `/postprocess/batch` | Batch process videos |
| 4 | POST | `/postprocess/save` | Save video to storage |
| 4 | GET | `/postprocess/autosave` | Get autosave setting |
| 4 | POST | `/postprocess/autosave` | Set autosave setting |
| 4 | POST | `/model/unload` | Unload main model |
| 4 | GET | `/model/status` | Get model status |

---

*Documentation generated for FramePack Studio API v1.0*
