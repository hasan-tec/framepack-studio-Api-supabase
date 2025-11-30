# FramePack Studio API Documentation

> **Base URL:** `http://localhost:8000`  
> **API Version:** 1.0.0  
> **Authentication:** API Key via `X-API-Key` header

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Generate Video](#generate-video)
  - [Job Status](#job-status)
  - [Cancel Job](#cancel-job)
  - [Queue Status](#queue-status)
  - [List LoRAs](#list-loras)
  - [WebSocket Progress](#websocket-progress)
  - [SSE Progress Stream](#sse-progress-stream)
  - [Delete Output File](#delete-output-file)
  - [Cleanup Job Files](#cleanup-job-files)
- [Webhook/Callback System](#webhookcallback-system)
- [Models & Enums](#models--enums)
- [Example Workflows](#example-workflows)

---

## Authentication

All endpoints (except `/health`) require an API key passed in the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/status/job-id
```

Set your API key via environment variable:
```bash
export FRAMEPACK_API_SECRET="your-secret-key"
```

---

## Rate Limiting

- **Default:** 10 requests per 60 seconds per API key
- Configure via environment variables:
  - `RATE_LIMIT_REQUESTS` - Max requests per window (default: 10)
  - `RATE_LIMIT_WINDOW` - Window in seconds (default: 60)

---

## Endpoints

### Health Check

Check API status and available resources. **No authentication required.**

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "vram_free_gb": 20.5,
  "high_vram_mode": false,
  "queue_length": 2,
  "available_loras": ["my_lora", "another_lora"],
  "available_models": ["Original", "Original with Endframe", "F1", "Video", "Video F1"]
}
```

---

### Generate Video

Submit a video generation job. Returns immediately with a job ID for polling.

```
POST /generate
```

**Headers:**
```
X-API-Key: your-secret-key
Content-Type: application/json
```

**Request Body:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Main prompt describing what to generate |
| `negative_prompt` | string | `"low quality, worst quality..."` | Qualities to avoid |
| `model_type` | enum | `"Original"` | Model: `Original`, `Original with Endframe`, `F1`, `Video`, `Video F1` |
| `input_image_base64` | string | `null` | Base64 encoded input image (for I2V) |
| `input_image_url` | string | `null` | URL to download input image |
| `end_frame_image_base64` | string | `null` | Base64 encoded end frame image |
| `end_frame_strength` | float | `1.0` | End frame influence (0.0-1.0) |
| `seed` | int | `-1` | Random seed (-1 for random) |
| `steps` | int | `25` | Diffusion steps (1-100) |
| `total_second_length` | float | `6.0` | Video length in seconds (1.0-120.0) |
| `resolution_w` | int | `640` | Video width (128-1920) |
| `resolution_h` | int | `640` | Video height (128-1920) |
| `cfg` | float | `1.0` | CFG Scale (1.0-3.0) |
| `gs` | float | `10.0` | Distilled CFG Scale (1.0-32.0) |
| `rs` | float | `0.0` | CFG Re-Scale (0.0-1.0) |
| `latent_type` | enum | `"Black"` | Background: `Black`, `White`, `Noise`, `Green Screen` |
| `latent_window_size` | int | `9` | Latent window size (1-33) |
| `cache_type` | enum | `"MagCache"` | Cache: `None`, `TeaCache`, `MagCache` |
| `teacache_num_steps` | int | `25` | TeaCache intermediate sections (1-50) |
| `teacache_rel_l1_thresh` | float | `0.15` | TeaCache L1 threshold (0.01-1.0) |
| `magcache_threshold` | float | `0.1` | MagCache error tolerance (0.01-1.0) |
| `magcache_max_consecutive_skips` | int | `2` | MagCache max skips (1-5) |
| `magcache_retention_ratio` | float | `0.25` | MagCache retention (0.0-1.0) |
| `blend_sections` | int | `4` | Prompt blending transitions (0-10) |
| `loras` | object | `null` | LoRA weights: `{"lora_name": 0.8}` |
| `input_video_url` | string | `null` | Input video URL (for Video model) |
| `combine_with_source` | bool | `false` | Combine with source video |
| `num_cleaned_frames` | int | `5` | Cleaned frames for Video model (1-20) |
| `save_metadata` | bool | `true` | Save generation metadata |
| `callback_url` | string | `null` | Webhook URL for job completion |
| `callback_token` | string | `null` | Bearer token for callback auth |

**Example Request (Text-to-Video):**
```json
{
  "prompt": "A majestic eagle soaring through clouds at sunset",
  "negative_prompt": "blurry, low quality",
  "steps": 30,
  "total_second_length": 5.0,
  "seed": 42
}
```

**Example Request (Image-to-Video with Webhook):**
```json
{
  "prompt": "The person starts dancing gracefully",
  "input_image_url": "https://example.com/image.jpg",
  "model_type": "Original",
  "steps": 25,
  "total_second_length": 6.0,
  "callback_url": "https://your-project.supabase.co/functions/v1/video-complete",
  "callback_token": "your-service-role-key"
}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "seed": 42,
  "resolution": {"width": 640, "height": 640},
  "estimated_time_seconds": 120.5,
  "message": "Job queued successfully. Estimated time: 120.5s"
}
```

---

### Job Status

Check the status of a generation job.

```
GET /status/{job_id}
```

**Response (Pending/Running):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "progress_percent": 45.5,
  "progress_desc": "Sampling 12/25 | Section 2/4",
  "eta_seconds": 65.2,
  "created_at": 1701234567.123,
  "started_at": 1701234570.456
}
```

**Response (Completed):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress_percent": 100,
  "progress_desc": "Completed",
  "result_url": "/outputs/20231130_123456_video.mp4",
  "created_at": 1701234567.123,
  "started_at": 1701234570.456,
  "completed_at": 1701234690.789
}
```

**Response (Failed):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "failed",
  "error": "CUDA out of memory",
  "created_at": 1701234567.123,
  "started_at": 1701234570.456,
  "completed_at": 1701234575.000
}
```

---

### Cancel Job

Cancel a pending or running job.

```
POST /cancel/{job_id}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "cancelled"
}
```

---

### Queue Status

Get current queue status and all jobs.

```
GET /queue
```

**Response:**
```json
{
  "summary": {
    "pending": 3,
    "running": 1,
    "completed": 15,
    "failed": 2,
    "total": 21
  },
  "jobs": [
    {
      "id": "job-1",
      "status": "running",
      "created_at": 1701234567.123,
      "model_type": "Original",
      "prompt": "A cat playing piano..."
    },
    {
      "id": "job-2",
      "status": "pending",
      "queue_position": 1,
      "model_type": "F1",
      "prompt": "Sunset over mountains..."
    }
  ]
}
```

---

### Clear Completed Jobs

Remove all completed and cancelled jobs from the queue.

```
DELETE /queue/completed
```

**Response:**
```json
{
  "removed_count": 15,
  "message": "Removed 15 completed/cancelled jobs"
}
```

---

### List LoRAs

Get available LoRA models.

```
GET /loras
```

**Response:**
```json
{
  "loras": ["artistic_style", "anime_v2", "realistic_skin"],
  "lora_dir": "./loras"
}
```

---

### WebSocket Progress

Real-time progress streaming via WebSocket. **No authentication required.**

```
WS /ws/progress/{job_id}
```

**Messages Received:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "running",
  "progress_desc": "Sampling 15/25",
  "timestamp": 1701234580.123
}
```

```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "result_url": "/outputs/video.mp4",
  "timestamp": 1701234690.789
}
```

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/progress/job-id-here');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Status: ${data.status}, Progress: ${data.progress_desc}`);
  
  if (data.status === 'completed') {
    console.log(`Video ready: ${data.result_url}`);
    ws.close();
  }
};
```

---

### SSE Progress Stream

Server-Sent Events for progress streaming (simpler than WebSocket).

```
GET /stream/progress/{job_id}
```

**JavaScript Example:**
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/stream/progress/job-id?x_api_key=your-key'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

---

### Delete Output File

Delete a specific video file from outputs. Use after uploading to external storage.

```
DELETE /outputs/{filename}
```

**Example:**
```bash
curl -X DELETE \
  -H "X-API-Key: your-secret-key" \
  http://localhost:8000/outputs/20231130_123456_video.mp4
```

**Response:**
```json
{
  "success": true,
  "message": "File 20231130_123456_video.mp4 deleted successfully",
  "deleted_files": ["20231130_123456_video.mp4"]
}
```

---

### Cleanup Job Files

Delete all files associated with a job (video + intermediates).

```
POST /cleanup/job/{job_id}
```

**Response:**
```json
{
  "success": true,
  "job_id": "a1b2c3d4",
  "deleted_files": ["a1b2c3d4_45.mp4", "a1b2c3d4_90.mp4"],
  "errors": null
}
```

---

## Webhook/Callback System

When you provide a `callback_url` in your generation request, the API will POST to that URL when the job completes or fails.

### Callback Payload

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "created_at": 1701234567.123,
  "started_at": 1701234570.456,
  "completed_at": 1701234690.789,
  "error": null,
  "result_url": "/outputs/video.mp4",
  "video_download_url": "http://localhost:8000/outputs/video.mp4",
  "video_filename": "video.mp4",
  "video_local_path": "/path/to/outputs/video.mp4",
  "video_file_size": 15234567,
  "metadata": {
    "model_type": "Original",
    "prompt": "A cat playing piano",
    "negative_prompt": "blurry, low quality",
    "seed": 42,
    "steps": 25,
    "cfg": 1.0,
    "gs": 10.0,
    "resolution_w": 640,
    "resolution_h": 640,
    "total_second_length": 6.0
  }
}
```

### Callback Headers

```
Content-Type: application/json
X-Job-ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
Authorization: Bearer <callback_token>  (if provided)
```

### Supabase Edge Function Example

```typescript
// supabase/functions/video-complete/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

serve(async (req) => {
  // Verify authorization
  const authHeader = req.headers.get('Authorization');
  if (authHeader !== `Bearer ${Deno.env.get('EXPECTED_TOKEN')}`) {
    return new Response('Unauthorized', { status: 401 });
  }

  const payload = await req.json();
  console.log('Received callback:', payload.job_id, payload.status);

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  );

  if (payload.status === 'completed') {
    // 1. Download video from API
    const videoResponse = await fetch(payload.video_download_url);
    const videoBlob = await videoResponse.blob();

    // 2. Upload to Supabase Storage
    const storagePath = `videos/${payload.job_id}.mp4`;
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('generated-videos')
      .upload(storagePath, videoBlob, {
        contentType: 'video/mp4',
        upsert: true
      });

    if (uploadError) {
      console.error('Upload error:', uploadError);
      return new Response(JSON.stringify({ error: uploadError }), { status: 500 });
    }

    // 3. Get public URL
    const { data: { publicUrl } } = supabase.storage
      .from('generated-videos')
      .getPublicUrl(storagePath);

    // 4. Update database record
    const { error: dbError } = await supabase
      .from('video_generations')
      .update({
        status: 'completed',
        video_url: publicUrl,
        completed_at: new Date(payload.completed_at * 1000).toISOString(),
        metadata: payload.metadata
      })
      .eq('job_id', payload.job_id);

    if (dbError) {
      console.error('DB error:', dbError);
      return new Response(JSON.stringify({ error: dbError }), { status: 500 });
    }

    // 5. Tell API to delete local file (cleanup)
    await fetch(`${Deno.env.get('FRAMEPACK_API_URL')}/outputs/${payload.video_filename}`, {
      method: 'DELETE',
      headers: { 'X-API-Key': Deno.env.get('FRAMEPACK_API_KEY')! }
    });

    return new Response(JSON.stringify({ success: true, url: publicUrl }));
  } 
  
  else if (payload.status === 'failed') {
    // Update database with error
    await supabase
      .from('video_generations')
      .update({
        status: 'failed',
        error: payload.error,
        completed_at: new Date(payload.completed_at * 1000).toISOString()
      })
      .eq('job_id', payload.job_id);

    return new Response(JSON.stringify({ success: true }));
  }

  return new Response(JSON.stringify({ received: true }));
});
```

---

## Models & Enums

### ModelType
| Value | Description |
|-------|-------------|
| `Original` | Standard image-to-video generation |
| `Original with Endframe` | I2V with end frame guidance |
| `F1` | F1 architecture model |
| `Video` | Video-to-video generation |
| `Video F1` | Video-to-video with F1 architecture |

### LatentType
| Value | Description |
|-------|-------------|
| `Black` | Black background for T2V |
| `White` | White background |
| `Noise` | Random noise background |
| `Green Screen` | Green screen background |

### CacheType
| Value | Description |
|-------|-------------|
| `None` | No caching (slowest, highest quality) |
| `TeaCache` | TeaCache acceleration |
| `MagCache` | MagCache acceleration (recommended) |

### JobStatus
| Value | Description |
|-------|-------------|
| `pending` | Job queued, waiting to start |
| `running` | Job currently processing |
| `completed` | Job finished successfully |
| `failed` | Job failed with error |
| `cancelled` | Job was cancelled |

---

## Example Workflows

### Basic Text-to-Video

```bash
# 1. Submit job
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene waterfall in a tropical forest",
    "steps": 25,
    "total_second_length": 5
  }'

# Response: {"job_id": "abc123", ...}

# 2. Poll for status
curl http://localhost:8000/status/abc123 -H "X-API-Key: your-key"

# 3. Download when complete
curl -O http://localhost:8000/outputs/abc123_video.mp4
```

### Image-to-Video with LoRA

```bash
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The woman turns and smiles at the camera",
    "input_image_url": "https://example.com/portrait.jpg",
    "model_type": "Original",
    "loras": {"realistic_skin": 0.7},
    "steps": 30,
    "total_second_length": 4
  }'
```

### With Supabase Webhook

```bash
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ocean waves crashing on a beach",
    "steps": 25,
    "callback_url": "https://myproject.supabase.co/functions/v1/video-complete",
    "callback_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

---

## Error Codes

| Status | Description |
|--------|-------------|
| `400` | Bad request (invalid parameters) |
| `401` | Missing or invalid API key |
| `403` | Access denied |
| `404` | Job or resource not found |
| `429` | Rate limit exceeded |
| `500` | Internal server error |

---

## Interactive Docs

The API includes interactive documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEPACK_API_SECRET` | `CHANGE_ME_IN_PRODUCTION` | API authentication key |
| `RATE_LIMIT_REQUESTS` | `10` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

---

*Generated for FramePack Studio API v1.0.0*
