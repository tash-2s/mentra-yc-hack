# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this repository.

## Common Commands

### Running the RTMP-to-WebRTC Server
```bash
uv run main.py --port 8080

# With face blurring enabled
uv run main.py --blur --allow-dir ./allowlist

# With frame capture enabled
uv run main.py --capture --capture-interval 1.0 --capture-quality 85

# With both blur and capture
uv run main.py --blur --capture

# With speech-to-text transcription (creates new timestamped file each startup)
uv run download_vosk_model.py  # First time only
uv run main.py --transcript

# With transcript and AI-powered consent detection
export OPENAI_API_KEY="your-api-key"  # Required for consent detection
uv run main.py --transcript

# All features combined
uv run main.py --blur --capture --transcript

# Performance optimized settings
uv run main.py --blur --fps 15 --detection-scale 0.4 --box-blur

# Maximum performance (lower quality)
uv run main.py --blur --fps 15 --detection-scale 0.25 --box-blur --blur-kernel 21 --detection-interval 10

# With RTMP relay to external server (stream to rtmp://localhost:1935/stream/live.example.com:1935__live__mykey)
uv run main.py --blur
```

### Performance Tuning Options
```bash
# Reduce framerate (default: 30)
--fps 15  # Halves processing load

# Multi-resolution face detection (default: 0.5 = 50%)
--detection-scale 0.4  # Detect at 40% resolution (6x faster than full res)

# Blur optimization
--blur-kernel 21      # Smaller kernel (default: 31, smaller is faster)
--box-blur           # Use box blur instead of Gaussian (5x faster)

# Detection frequency
--detection-interval 10  # Detect faces every 10 frames (default: 7)

# Disable motion detection (not recommended)
--no-motion-detection  # Always use fixed detection interval
```

### Type Checking
```bash
uv run basedpyright
```

## Architecture Overview

This is a low-latency RTMP-to-WebRTC bridge service that achieves low end-to-end latency by directly piping raw video/audio from FFmpeg to custom aiortc tracks.

### Data Flow Pipeline
```
RTMP Input → FFmpeg → Raw YUV420p/PCM pipes → Custom Tracks → MediaRelay → WebRTC Peers
                ↓                                    ↓              ↓
           Transcript                          Blur/Capture    RTMP Relay
           (from PCM)                          (once, not per-peer) (optional)
                ↓
        Consent Detection
        (monitors transcript)
```

### Core Components

**main.py** - Single-file service containing:
- Web server starts immediately on port 8080
- FFmpeg subprocess (with `-listen 1`) acts as RTMP server on port 1935
- FFmpeg outputs raw YUV420p video to stdout and PCM audio to pipes
- Custom track classes (RawVideoStreamTrack, PCMAudioStreamTrack) read from pipes
- UnifiedTransformTrack applies blur/capture once per stream
- MediaRelay distributes to multiple WebRTC clients with O(1) CPU scaling
- Built-in web interface with auto-retry logic

**Supporting Components**:
- **allowlist.py** - Manages face embeddings and matching for the allow-list
- **transcript.py** - Real-time speech-to-text using Vosk (offline)
- **consent_detector.py** - AI-powered consent detection using OpenAI API
- **align_faces.py** - Utility to preprocess face images for better matching
- **rtmp_relay.py** - RTMP relay functionality for streaming to external servers
- **index.html** - Web interface with automatic reconnection logic

### Key Implementation Details

#### Custom Track Classes

**RawVideoStreamTrack** reads raw YUV420p from FFmpeg's stdout:
- Frame size calculation: `width * height * 1.5` bytes (YUV420p format)
- Buffered reading to handle full frames
- Direct memory copy for Y, U, V planes
- Shared monotonic clock for A/V synchronization

**PCMAudioStreamTrack** reads raw PCM audio from pipe:
- 48kHz mono, 16-bit samples
- 20ms frames (960 samples) for WebRTC compatibility
- Aligns with video clock on first frame
- Uses counter-based timestamps after alignment

**TranscriptWriter** generates real-time transcripts:
- Uses wall clock time (datetime.now) for timestamps
- Ensures timestamps align with capture images
- Processes audio through Vosk in real-time

**UnifiedTransformTrack** applies transforms efficiently:
- Multi-resolution face detection (configurable scale, default 50%)
- Motion detection skips static frames (up to 80% reduction)
- Adaptive detection interval (2-20 frames based on scene)
- Optimized blur kernel (31x31 vs 51x51, 3x faster)
- Optional box blur (5x faster than Gaussian)
- Caches blur regions between detections
- Asynchronous JPEG capture to avoid blocking
- Handles frame size changes gracefully

#### FFmpeg Configuration

Multiple audio outputs for transcript:
- Main audio: 48kHz for WebRTC
- Transcript audio: 16kHz for Vosk (via separate pipe)

#### Consent Detection

When `--transcript` is enabled, automatic consent detection monitors for recording consent:
- **Monitoring interval**: Checks every 3 seconds
- **Context window**: Analyzes last 3 lines of transcript
- **AI model**: Uses OpenAI's gpt-4o-mini for efficiency
- **Consent patterns**: Detects phrases like "I consent to be recorded", "I give permission to being captured on camera", etc.
- **Smart caching**: Only processes new transcript content
- **Logging format**: `[2025-07-10-22-15-30] Consent detected: Yes | Name: Takeshi | Time: 2025-07-10-21-59-09`
- **Image capture**: When consent is detected and `--capture` is enabled:
  - Finds the closest capture image within ±2 seconds of consent timestamp
  - Detects faces using YuNet and selects the main face (largest + most central)
  - Crops and saves face as 256x256 image with gray padding (compatible with allowlist format)
  - Saves to timestamped directory: `./consented_captures/YYYY-MM-DD-HH-MM-SS/`
  - File format: `YYYY-MM-DD-HH-MM-SS_PersonName.jpg`
  - Uses "anonymous" if no name is detected
  - Skips saving if no suitable face is found
- **Requirements**: Set `OPENAI_API_KEY` environment variable

#### Dynamic Allowlist Updates

When both `--transcript` and `--blur` are enabled, consented faces are dynamically added to the allowlist:
- **Real-time flow**: Face blurred → Person consents → Face automatically unblurred with name
- **Implementation**:
  - Each run creates a fresh timestamped `consented_captures` directory
  - AllowListManager monitors this directory using watchdog
  - New consent images are detected instantly (file creation events)
  - Face encodings generated and added to active allowlist
  - Thread-safe updates ensure smooth video processing
- **Performance**: 
  - Face encoding happens outside video processing thread
  - Only locks during list updates (minimal impact)
  - Watchdog provides instant file detection (no polling delay)
- **Isolation**: Each main.py run uses its own timestamped directory
  - Previous run's consents don't carry over
  - Clean slate for each streaming session

#### Face Detection & Blurring

- **YuNet model**: Fast CNN-based face detector
- **face_recognition**: dlib-based embeddings for matching
- **Allowlist structure**: `allowlist/PersonName/` with multiple images per person
- **Distance threshold**: 0.45 default (lower = stricter matching)
- **Performance optimizations (v2)**:
  - Multi-resolution detection (0.25x-1.0x configurable)
  - Motion detection for static scene optimization
  - Adaptive frame skipping (2-20 frames)
  - Optimized blur kernels (21x21 to 51x51)
  - Box blur option (5x faster)
  - Downsampled motion detection (16x fewer pixels)
  - Cached blur regions between detections

#### RTMP Relay

The system can relay processed video to external RTMP servers using dynamic playpath detection:
- **Playpath detection**: FFmpeg outputs "Unexpected stream X, expecting stream" 
- **Format**: Stream to `rtmp://localhost:1935/stream/target_server:port__path__key`
- **Example**: `rtmp://localhost:1935/stream/live.example.com:1935__live__mykey`
- **Requirements**: Relay only works when `--blur` or `--capture` is enabled
- **Implementation**:
  - Fixed listen path `/stream` in FFmpeg
  - Regex pattern extracts playpath from stderr
  - `rtmp_relay.py` contains `RTMPRelayHandler`
  - Second FFmpeg process handles relay output
  - Processes already-transformed frames (no duplicate processing)
  - Synchronized audio/video relay
  - Automatic start/stop with stream lifecycle
- **Performance**: Minimal overhead as frames are already processed
- **Failure handling**: Relay failures don't affect main stream

### Stream Lifecycle Management

1. **Stream Start**:
   - FFmpeg starts in listen mode
   - Waits for RTMP connection
   - Creates tracks when data flows
   - Sets `stream_ready = True`

2. **Active Streaming**:
   - Transforms applied once per frame
   - MediaRelay distributes to all peers
   - Periodic sync status logging

3. **Stream End**:
   - FFmpeg process exits
   - `track.stop()` sends RTCP BYE
   - Peer connections closed cleanly
   - Shared clock reset
   - FFmpeg restarts after 2s delay

### Reconnection Mechanism

Automatic reconnection without page reload:
1. Server detects stream end via FFmpeg process exit
2. Server stops tracks (sends RTCP BYE to browsers)
3. Browser detects `track.onended` event
4. Browser shows "Waiting..." and retries every 2 seconds
5. When new RTMP stream arrives, browser reconnects automatically

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Pipe buffer overflow** | Use `-thread_queue_size 512` + buffered reading in tracks |
| **A/V desynchronization** | Shared monotonic clock + aligned timestamps |
| **High CPU on face blur** | Frame skipping + cached regions + optimized kernels |
| **Memory leaks** | Proper cleanup in all error paths + weak references |
| **Async blocking** | All I/O is async (aiofiles for capture) |
| **Connection state issues** | RTCP BYE + proper event handlers |
