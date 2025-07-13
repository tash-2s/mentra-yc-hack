"""
Low-latency RTMP to WebRTC bridge using aiortc

This implementation:
1. Uses FFmpeg to receive RTMP and output raw video/audio
2. Creates custom tracks from raw pipes for minimal latency
3. Applies transforms once per stream (not per peer)
4. Relays to WebRTC clients with sub-300ms latency
"""

import argparse
import asyncio
import fractions
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import aiofiles
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame

from rtmp_relay import RTMPRelayHandler

logger = logging.getLogger("rtmp-webrtc")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(message)s")

# Global state
video_track: Optional[VideoStreamTrack] = None
audio_track: Optional[AudioStreamTrack] = None
relay = MediaRelay()
rtmp_process: Optional[asyncio.subprocess.Process] = None
stream_ready = False
active_peers: set[RTCPeerConnection] = set()
shutdown_event = asyncio.Event()
transcript_writer = None
consent_detector = None
allow_manager = None

# Shared monotonic clock for A/V synchronization
STREAM_START_TIME = None

def get_stream_elapsed_time():
    """Get elapsed time since stream start using a shared monotonic clock"""
    global STREAM_START_TIME
    if STREAM_START_TIME is None:
        STREAM_START_TIME = time.perf_counter()
    return time.perf_counter() - STREAM_START_TIME

def reset_stream_clock():
    """Reset the shared clock when a new stream starts"""
    global STREAM_START_TIME
    STREAM_START_TIME = None


class RawVideoStreamTrack(VideoStreamTrack):
    """Reads raw YUV420p video from FFmpeg pipe"""

    def __init__(self, process, width=1280, height=720, fps=30):
        super().__init__()
        self.process = process
        self.width = width
        self.height = height
        self.fps = fps
        # YUV420p: 1.5 bytes per pixel
        self.frame_size = int(width * height * 1.5)

    async def recv(self):
        # Use shared monotonic clock for A/V sync
        elapsed = get_stream_elapsed_time()
        pts = int(elapsed * self.fps)
        time_base = fractions.Fraction(1, self.fps)

        # Read exactly one frame worth of YUV420p data with buffering
        try:
            # Read in chunks until we have a full frame
            raw_data = b''
            bytes_needed = self.frame_size

            while bytes_needed > 0:
                chunk = await self.process.stdout.read(bytes_needed)
                if not chunk:
                    if len(raw_data) == 0:
                        logger.info("Video stream ended cleanly")
                    else:
                        logger.info(f"Video stream ended (partial frame: {len(raw_data)} of {self.frame_size} bytes)")
                    raise StopAsyncIteration
                raw_data += chunk
                bytes_needed -= len(chunk)

            if len(raw_data) != self.frame_size:
                logger.error(f"Video frame size mismatch: got {len(raw_data)}, expected {self.frame_size}")
                raise StopAsyncIteration
        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.error(f"Error reading video data: {e}")
            raise StopAsyncIteration

        # Create VideoFrame from raw YUV420p
        frame = VideoFrame(width=self.width, height=self.height, format='yuv420p')

        # Split YUV components
        y_size = self.width * self.height
        u_size = v_size = (self.width // 2) * (self.height // 2)

        # Direct memory copy for efficiency
        y_data = raw_data[:y_size]
        u_data = raw_data[y_size:y_size + u_size]
        v_data = raw_data[y_size + u_size:y_size + u_size + v_size]

        frame.planes[0].update(y_data)
        frame.planes[1].update(u_data)
        frame.planes[2].update(v_data)

        frame.pts = pts
        frame.time_base = time_base
        return frame



class PCMAudioStreamTrack(AudioStreamTrack):
    """Read raw PCM audio (simpler but more bandwidth)"""

    def __init__(self, audio_fd, relay_handler=None):
        super().__init__()
        self.pipe = os.fdopen(audio_fd, 'rb')
        self.sample_rate = 48000
        self.samples_per_frame = 960  # 20ms at 48kHz
        self.bytes_per_frame = self.samples_per_frame * 2  # 16-bit samples
        self._time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp = 0
        self._start_aligned = False
        self.relay_handler = relay_handler

    async def recv(self):
        # Align with video clock on first frame, then use counter
        if not self._start_aligned:
            # Get initial timestamp aligned with video
            elapsed = get_stream_elapsed_time()
            self._timestamp = int(elapsed * self.sample_rate)
            self._start_aligned = True

        pts = self._timestamp
        self._timestamp += self.samples_per_frame  # Increment by exactly one frame
        time_base = self._time_base

        # Read PCM data with buffering
        try:
            # Read in chunks until we have a full frame
            pcm_data = b''
            bytes_needed = self.bytes_per_frame

            while bytes_needed > 0:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self.pipe.read, bytes_needed
                )
                if not chunk:
                    if len(pcm_data) == 0:
                        logger.info("Audio stream ended cleanly")
                    else:
                        logger.info(f"Audio stream ended (partial frame: {len(pcm_data)} of {self.bytes_per_frame} bytes)")
                    raise StopAsyncIteration
                pcm_data += chunk
                bytes_needed -= len(chunk)

            if len(pcm_data) != self.bytes_per_frame:
                logger.error(f"Audio frame size mismatch: got {len(pcm_data)}, expected {self.bytes_per_frame}")
                raise StopAsyncIteration
        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.error(f"Error reading audio data: {e}")
            raise StopAsyncIteration

        # Send to relay if active
        if self.relay_handler and self.relay_handler.is_active():
            asyncio.create_task(self.relay_handler.queue_audio(pcm_data))

        # Create AudioFrame from PCM
        frame = AudioFrame(format='s16', layout='mono', samples=self.samples_per_frame)
        frame.planes[0].update(pcm_data)
        frame.sample_rate = self.sample_rate
        frame.pts = pts
        frame.time_base = self._time_base
        return frame

    def stop(self):
        """Stop the audio track and reset state"""
        super().stop()
        self._start_aligned = False
        self._timestamp = 0


class UnifiedTransformTrack(VideoStreamTrack):
    """Applies blur and capture transforms once per stream with performance optimizations"""

    def __init__(self, source_track, blur_enabled=False, capture_enabled=False,
                 allow_manager=None, capture_dir=None, capture_interval=1.0, capture_quality=85,
                 detection_scale=0.5, blur_kernel_size=31, use_box_blur=False,
                 motion_detection_enabled=True, detection_interval=7, relay_handler=None):
        super().__init__()
        self.source = source_track
        self.blur_enabled = blur_enabled
        self.capture_enabled = capture_enabled
        self.allow_manager = allow_manager
        self.relay_handler = relay_handler
        self.capture_dir = Path(capture_dir) if capture_dir else Path('./captures')
        self.capture_interval = capture_interval
        self.capture_quality = capture_quality
        self.last_capture = 0

        # Performance optimization: frame skipping for face detection
        self.frame_count = 0
        self.face_detection_interval = detection_interval  # Configurable
        self.cached_blur_regions = []  # Cache blur regions between detections
        self.last_frame_size = None

        # Motion detection for skipping static frames
        self.motion_detection_enabled = motion_detection_enabled
        self.prev_frame_gray = None
        self.motion_threshold = 3.0  # Threshold for frame difference
        self.static_frame_count = 0
        self.max_detection_interval = detection_interval * 2  # Double the base interval for static scenes

        # Multi-resolution detection settings
        self.detection_scale = detection_scale  # Configurable scale

        # Optimized blur parameters
        blur_kernel_size = max(3, blur_kernel_size)  # Ensure minimum size
        if blur_kernel_size % 2 == 0:  # Ensure odd number
            blur_kernel_size += 1
        self.blur_kernel_size = (blur_kernel_size, blur_kernel_size)
        self.blur_sigma = blur_kernel_size // 3  # Adaptive sigma

        # Create capture directory if needed
        if self.capture_enabled:
            self.capture_dir.mkdir(parents=True, exist_ok=True)

        # Initialize face detection if blur is enabled
        if self.blur_enabled:
            try:
                logger.info("Initializing YuNet face detection for blur functionality")
                model_path = Path("face_detection_yunet_2023mar.onnx")
                if not model_path.exists():
                    raise RuntimeError(
                        "YuNet model not found. Please run: uv run download_yunet_model.py"
                    )

                self.detector = cv2.FaceDetectorYN_create(  # type: ignore
                    model=str(model_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )

                # Option to use box blur for even better performance
                self.use_box_blur = use_box_blur
                logger.info(f"YuNet face detector initialized (detecting every {self.face_detection_interval} frames)")
            except Exception as e:
                logger.error(f"Failed to initialize YuNet face detector: {e}")
                raise RuntimeError(f"Cannot initialize face blur: {e}")

    async def recv(self):
        frame = await self.source.recv()

        # Validate frame type
        if not isinstance(frame, VideoFrame):
            logger.warning(f"Received non-VideoFrame: {type(frame)}")
            return frame

        # Convert to numpy for processing
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Check if frame size changed (e.g., resolution change)
        if self.last_frame_size != (w, h):
            self.last_frame_size = (w, h)
            self.cached_blur_regions = []  # Clear cache on size change

        # Apply capture first (before any modifications)
        if self.capture_enabled:
            now = time.time()
            if now - self.last_capture >= self.capture_interval:
                # Only copy when actually capturing
                asyncio.create_task(self._capture_frame_async(img.copy()))
                self.last_capture = now

        # Apply blur if enabled
        if self.blur_enabled and self.allow_manager:
            try:
                # Motion detection to skip static frames
                motion_detected = True
                if self.motion_detection_enabled:
                    # Downsample for faster motion detection
                    gray_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (w//4, h//4))

                    if self.prev_frame_gray is not None:
                        # Calculate frame difference on downsampled images
                        diff = cv2.absdiff(self.prev_frame_gray, gray_small)
                        motion_score = cv2.mean(diff)[0]  # type: ignore
                        motion_detected = motion_score > self.motion_threshold

                        if not motion_detected:
                            self.static_frame_count += 1
                        else:
                            self.static_frame_count = 0

                    self.prev_frame_gray = gray_small

                # Adaptive detection interval based on motion
                if not motion_detected and self.static_frame_count > 5:
                    # Static scene: use maximum interval
                    adaptive_interval = self.max_detection_interval
                else:
                    # Dynamic scene or faces present: use normal interval
                    adaptive_interval = self.face_detection_interval

                # Only detect faces every N frames for performance
                should_detect = (self.frame_count % adaptive_interval == 0) or self.static_frame_count == 0
                self.frame_count += 1

                if should_detect:
                    # Multi-resolution detection: detect at lower resolution
                    if self.detection_scale < 1.0:
                        # Downscale for detection
                        detect_w = int(w * self.detection_scale)
                        detect_h = int(h * self.detection_scale)
                        detect_img = cv2.resize(img, (detect_w, detect_h))

                        # Perform YuNet face detection on smaller image
                        self.detector.setInputSize((detect_w, detect_h))  # type: ignore
                        _, faces = self.detector.detect(detect_img)  # type: ignore

                        # Scale coordinates back to original size
                        if faces is not None:
                            faces[:, 0:4] /= self.detection_scale
                        else:
                            faces = []
                    else:
                        # Original resolution detection
                        self.detector.setInputSize((w, h))  # type: ignore
                        _, faces = self.detector.detect(img)  # type: ignore
                        if faces is None:
                            faces = []

                    # Update cached blur regions
                    self.cached_blur_regions = []

                    # Adjust detection interval based on whether faces were found
                    if len(faces) == 0:
                        # No faces found, check more frequently
                        self.face_detection_interval = 5  # Increased from 2
                    else:
                        # Faces found, can check less frequently
                        self.face_detection_interval = 7  # Increased from 3

                    # Check each face against allowlist
                    for face in faces:
                        x, y, w, h = map(int, face[0:4])  # YuNet format

                        # Ensure face region is within image bounds
                        x, y = max(0, x), max(0, y)
                        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)

                        if x2 > x and y2 > y:  # Valid region
                            face_img = img[y:y2, x:x2]
                            if face_img.size > 0:
                                # Convert BGR to RGB for face_recognition
                                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                try:
                                    name = self.allow_manager.match(face_img_rgb)
                                    if name:
                                        # Store allowed face info for display
                                        self.cached_blur_regions.append({
                                            'type': 'allowed',
                                            'name': name,
                                            'x': x, 'y': y, 'w': w, 'h': h
                                        })
                                    else:
                                        # Calculate and cache blur region
                                        padding_x = int(w * 0.4)
                                        padding_y = int(h * 0.4)
                                        blur_x = max(0, x - padding_x)
                                        blur_y = max(0, y - padding_y)
                                        blur_w = min(img.shape[1] - blur_x, w + 2 * padding_x)
                                        blur_h = min(img.shape[0] - blur_y, h + 2 * padding_y)

                                        self.cached_blur_regions.append({
                                            'type': 'blur',
                                            'x': blur_x, 'y': blur_y, 'w': blur_w, 'h': blur_h
                                        })
                                except Exception as e:
                                    # If matching fails, cache blur region for safety
                                    logger.warning(f"Face matching error: {e}")
                                    padding_x = int(w * 0.4)
                                    padding_y = int(h * 0.5)
                                    blur_x = max(0, x - padding_x)
                                    blur_y = max(0, y - padding_y)
                                    blur_w = min(img.shape[1] - blur_x, w + 2 * padding_x)
                                    blur_h = min(img.shape[0] - blur_y, h + 2 * padding_y)

                                    self.cached_blur_regions.append({
                                        'type': 'blur',
                                        'x': blur_x, 'y': blur_y, 'w': blur_w, 'h': blur_h
                                    })

                # Apply cached regions (runs every frame)
                for region in self.cached_blur_regions:
                    if region['type'] == 'allowed':
                        # Show name and box for allowed faces
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.putText(img, region['name'], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.4, (0,255,0), 3)
                        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
                    else:  # blur
                        # Apply blur to cached region
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        blur_roi = img[y:y+h, x:x+w]
                        if blur_roi.size > 0:
                            if self.use_box_blur:
                                # Box blur is 5x faster than Gaussian
                                img[y:y+h, x:x+w] = cv2.boxFilter(blur_roi, -1, self.blur_kernel_size)
                            else:
                                # Optimized Gaussian blur
                                img[y:y+h, x:x+w] = cv2.GaussianBlur(blur_roi, self.blur_kernel_size, self.blur_sigma)
            except Exception as e:
                logger.error(f"Error in face blur: {e}")

        # Convert back to VideoFrame
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts if frame.pts is not None else 0
        new_frame.time_base = frame.time_base

        # Send to relay if active
        if self.relay_handler and self.relay_handler.is_active():
            asyncio.create_task(self.relay_handler.queue_video_frame(new_frame))

        return new_frame

    async def _capture_frame_async(self, img):
        """Write JPEG asynchronously to avoid blocking"""
        try:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
            filename = self.capture_dir / f"{timestamp}.jpg"

            # Encode in memory first
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.capture_quality])

            # Write asynchronously
            async with aiofiles.open(filename, 'wb') as f:
                await f.write(buffer.tobytes())
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")


async def start_rtmp_receiver(args, width=1280, height=720):
    """Start FFmpeg to receive RTMP and output raw video/audio"""
    global rtmp_process, transcript_writer, consent_detector

    # Create pipes for audio
    audio_read, audio_write = os.pipe()
    os.set_inheritable(audio_write, True)
    logger.debug(f"Created audio pipe: read_fd={audio_read}, write_fd={audio_write}")

    transcript_read, transcript_write = None, None
    if args.transcript:
        transcript_read, transcript_write = os.pipe()
        os.set_inheritable(transcript_write, True)

    cmd = [
        "ffmpeg",
        # Critical low latency flags
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-probesize", "32768",  # 32KB instead of 32 bytes
        "-analyzeduration", "0",

        # RTMP input with listen mode
        "-listen", "1",
        "-i", "rtmp://0.0.0.0:1935/live/stream",

        # Thread queue to prevent backpressure
        "-thread_queue_size", "512",

        # Video output settings
        "-map", "0:v:0",
        "-c:v", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-r", str(args.fps),
        "-f", "rawvideo",
        "-",  # stdout for video

        # Audio output - PCM (aiortc will encode to Opus automatically)
        "-map", "0:a:0",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-ar", "48000",
        "-f", "s16le",
        f"pipe:{audio_write}"
    ]

    # For transcript, add another PCM output
    if args.transcript:
        cmd.extend([
            "-map", "0:a:0",
            "-c:a", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            f"pipe:{transcript_write}"
        ])

    logger.info("Starting FFmpeg")

    rtmp_process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        pass_fds=[audio_write] + ([transcript_write] if transcript_write else [])
    )

    # Close write ends in parent
    os.close(audio_write)
    if transcript_write:
        os.close(transcript_write)

    # Start transcript writer if enabled
    if args.transcript and transcript_read:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        transcript_file = Path("./transcripts") / f"transcript-{timestamp}.txt"
        transcript_file.parent.mkdir(parents=True, exist_ok=True)

        from transcript import TranscriptWriter
        from consent_detector import ConsentDetector

        transcript_writer = TranscriptWriter(
            model_path=Path(args.transcript_model),
            transcript_file=transcript_file
        )
        # Create a subprocess for transcript to isolate it
        asyncio.create_task(start_transcript_subprocess(transcript_read, transcript_writer))
        logger.info(f"Transcript will be saved to: {transcript_file}")

        # Start consent detector with timestamped directory
        global consent_detector
        consent_detector = ConsentDetector(
            consent_capture_dir=args.timestamped_consent_dir
        )
        asyncio.create_task(consent_detector.monitor_loop())
        logger.info(f"Consent detection enabled with output to: {args.timestamped_consent_dir}")

    return rtmp_process, audio_read, transcript_read


async def start_transcript_subprocess(transcript_fd, writer):
    """Run transcript writer in isolated subprocess"""
    try:
        # Read from pipe and write to transcript
        pipe = os.fdopen(transcript_fd, 'rb')
        writer.start_from_pipe(pipe)
    except Exception as e:
        logger.error(f"Transcript subprocess error: {e}")


async def handle_rtmp_stream(process, audio_fd, args, relay_handler=None):
    """Handle incoming RTMP stream and create tracks"""
    global video_track, audio_track, stream_ready, allow_manager

    # Pattern to match: "Unexpected stream foobar, expecting stream"
    PLAYPATH_PATTERN = re.compile(r"Unexpected stream ([^,]+), expecting stream")
    detected_playpath = None

    # Monitor stderr for stream start and playpath
    async def monitor_stderr():
        nonlocal detected_playpath
        stream_started = False
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            line_str = line.decode().strip()
            if line_str:
                # Check for playpath
                match = PLAYPATH_PATTERN.search(line_str)
                if match:
                    detected_playpath = match.group(1)
                    logger.info(f"Detected playpath: {detected_playpath.replace("__", "/")}")

                # Filter and log important messages
                if "Error during demuxing: Input/output error" in line_str:
                    # This is expected when RTMP client disconnects
                    logger.info("RTMP client disconnected")
                elif any(indicator in line_str for indicator in [
                    "Error", "error", "failed", "Failed"
                ]):
                    logger.warning(f"ffmpeg: {line_str}")
                elif any(indicator in line_str for indicator in [
                    "Stream mapping:", "Output #0", "frame=", "Writing output"
                ]):
                    logger.info(f"ffmpeg: {line_str}")

                # Check if stream has started
                if not stream_started and ("Stream mapping:" in line_str or "Output #0" in line_str):
                    logger.info("RTMP stream connected and encoding started")
                    stream_started = True
                    # Continue monitoring to see all output

        return stream_started

    # Start continuous stderr monitoring
    stderr_task = asyncio.create_task(monitor_stderr())

    # Wait for stream to actually start producing data
    logger.info("Waiting for FFmpeg to start producing data...")
    await asyncio.sleep(2.0)

    # Check if process is still running
    if process.returncode is not None:
        logger.error(f"FFmpeg exited early with code: {process.returncode}")
        stderr = await process.stderr.read()
        if stderr:
            logger.error(f"FFmpeg stderr: {stderr.decode()}")
        return

    # Create relay handler from command line arg or detected playpath
    if args.relay_target:
        # Use provided relay target
        logger.info(f"Creating relay handler for target: {args.relay_target}")
        relay_handler = RTMPRelayHandler(args.relay_target, args.width, args.height, args.fps)
    # elif detected_playpath:
    #     # Convert playpath to relay target (replace __ with /)
    #     relay_target = detected_playpath.replace("__", "/")
    #     logger.info(f"Creating relay handler for target: {relay_target}")
    #     relay_handler = RTMPRelayHandler(relay_target, args.width, args.height, args.fps)

    # Create raw tracks from FFmpeg output
    logger.info(f"Creating video and audio tracks with shared clock ({args.width}x{args.height})")
    raw_video = RawVideoStreamTrack(process, width=args.width, height=args.height, fps=30)
    raw_audio = PCMAudioStreamTrack(audio_fd, relay_handler=relay_handler)

    # Apply transforms if needed
    if args.blur or args.capture:
        if args.blur:
            from allowlist import AllowListManager
            consented_dir = getattr(args, 'timestamped_consent_dir', None)
            allow_manager = AllowListManager(
                Path(args.allow_dir),
                args.distance,
                consented_captures_dir=consented_dir
            )
            logger.info(f"Initialized AllowListManager with directory: {args.allow_dir}, distance: {args.distance}")
            if consented_dir:
                logger.info(f"AllowListManager will monitor for new consents in: {consented_dir}")
        else:
            allow_manager = None

        video_track = UnifiedTransformTrack(
            raw_video,
            blur_enabled=args.blur,
            capture_enabled=args.capture,
            allow_manager=allow_manager,
            capture_dir=Path(args.capture_dir),
            capture_interval=args.capture_interval,
            capture_quality=args.capture_quality,
            detection_scale=args.detection_scale,
            blur_kernel_size=args.blur_kernel,
            use_box_blur=args.box_blur,
            motion_detection_enabled=not args.no_motion_detection,
            detection_interval=args.detection_interval,
            relay_handler=relay_handler
        )
        logger.info(f"Created unified transform track (blur={args.blur}, capture={args.capture})")

        # Start relay if configured (only works with transforms)
        if relay_handler:
            try:
                await relay_handler.start()
                logger.info(f"✓ RTMP relay started to: {relay_handler.target_url}")
            except Exception as e:
                logger.error(f"Failed to start RTMP relay: {e}")
                # Continue without relay
    else:
        video_track = raw_video
        if relay_handler:
            logger.warning("RTMP relay requires --blur or --capture to be enabled. Relay will not work.")

    audio_track = raw_audio
    stream_ready = True
    logger.info("✓ Stream ready for WebRTC connections")

    # Log sync status periodically
    async def monitor_sync():
        while stream_ready:
            await asyncio.sleep(10)
            if stream_ready:
                elapsed = get_stream_elapsed_time()
                logger.debug(f"Stream sync check - elapsed: {elapsed:.2f}s")

    asyncio.create_task(monitor_sync())

    # Return relay handler for cleanup
    return relay_handler


async def monitor_rtmp_process(process, relay_handler=None):
    """Monitor RTMP process and handle lifecycle"""
    global video_track, audio_track, stream_ready, transcript_writer, consent_detector, allow_manager

    # Wait for process to exit
    await process.wait()
    exit_code = process.returncode

    if exit_code == 0 or exit_code == 255:
        logger.info("Stream ended normally")
    else:
        logger.warning(f"FFmpeg exited with error code: {exit_code}")

    # Clean shutdown
    stream_ready = False

    # Stop tracks first (sends RTCP BYE)
    if video_track:
        video_track.stop()
        logger.info("Stopped video track")
    if audio_track:
        audio_track.stop()
        logger.info("Stopped audio track")

    # Stop transcript writer
    if transcript_writer:
        transcript_writer.stop()

    # Stop consent detector
    if consent_detector:
        consent_detector.stop()

    # Stop allow manager observer
    if allow_manager:
        allow_manager.stop()

    # Stop relay handler
    if relay_handler:
        await relay_handler.stop()

    # Close all peer connections
    for pc in list(active_peers):
        try:
            await pc.close()
        except Exception as e:
            logger.error(f"Error closing peer: {e}")
    active_peers.clear()

    video_track = None
    audio_track = None
    transcript_writer = None
    consent_detector = None
    allow_manager = None

    # Reset the shared clock for the next stream
    reset_stream_clock()

    # Restart after a short delay
    if not shutdown_event.is_set():
        logger.info("Ready for next RTMP connection in 2 seconds...")
        await asyncio.sleep(2)


async def rtmp_loop(args):
    """Main RTMP receiver loop"""
    while not shutdown_event.is_set():
        try:
            logger.info("Starting RTMP listener on port 1935...")

            # Start FFmpeg
            process, audio_fd, _transcript_fd = await start_rtmp_receiver(
                args, width=args.width, height=args.height
            )

            # Handle the stream
            relay_handler = await handle_rtmp_stream(process, audio_fd, args)

            # Monitor until it exits
            await monitor_rtmp_process(process, relay_handler)

        except Exception as e:
            logger.error(f"Error in RTMP loop: {e}", exc_info=True)
            if not shutdown_event.is_set():
                await asyncio.sleep(5)


async def offer(request: web.Request):
    """Handle WebRTC offer"""
    global video_track, audio_track, stream_ready

    params = await request.json()

    # Check if stream is ready
    if not stream_ready or not video_track or not audio_track:
        logger.info("WebRTC offer received but no RTMP stream available yet")
        return web.json_response({"error": "No stream available"}, status=503)

    pc = RTCPeerConnection()
    active_peers.add(pc)
    logger.info(f"Creating peer connection (total active: {len(active_peers)})")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ["closed", "failed"]:
            active_peers.discard(pc)
            logger.info(f"Removed peer (remaining: {len(active_peers)})")

    # Track ended events for reconnection
    @pc.on("track")
    def on_track(track):
        @track.on("ended")
        async def on_ended():
            logger.info("Track ended on client side")

    # Add relayed tracks with buffering to smooth jitter
    if video_track:
        pc.addTrack(relay.subscribe(video_track, buffered=True))
        logger.info("Added video track to peer")
    if audio_track:
        pc.addTrack(relay.subscribe(audio_track, buffered=True))
        logger.info("Added audio track to peer")

    # Set remote description and create answer
    offer_obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer_obj)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def index(_request):
    """Serve the test page"""
    html_path = Path(__file__).parent / "index.html"
    html = html_path.read_text()
    return web.Response(text=html, content_type="text/html")


async def cleanup():
    """Cleanup resources"""
    global rtmp_process, video_track, audio_track, transcript_writer, consent_detector, allow_manager

    # Signal shutdown
    shutdown_event.set()

    # Stop tracks
    if video_track:
        video_track.stop()
    if audio_track:
        audio_track.stop()

    # Stop transcript writer
    if transcript_writer:
        transcript_writer.stop()

    # Stop consent detector
    if consent_detector:
        consent_detector.stop()

    # Stop allow manager observer
    if allow_manager:
        allow_manager.stop()

    # Close all peers
    for pc in list(active_peers):
        try:
            await pc.close()
        except Exception as e:
            logger.warning(f"Error closing peer: {e}")
    active_peers.clear()

    # Terminate FFmpeg
    if rtmp_process:
        try:
            rtmp_process.terminate()
            await asyncio.wait_for(rtmp_process.wait(), timeout=2)
        except asyncio.TimeoutError:
            rtmp_process.kill()
            await rtmp_process.wait()
        except Exception as e:
            logger.warning(f"Error terminating ffmpeg: {e}")


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Low-latency RTMP to WebRTC bridge")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--blur", action="store_true",
                        help="Enable face blur with allow-list")
    parser.add_argument("--allow-dir", default="./allowlist",
                        help="Directory with allow-list images")
    parser.add_argument("--distance", type=float, default=0.45,
                        help="Face matching threshold (lower = stricter)")
    parser.add_argument("--capture", action="store_true",
                        help="Enable frame capture")
    parser.add_argument("--capture-dir", default="./captures",
                        help="Directory to save captured frames")
    parser.add_argument("--capture-interval", type=float, default=1.0,
                        help="Seconds between frame captures")
    parser.add_argument("--capture-quality", type=int, default=85,
                        help="JPEG quality for captures (1-100)")
    parser.add_argument("--transcript", action="store_true",
                        help="Enable speech-to-text transcription")
    parser.add_argument("--transcript-model", default="./vosk-model-en-us-0.22-lgraph",
                        help="Path to Vosk model directory")
    parser.add_argument("--width", type=int, default=1280,
                        help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Video height (default: 720)")

    # Performance tuning options
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30, lower for better performance)")
    parser.add_argument("--detection-scale", type=float, default=0.5,
                        help="Scale factor for face detection (default: 0.5 = 50%%, lower is faster)")
    parser.add_argument("--blur-kernel", type=int, default=31,
                        help="Blur kernel size (default: 31, must be odd, smaller is faster)")
    parser.add_argument("--box-blur", action="store_true",
                        help="Use box blur instead of Gaussian (5x faster)")
    parser.add_argument("--no-motion-detection", action="store_true",
                        help="Disable motion detection optimization")
    parser.add_argument("--detection-interval", type=int, default=7,
                        help="Frames between face detections (default: 7, higher is faster)")
    parser.add_argument("--relay-target",
                        help="RTMP relay target URL (e.g., live.example.com:1935/live/key)")
    args = parser.parse_args()

    # Create timestamped consented captures directory if transcript is enabled
    timestamped_consent_dir = None
    if args.transcript:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        timestamped_consent_dir = Path("./consented_captures") / timestamp
        timestamped_consent_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created timestamped consent directory: {timestamped_consent_dir}")

    # Validate transcript model if enabled
    if args.transcript:
        if not Path(args.transcript_model).exists():
            logger.error(f"Vosk model not found: {args.transcript_model}")
            logger.error("Download a model from: https://alphacephei.com/vosk/models")
            logger.error("Or run: uv run download_vosk_model.py")
            return

    # Create web app
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    # Start web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    logger.info("="*50)
    logger.info("Low-Latency RTMP to WebRTC Bridge")
    logger.info(f"Web interface: http://localhost:{args.port}")
    logger.info(f"RTMP endpoint: rtmp://localhost:1935/stream")
    if args.relay_target:
        logger.info(f"Relay target: {args.relay_target}")
    else:
        logger.info("Relay format: rtmp://localhost:1935/stream/target.com:port__path__key")
    logger.info("="*50)

    # Add timestamped consent directory to args for easy passing
    args.timestamped_consent_dir = timestamped_consent_dir

    # Start RTMP receiver loop
    rtmp_task = asyncio.create_task(rtmp_loop(args))

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await cleanup()
        rtmp_task.cancel()
        try:
            await rtmp_task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
