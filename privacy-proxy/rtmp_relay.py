"""
RTMP relay functionality for streaming processed video to external RTMP servers.

This module manages FFmpeg processes to relay the processed video/audio streams
to external RTMP servers based on playpath detection.
"""

import asyncio
import logging
import os
from typing import Optional

from av import VideoFrame

logger = logging.getLogger("rtmp-webrtc")


class RTMPRelayHandler:
    """Handles relaying processed video/audio to an external RTMP server"""
    
    def __init__(self, target_url: str, width: int, height: int, fps: int):
        self.target_url = target_url
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self.video_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._writer_task: Optional[asyncio.Task] = None
        self._active = False
        
    async def start(self):
        """Start FFmpeg relay process"""
        try:
            cmd = [
                "ffmpeg",
                # Input settings for raw video
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "-",  # Video from stdin
                
                # Input settings for raw audio
                "-f", "s16le",
                "-ar", "48000",
                "-ac", "1",
                "-i", "pipe:3",  # Audio from separate pipe
                
                # Output video settings
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                
                # Output audio settings
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "48000",
                "-ac", "2",
                
                # Output format
                "-f", "flv",
                "-flvflags", "no_duration_filesize",
                f"rtmp://{self.target_url}"
            ]
            
            # Create pipe for audio
            audio_read, audio_write = os.pipe()
            os.set_inheritable(audio_read, True)
            
            logger.info(f"Starting RTMP relay to: {self.target_url}")
            
            self.ffmpeg_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=[audio_read]
            )
            
            # Close read end in parent
            os.close(audio_read)
            
            # Store write end for audio
            self.audio_write_fd = audio_write
            
            # Start writer task
            self._active = True
            self._writer_task = asyncio.create_task(self._write_loop())
            
            # Monitor stderr
            asyncio.create_task(self._monitor_stderr())
            
        except Exception as e:
            logger.error(f"Failed to start RTMP relay: {e}")
            self._active = False
            raise
    
    async def _monitor_stderr(self):
        """Monitor FFmpeg stderr for errors"""
        if not self.ffmpeg_process or not self.ffmpeg_process.stderr:
            return
            
        while True:
            line = await self.ffmpeg_process.stderr.readline()
            if not line:
                break
            line_str = line.decode().strip()
            if line_str and ("error" in line_str.lower() or "fail" in line_str.lower()):
                logger.warning(f"Relay ffmpeg: {line_str}")
    
    async def _write_loop(self):
        """Write queued frames to FFmpeg"""
        try:
            while self._active:
                # Get video frame
                video_data = await self.video_queue.get()
                if video_data and self.ffmpeg_process and self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.write(video_data)
                    
                # Process any pending audio
                try:
                    while True:
                        audio_data = self.audio_queue.get_nowait()
                        if audio_data:
                            os.write(self.audio_write_fd, audio_data)
                except asyncio.QueueEmpty:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in relay write loop: {e}")
            self._active = False
    
    async def queue_video_frame(self, frame: VideoFrame):
        """Queue a video frame for relay"""
        if not self._active:
            return
            
        try:
            # Convert to YUV420p and get raw bytes
            img = frame.to_ndarray(format="yuv420p")
            raw_data = img.tobytes()
            
            # Queue the data (drop if full)
            try:
                self.video_queue.put_nowait(raw_data)
            except asyncio.QueueFull:
                # Drop frame if queue is full
                pass
                
        except Exception as e:
            logger.error(f"Error queuing video frame: {e}")
    
    async def queue_audio(self, audio_data: bytes):
        """Queue audio data for relay"""
        if not self._active:
            return
            
        try:
            self.audio_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            # Drop audio if queue is full
            pass
    
    async def stop(self):
        """Stop the relay"""
        self._active = False
        
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            await self.ffmpeg_process.wait()
        
        if hasattr(self, 'audio_write_fd'):
            try:
                os.close(self.audio_write_fd)
            except:
                pass
        
        logger.info("RTMP relay stopped")
    
    def is_active(self) -> bool:
        """Check if relay is active"""
        return self._active