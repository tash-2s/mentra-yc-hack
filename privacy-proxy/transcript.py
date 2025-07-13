# transcript.py
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from vosk import Model, KaldiRecognizer, SetLogLevel

logger = logging.getLogger("transcript")


class TranscriptWriter:
    """Handles real-time transcription using Vosk and writes to file"""

    def __init__(self, model_path: Path, transcript_file: Path,
                 sample_rate: int = 16000):

        self.model_path = Path(model_path)
        self.transcript_file = Path(transcript_file)
        self.sample_rate = sample_rate
        self.chunk_size = 4000  # ~0.25 seconds of audio
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.process = None
        self.pipe = None

        # Silence Vosk logs
        SetLogLevel(-1)

        # Load Vosk model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {self.model_path}")

        logger.info(f"Loading Vosk model from: {self.model_path}")
        self.model = Model(str(self.model_path))
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)  # Enable word-level timestamps

        # Create transcript directory if needed
        self.transcript_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"TranscriptWriter initialized. Output: {self.transcript_file}")

    def start(self, process):
        """Start transcription in a separate thread"""
        if self.running:
            logger.warning("Transcription already running")
            return

        self.process = process
        self.running = True
        self.thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.thread.start()
        logger.info("Transcription thread started")

    def stop(self):
        """Stop transcription"""
        if not self.running:
            return

        logger.info("Stopping transcription...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None

        logger.info("Transcription stopped")

    def start_from_pipe(self, pipe):
        """Start transcription from a pipe file descriptor"""
        if self.running:
            logger.warning("Transcription already running")
            return

        self.pipe = pipe
        self.running = True
        self.thread = threading.Thread(target=self._transcribe_loop_pipe, daemon=True)
        self.thread.start()
        logger.info("Transcription thread started (pipe mode)")

    def _transcribe_loop(self):
        """Main transcription loop running in separate thread"""
        try:
            with open(self.transcript_file, "a", encoding="utf-8") as f:
                logger.info("Transcription loop started, waiting for audio data...")
                chunks_received = 0

                while self.running and self.process and self.process.poll() is None:
                    # Read audio chunk from FFmpeg stdout
                    chunk = self.process.stdout.read(self.chunk_size)
                    if not chunk:
                        logger.debug("No audio data received")
                        break

                    chunks_received += 1
                    if chunks_received == 1:
                        logger.info(f"First audio chunk received: {len(chunk)} bytes")
                    elif chunks_received % 100 == 0:
                        logger.debug(f"Received {chunks_received} chunks")

                    # Process with Vosk
                    if self.recognizer.AcceptWaveform(chunk):
                        # Final result available
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()

                        if text:
                            # Use wall clock time for timestamp
                            utc_time = datetime.now(tz=timezone.utc)
                            timestamp = utc_time.strftime("%Y-%m-%d-%H-%M-%S")

                            # Write to file
                            line = f"[{timestamp}] {text}\n"
                            f.write(line)
                            f.flush()

                            logger.info(f"Transcript: {line.strip()}")

                logger.info(f"Transcription loop ending. Total chunks: {chunks_received}")

                # Process any remaining audio
                final_result = json.loads(self.recognizer.FinalResult())
                text = final_result.get("text", "").strip()
                if text:
                    utc_time = datetime.now(tz=timezone.utc)
                    timestamp = utc_time.strftime("%Y-%m-%d-%H-%M-%S")
                    line = f"[{timestamp}] {text}\n"
                    f.write(line)
                    f.flush()
                    logger.info(f"Final transcript: {line.strip()}")

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Transcription loop ended")

    def _transcribe_loop_pipe(self):
        """Transcription loop for pipe-based audio"""
        try:
            with open(self.transcript_file, "a", encoding="utf-8") as f:
                logger.info("Transcription loop started (pipe mode), waiting for audio data...")
                chunks_received = 0

                while self.running:
                    # Read audio chunk from pipe
                    try:
                        chunk = self.pipe.read(self.chunk_size) if self.pipe else b''
                        if not chunk:
                            logger.debug("No audio data received from pipe")
                            break
                    except Exception as e:
                        logger.error(f"Error reading from pipe: {e}")
                        break

                    chunks_received += 1
                    if chunks_received == 1:
                        logger.info(f"First audio chunk received: {len(chunk)} bytes")
                    elif chunks_received % 100 == 0:
                        logger.debug(f"Received {chunks_received} chunks")

                    # Process with Vosk
                    if self.recognizer.AcceptWaveform(chunk):
                        # Final result available
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()

                        if text:
                            # Use wall clock time for timestamp
                            utc_time = datetime.now(tz=timezone.utc)
                            timestamp = utc_time.strftime("%Y-%m-%d-%H-%M-%S")

                            # Write to file
                            line = f"[{timestamp}] {text}\n"
                            f.write(line)
                            f.flush()

                            logger.info(f"Transcript: {line.strip()}")

                logger.info(f"Transcription loop ending. Total chunks: {chunks_received}")

                # Process any remaining audio
                final_result = json.loads(self.recognizer.FinalResult())
                text = final_result.get("text", "").strip()
                if text:
                    utc_time = datetime.now(tz=timezone.utc)
                    timestamp = utc_time.strftime("%Y-%m-%d-%H-%M-%S")
                    line = f"[{timestamp}] {text}\n"
                    f.write(line)
                    f.flush()
                    logger.info(f"Final transcript: {line.strip()}")

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Transcription loop ended (pipe mode)")
            if self.pipe:
                try:
                    self.pipe.close()
                except:
                    pass
