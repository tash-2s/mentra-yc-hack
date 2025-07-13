# Privacy Proxy and Sample DEMO App

Requirements: ffmpeg, OPENAI_API_KEY

```
uv run main.py --blur --capture --transcript --fps 14 --width 768 --height 1024 --box-blur --detection-scale 1.0 --detection-interval 7 --no-motion-detection --blur-kernel 51 --transcript-model vosk-model-small-en-us-0.15
# ngrok tcp 1935
```
