# F5 TTS WebUI
a simplied webui for f5-tts-mlx, which is dedicated for MacOS devices with Apple Silicon like macbook pro, macmini, etc.
# Launch
``` 
conda create -n f5-tts-webui python=3.11
conda activate f5-tts-webui
cd f5-tts-webui
pip install -r requirements.txt 
python app.py
```
You may need to wait for a while, it will download the model and the tokenizer.
# UI
![UI](./tts-webui.png)
# Thanks
Thanks f5-tts-mlx, f5-tts for the great work.