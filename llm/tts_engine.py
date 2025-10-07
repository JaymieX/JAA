import os
from pathlib import Path
from elevenlabs.client import ElevenLabs

class TTSEngine:
    def __init__(self):
        key = os.environ["ELEVEN_API_KEY"]
        self.client = ElevenLabs(api_key=key)
        print("TTS Loaded")
        
        
    def synthesize_speech(self, text, filename="response.mp3"):
        audio_generator  = self.client.text_to_speech.convert(
            voice_id="XrExE9yKIg1WjnnlVkGX", # Informative voice
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings={
                "stability":        0.3,
                "similarity_boost": 0.75,
                "style":            0.9
            }
        )

        # Ensure filepath is in the same directory as this script
        script_dir = Path(__file__).resolve().parent
        filepath = script_dir / filename

        with open(filepath, "wb") as f:
            for chunk in audio_generator:
                if chunk:  # Safety check
                    f.write(chunk)

        return str(filepath)
