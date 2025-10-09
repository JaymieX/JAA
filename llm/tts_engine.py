import os
import asyncio
from pathlib import Path
from elevenlabs.client import ElevenLabs

class TTSEngine:
    def __init__(self, use_elevenlabs=True):
        self.use_elevenlabs = use_elevenlabs

        if self.use_elevenlabs:
            key = os.environ["ELEVEN_API_KEY"]
            self.client = ElevenLabs(api_key=key)
            print("TTS Loaded (ElevenLabs)")
        else:
            import edge_tts
            self.edge_tts = edge_tts
            self.voice = "en-US-AriaNeural"  # Default Edge TTS voice
            print("TTS Loaded (Edge TTS)")


    def synthesize_speech(self, text, filename="response.mp3"):
        # Ensure filepath is in the same directory as this script
        script_dir = Path(__file__).resolve().parent
        filepath = script_dir / filename

        if self.use_elevenlabs:
            audio_generator = self.client.text_to_speech.convert(
                voice_id="XrExE9yKIg1WjnnlVkGX",  # Informative voice
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings={
                    "stability":        0.3,
                    "similarity_boost": 0.75,
                    "style":            0.9
                }
            )

            with open(filepath, "wb") as f:
                for chunk in audio_generator:
                    if chunk:  # Safety check
                        f.write(chunk)
        else:
            # Edge TTS implementation
            asyncio.run(self._edge_tts_generate(text, filepath))

        return str(filepath)

    async def _edge_tts_generate(self, text, filepath):
        """Helper method for Edge TTS async generation"""
        communicate = self.edge_tts.Communicate(text, self.voice)
        await communicate.save(str(filepath))
