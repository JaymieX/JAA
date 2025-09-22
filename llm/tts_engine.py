from pathlib import Path
import torch
from TTS.api import TTS

class TTSEngine:
    def __init__(self):
        self.tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        print("TTS Loaded")
        
        
    def synthesize_speech(self, text, filename="response.wav"):
        try:
            # Get the directory where this script lives
            script_dir = Path(__file__).resolve().parent
            file_path = script_dir / filename
            
            self.tts_engine.tts_to_file(
                text=text,
                file_path=str(file_path),
                speaker="Ana Florence",  # A good, standard female voice
                language="en"
            )

            return filename
        
        except Exception as e:
            # Print a more detailed error message
            print(f"Error during TTS synthesis: {e}")
            import traceback
            traceback.print_exc()
            return None