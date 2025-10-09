import platform
import os

# vLLM setup
if platform.system() != 'Windows':
    
    # CRITICAL: Set vLLM to use spawn for multiprocessing BEFORE any imports
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    import multiprocessing
    # Also set Python's default multiprocessing method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import sys

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from asr import Asr, ModelSize
from llm import LLM
from llm_profiles import LLMProFile
from tts_engine import TTSEngine


# Parse command line arguments
is_server = "--server" in sys.argv
use_elevenlabs = "--11labs" in sys.argv
is_local = not is_server

# Parse ASR size argument (--asr=small|medium|large-v2|large-v3)
asr_size = ModelSize.SMALL  # Default
for arg in sys.argv:
    if arg.startswith("--asr="):
        asr_value = arg.split("=")[1].upper().replace("-", "_")
        try:
            asr_size = ModelSize[asr_value]
        except KeyError:
            print(f"Warning: Invalid ASR size '{arg.split('=')[1]}'. Using default: small")

# Parse LLM profile argument (--llm=small|large|super_large)
llm_profile = LLMProFile.SMALL  # Default
for arg in sys.argv:
    if arg.startswith("--llm="):
        llm_value = arg.split("=")[1].upper()
        try:
            llm_profile = LLMProFile[llm_value]
        except KeyError:
            print(f"Warning: Invalid LLM profile '{arg.split('=')[1]}'. Using default: small")

# Validate: local mode can only use SMALL LLM
if is_local and llm_profile != LLMProFile.SMALL:
    print("ERROR: Local mode (without --server) can only use --llm=small")
    print("Your machine won't be able to handle larger models.")
    print("Either use --llm=small or add --server flag for server deployment.")
    sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the Uvicorn worker starts
    print("--- Lifespan started... ---")
    print(f"--- Mode: {'Server' if is_server else 'Local'} ---")
    print(f"--- ASR Model: {asr_size.value} ---")
    print(f"--- LLM Profile: {llm_profile.name} ---")
    print(f"--- TTS Engine: {'ElevenLabs' if use_elevenlabs else 'Edge TTS'} ---")
    
    # -- Env --
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir / ".env"
    was_loaded = load_dotenv(dotenv_path=dotenv_path)
    
    print(f"Was .env file loaded? {was_loaded}")
    
    # --- Loading LLM FIRST (vLLM must initialize CUDA before other libraries) ---
    app.state.llm = LLM(llm_profile, os.getenv('MY_NOTION_TOKEN'), os.getenv('MY_NOTION_PAGE_ID'))

    # --- Loading ASR ---
    app.state.asr_model = Asr(asr_size)
    
    # --- Loading TTS ---
    app.state.tts = TTSEngine(use_elevenlabs=use_elevenlabs)
    
    yield

    # Code below yield runs on shutdown (optional)
    print("--- Lifespan endded... ---")
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page for the voice assistant UI."""
    html_path = Path(__file__).resolve().parent / "front_end/index.html"
    
    if html_path.is_file():
        return html_path.read_text(encoding="utf-8-sig")
    else:
        return "<h1>Error: index.html not found</h1><p>Please make sure the index.html file is in the same directory as your Python script.</p>", 404


@app.get("/audio/{filename}", response_class=PlainTextResponse)
async def get_audio(filename: str):
    audio_path = Path(__file__).resolve().parent / filename
    
    if audio_path.is_file():
        return audio_path
    
    return ""


@app.post("/chat/")
async def chat_endpoint(request: Request, user_text: str = Form(...)):
    llm: LLM = request.app.state.llm

    print(f"USER QUERY: {user_text}")

    # If text is empty, don't bother with the LLM
    if not user_text.strip():
        print("No text recived.")
        return JSONResponse(
            status_code=400,
            content={"error": "No text recived."}
        )
    
    # Route the LLM output through function calling logic
    final_response = llm.generate_response(user_text)
    print(f"FINAL RESPONSE: {final_response}")
    print("=" * 50)  # Separator between queries

    
    return JSONResponse(content={
        "user_text": user_text,
        "bot_text": final_response,  # Show the final response after function calling
    })


@app.post("/asr/")
async def asr_endpoint(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()

    asr_model : Asr       = request.app.state.asr_model

    user_text = asr_model.transcribe_audio(audio_bytes)
    print(f"USER QUERY: {user_text}")

    # If transcription is empty, don't bother with the LLM
    if not user_text.strip():
        print("No speech detected in audio")
        return JSONResponse(
            status_code=400,
            content={"error": "No speech detected in audio."}
        )
        
    return JSONResponse(content={
        "user_text": user_text
    })


@app.post("/tts/")
async def tts_endpoint(request: Request, user_text: str = Form(...)):
    print(f"USER QUERY: {user_text}")
    
    tts : TTSEngine = request.app.state.tts

    # If transcription is empty, don't bother with the LLM
    if not user_text.strip():
        print("No speech detected in audio")
        return JSONResponse(
            status_code=400,
            content={"error": "No speech detected in audio."}
        )

    # TTS
    output_filename = "response.mp3"
    output_audio_path = tts.synthesize_speech(user_text, filename=output_filename)

    if output_audio_path:
        return JSONResponse(content={
            "user_text": user_text,
            "audio_url": f"/audio/{output_filename}"  # Provide a URL to the audio
        })
        
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate audio response."}
        )


if __name__ == "__main__":
    if is_local:
        uvicorn.run("server_api:app", host="127.0.0.1", port=8000, reload=False)
    else:
        uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=False)
    