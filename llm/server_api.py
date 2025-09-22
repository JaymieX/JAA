from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import os
import sys

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

#from asr import Asr, ModelSize
from llm import LLM, LLMProFile
#from tts_engine import TTSEngine


is_local = len(sys.argv) < 2 or sys.argv[1] != "server"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the Uvicorn worker starts
    print("--- Lifespan started... ---")
    
    # --- Profiles ---
    if is_local:
        print("--- Using local setup... ---")
        #asr_size    : ModelSize  = ModelSize.SMALL
        llm_profile : LLMProFile = LLMProFile.SMALL
    else:
        print("--- Using server setup... ---")
        #asr_size    : ModelSize  = ModelSize.LARGE_V3
        llm_profile : LLMProFile = LLMProFile.LARGE
    
    # -- Env --
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir / ".env"
    was_loaded = load_dotenv(dotenv_path=dotenv_path)
    
    print(f"Was .env file loaded? {was_loaded}")
    
    # --- Loading ASR ---
    #app.state.asr_model = Asr(asr_size)
    
    # --- Loading LLM ---
    app.state.llm = LLM(llm_profile, os.getenv('MY_NOTION_TOKEN'), os.getenv('MY_NOTION_PAGE_ID'))
    
    # --- Loading TTS ---
    # app.state.tts = TTSEngine()
    
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


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = Path(__file__).resolve().parent / filename
    
    if ".." in filename or "/" in filename:
        return {"error": "Invalid filename"}, 400
    
    if audio_path.is_file():
        # Create the response object first
        response = FileResponse(audio_path, media_type="audio/wav")

        # Add headers to prevent caching on the browser side.
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response
    
    return {"error": "File not found"}, 404


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
    final_response = llm.router(user_text)
    print(f"FINAL RESPONSE: {final_response}")
    print("=" * 50)  # Separator between queries

    
    return JSONResponse(content={
        "user_text": user_text,
        "bot_text": final_response,  # Show the final response after function calling
    })


"""
@app.post("/voice-chat/")
async def voice_chat_endpoint(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()

    asr_model : Asr       = request.app.state.asr_model
    llm       : LLM       = request.app.state.llm
    tts       : TTSEngine = request.app.state.tts

    user_text = asr_model.transcribe_audio(audio_bytes)
    print(f"USER QUERY: {user_text}")

    # If transcription is empty, don't bother with the LLM
    if not user_text.strip():
        print("No speech detected in audio")
        return JSONResponse(
            status_code=400,
            content={"error": "No speech detected in audio."}
        )
    
    # Route the LLM output through function calling logic
    final_response = llm.router(user_text)
    print(f"FINAL RESPONSE: {final_response}")
    print("=" * 50)  # Separator between queries

    # TTS
    output_filename = "response.wav"
    output_audio_path = tts.synthesize_speech(final_response, filename=output_filename)

    if output_audio_path:
        return JSONResponse(content={
            "user_text": user_text,
            "bot_text": final_response,  # Show the final response after function calling
            "audio_url": f"/audio/{output_filename}"  # Provide a URL to the audio
        })
        
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate audio response."}
        )
"""

if __name__ == "__main__":
    if is_local:
        uvicorn.run("server_api:app", host="127.0.0.1", port=8000, reload=True)
    else:
        uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=False)
    