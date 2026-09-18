import os
import io
import wave
import uuid
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from detector import VoiceCloneDetector
from demo_samples import DEMO_SCENARIOS, AUDIO_DIR, ensure_demo_audio_files
import database

app = FastAPI(
    title="Voice Guard AI Engine",
    description="Real-time Voice Clone and Deepfake Audio Detection Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = VoiceCloneDetector(sample_rate=16000)

# Ensure sample files exist on startup
ensure_demo_audio_files()

class ActionRequest(BaseModel):
    call_id: str
    action: str  # "CALL_TERMINATED_SAFELY", "CALLBACK_VERIFICATION_TRIGGERED", "SECURITY_CHALLENGE_ISSUED", "VERIFIED_SAFE"

class ChunkAnalysisRequest(BaseModel):
    audio_base64: str
    caller_tag: Optional[str] = "Live Call"
    caller_number: Optional[str] = "+91 Unknown"
    sample_rate: Optional[int] = 16000

@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "engine": "Voice Guard ASVspoof Anomaly Detector",
        "sample_rate": 16000,
        "active_models": ["Neural Vocoder Phase Analyzer", "MFCC Cepstral Variance", "Biological Jitter Profiler"]
    }

@app.get("/api/scenarios")
def list_scenarios():
    return DEMO_SCENARIOS

@app.get("/api/scenarios/{scenario_id}/audio")
def get_scenario_audio(scenario_id: str):
    matched = next((s for s in DEMO_SCENARIOS if s["id"] == scenario_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="Scenario not found")
    file_path = os.path.join(AUDIO_DIR, matched["filename"])
    if not os.path.exists(file_path):
        ensure_demo_audio_files()
    return FileResponse(file_path, media_type="audio/wav")

@app.post("/api/scenarios/{scenario_id}/analyze")
def analyze_scenario(scenario_id: str):
    matched = next((s for s in DEMO_SCENARIOS if s["id"] == scenario_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    file_path = os.path.join(AUDIO_DIR, matched["filename"])
    if not os.path.exists(file_path):
        ensure_demo_audio_files()

    with wave.open(file_path, "r") as wf:
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        sr = wf.getframerate()
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

    result = detector.analyze(audio_float, caller_tag=matched["caller_name"])
    event_id = f"call-{uuid.uuid4().hex[:8]}"

    database.log_call_event(
        event_id=event_id,
        caller_name=matched["caller_name"],
        caller_number=matched["caller_number"],
        synthetic_probability=result["synthetic_probability"],
        human_probability=result["human_probability"],
        risk_level=result["risk_level"],
        action_taken="SIMULATED_SCREENING",
        cues_detected=result["detected_cues"],
        metrics=result["metrics"],
        latency_ms=result["latency_ms"]
    )

    return {
        "event_id": event_id,
        "scenario": matched,
        "analysis": result
    }

@app.post("/api/analyze-chunk")
def analyze_audio_chunk(payload: ChunkAnalysisRequest):
    try:
        raw_bytes = base64.b64decode(payload.audio_base64)
        # Check if bytes is raw PCM 16-bit or Float32
        if len(raw_bytes) % 2 == 0:
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_data = audio_int16.astype(np.float32) / 32768.0
        else:
            audio_data = np.frombuffer(raw_bytes, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {str(e)}")

    result = detector.analyze(audio_data, caller_tag=payload.caller_tag)
    
    # If high risk or critical, log event
    event_id = f"live-{uuid.uuid4().hex[:8]}"
    if result["risk_level"] in ["CRITICAL_CLONE", "SUSPICIOUS"]:
        database.log_call_event(
            event_id=event_id,
            caller_name=payload.caller_tag,
            caller_number=payload.caller_number,
            synthetic_probability=result["synthetic_probability"],
            human_probability=result["human_probability"],
            risk_level=result["risk_level"],
            action_taken="FLAGGED_ALERT",
            cues_detected=result["detected_cues"],
            metrics=result["metrics"],
            latency_ms=result["latency_ms"]
        )

    return {
        "event_id": event_id,
        "analysis": result
    }

@app.post("/api/analyze-file")
async def analyze_uploaded_file(file: UploadFile = File(...)):
    content = await file.read()
    
    # Try reading as WAV
    try:
        bio = io.BytesIO(content)
        with wave.open(bio, 'rb') as wf:
            n_frames = wf.getnframes()
            sr = wf.getframerate()
            raw_data = wf.readframes(n_frames)
            audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
    except Exception:
        # Fallback: interpret as 16-bit PCM raw samples
        audio_int16 = np.frombuffer(content[:len(content) - (len(content)%2)], dtype=np.int16)
        if len(audio_int16) == 0:
            raise HTTPException(status_code=400, detail="Could not decode audio file format. Please upload standard WAV audio.")
        audio_float = audio_int16.astype(np.float32) / 32768.0

    result = detector.analyze(audio_float, caller_tag=file.filename)
    event_id = f"file-{uuid.uuid4().hex[:8]}"

    database.log_call_event(
        event_id=event_id,
        caller_name=f"File: {file.filename}",
        caller_number="Uploaded Recording",
        synthetic_probability=result["synthetic_probability"],
        human_probability=result["human_probability"],
        risk_level=result["risk_level"],
        action_taken="FILE_INSPECTED",
        cues_detected=result["detected_cues"],
        metrics=result["metrics"],
        latency_ms=result["latency_ms"]
    )

    return {
        "event_id": event_id,
        "filename": file.filename,
        "analysis": result
    }

@app.get("/api/logs")
def get_call_logs():
    return database.get_all_logs(limit=50)

@app.post("/api/take-action")
def take_action(payload: ActionRequest):
    database.update_call_action(payload.call_id, payload.action)
    return {
        "status": "SUCCESS",
        "call_id": payload.call_id,
        "action": payload.action,
        "message": f"Security action '{payload.action}' executed and logged."
    }

@app.get("/api/stats")
def get_threat_stats():
    return database.get_stats()

@app.websocket("/ws/live-stream")
async def websocket_audio_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Can receive either binary audio chunks or json
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # 16-bit PCM or Float32 from browser
                try:
                    audio_float = np.frombuffer(raw_bytes, dtype=np.float32)
                except Exception:
                    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0

                result = detector.analyze(audio_float, caller_tag="Live Stream")
                await websocket.send_json(result)

            elif "text" in message and message["text"]:
                import json
                data = json.loads(message["text"])
                if "audio_base64" in data:
                    raw_bytes = base64.b64decode(data["audio_base64"])
                    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    result = detector.analyze(audio_float, caller_tag=data.get("caller_tag", "Live Stream"))
                    await websocket.send_json(result)
                elif data.get("type") == "PING":
                    await websocket.send_json({"type": "PONG"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
