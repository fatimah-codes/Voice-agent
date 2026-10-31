import asyncio
import base64
import io
import os
import wave
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    TTSModelSettings,
    VoicePipeline,
    VoicePipelineConfig,
)

# Page configuration - Minimalist & Full Viewport
st.set_page_config(
    page_title="MindAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide all default Streamlit chrome to maximize centered hero immersion
st.markdown("""
<style>
  #MainMenu, header, footer, .stDeployButton, div[data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
  }
  
  html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #F6EAF2 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
  }

  .main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100vw !important;
  }

  iframe {
    border: none !important;
    width: 100vw !important;
    height: 100vh !important;
    display: block !important;
  }
</style>
""", unsafe_allow_html=True)

SAMPLE_RATE = 24_000

# System key detection & auto-authentication
system_key = os.getenv("OPENAI_API_KEY", "").strip()
has_system_key = bool(system_key)

if "authenticated_key" not in st.session_state:
    st.session_state.authenticated_key = system_key if has_system_key else None
if "last_action_time" not in st.session_state:
    st.session_state.last_action_time = 0
if "component_args" not in st.session_state:
    st.session_state.component_args = {
        "status": "idle",
        "is_authenticated": bool(st.session_state.authenticated_key),
        "has_system_key": has_system_key,
    }

# Sync authentication state
is_auth = bool(st.session_state.authenticated_key)
st.session_state.component_args["is_authenticated"] = is_auth
st.session_state.component_args["has_system_key"] = has_system_key

@st.cache_resource
def get_voice_pipeline(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    agent = Agent(
        name="Travel Advisor",
        instructions=(
            "You are MindAI's intelligent, calm voice advisor. "
            "Keep spoken answers short, natural, warm, and easy to follow. "
            "Respond in 1 to 2 sentences maximum."
        ),
        model="gpt-4o-mini",
    )
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        config=VoicePipelineConfig(
            tts_settings=TTSModelSettings(voice="coral"),
            trace_include_sensitive_audio_data=False,
        ),
    )
    return pipeline

async def run_pipeline(audio_arr: np.ndarray, api_key: str):
    pipeline = get_voice_pipeline(api_key)
    audio_input = AudioInput(buffer=audio_arr, frame_rate=SAMPLE_RATE)
    result = await pipeline.run(audio_input)

    audio_chunks = []
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            audio_chunks.append(event.data)

    pcm_data = b"".join(audio_chunks)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    wav_bytes = buf.getvalue()
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")
    data_uri = f"data:audio/wav;base64,{b64_audio}"
    return result.total_output_text, data_uri

# Declare custom component
_component_path = os.path.join(os.path.dirname(__file__), "frontend")
mind_voice_agent = components.declare_component("mind_voice_agent", path=_component_path)

# Render component
component_val = mind_voice_agent(**st.session_state.component_args, key="mind_voice_agent_instance")

# Handle frontend messages
if component_val and isinstance(component_val, dict):
    action = component_val.get("action")
    timestamp = component_val.get("timestamp", 0)

    if timestamp > st.session_state.last_action_time:
        st.session_state.last_action_time = timestamp

        # Handle API Key Submission
        if action == "set_api_key":
            submitted_key = component_val.get("api_key", "").strip()
            if submitted_key:
                st.session_state.authenticated_key = submitted_key
                st.session_state.component_args["is_authenticated"] = True
                st.rerun()

        # Handle Use System Key
        elif action == "use_system_key":
            if has_system_key:
                st.session_state.authenticated_key = system_key
                st.session_state.component_args["is_authenticated"] = True
                st.rerun()

        # Handle Voice Query
        elif action == "query":
            current_key = st.session_state.authenticated_key or system_key
            if not current_key:
                st.session_state.component_args = {
                    "status": "error",
                    "error_message": "Please enter an OpenAI API key first.",
                    "is_authenticated": False,
                    "has_system_key": has_system_key,
                    "timestamp": timestamp,
                }
                st.rerun()
            else:
                st.session_state.authenticated_key = current_key
                audio_b64 = component_val.get("audio_b64", "")
                if audio_b64:
                    try:
                        raw_wav = base64.b64decode(audio_b64)
                        with wave.open(io.BytesIO(raw_wav), "rb") as wf:
                            n_channels = wf.getnchannels()
                            framerate = wf.getframerate()
                            frames = wf.readframes(wf.getnframes())
                            audio_arr = np.frombuffer(frames, dtype=np.int16)

                        # Convert stereo to mono if needed
                        if n_channels > 1:
                            audio_arr = audio_arr.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

                        # Resample to 24000 if browser hardware was 48000/44100
                        if framerate != SAMPLE_RATE and len(audio_arr) > 0:
                            target_len = int(len(audio_arr) * SAMPLE_RATE / framerate)
                            if target_len > 0:
                                indices = np.linspace(0, len(audio_arr) - 1, target_len)
                                audio_arr = np.interp(indices, np.arange(len(audio_arr)), audio_arr).astype(np.int16)

                        # Ensure non-empty audio
                        if len(audio_arr) == 0:
                            raise ValueError("No audio data received.")

                        text_resp, audio_uri = asyncio.run(
                            run_pipeline(audio_arr, current_key)
                        )

                        st.session_state.component_args = {
                            "status": "speaking",
                            "transcript": text_resp,
                            "audio_url": audio_uri,
                            "is_authenticated": True,
                            "has_system_key": has_system_key,
                            "timestamp": timestamp,
                        }
                        st.rerun()
                    except Exception as e:
                        st.session_state.component_args = {
                            "status": "error",
                            "error_message": f"{str(e)}",
                            "is_authenticated": True,
                            "has_system_key": has_system_key,
                            "timestamp": timestamp,
                        }
                        st.rerun()
