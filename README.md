# Production Blueprint: Local Voice AI Agent Architecture
## System Specification: LiveKit Core + Whistle STT + Kokoro TTS
**Target Audience:** Senior Infrastructure Engineers / Open-Source Core Maintainers  
**Status:** Architecture Approved  
**Last Revised:** October 2026  

This document serves as the technical blueprint for implementing a modular, highly performant Voice AI Agent using the LiveKit infrastructure framework. The core system bypasses traditional, high-latency cloud components by integrating an ultra-lightweight on-device Speech-to-Text model (**Whistle STT**) and a high-efficiency local Text-to-Speech framework (**Kokoro TTS**), coordinated via standard WebRTC pipelines.

---

## 1. System Topology & Data Flow Matrix

The architecture splits operational workloads between WebRTC real-time transport, local disk inference, and stateless microservices to preserve thread resources.

```
       +---------------------------------------------+
       |             LiveKit Cloud / Server          |
       +----------------------+----------------------+
                              |
                     WebRTC (Audio Tracks)
                              |
                              v
       +----------------------+----------------------+
       |          LiveKit Python Worker Node         |
       |                                             |
       |  +---------------------------------------+  |
       |  | Custom WhistleSTT Adapter (Non-Stream)|  |
       |  | -> Allocates AudioBuffer Array        |  |
       |  | -> Dispatches to ThreadPoolExecutor   |  |
       |  | -> Invokes Native C++ Needle Engine   |  |
       |  +-------------------+-------------------+  |
       |                      |                      |
       |            Inferred Transcript              |
       |                      |                      |
       |                      v                      |
       |  +-------------------+-------------------+  |
       |  | LLM Orchestration Layer (GPT-4o-Mini) |  |
       |  +-------------------+-------------------+  |
       |                      |                      |
       |               Streaming Text                |
       |                      |                      |
       |                      v                      |
       |  +-------------------+-------------------+  |
       |  | LiveKit OpenAI TTS Adapter Client     |  |
       |  +-------------------+-------------------+  |
       +----------------------+----------------------+
                              |
                     HTTP POST /v1/audio/speech
                              |
                              v
       +----------------------+----------------------+
       |        Kokoro-FastAPI Docker Container      |
       |    (StyleTTS 2 Engine / Port: 8880)         |
       +---------------------------------------------+
```

| Component | Subsystem Tech | Latency Target | Deployment Vector |
| :--- | :--- | :--- | :--- |
| **Ingress/Egress Gateway** | LiveKit WebRTC Protocol | $<10\text{ ms}$ Network Jitter | LiveKit Cloud or Distributed Self-Hosted Server |
| **Speech-to-Text (STT)** | Whistle (`cactus-needle`) | $11.1\text{ ms}$ TTFT (10s audio) | On-Node Native C++ Binding Execution via ThreadPool |
| **Brain / Context Layer** | OpenAI/Custom LLM Protocol | Variable ($~200{-}500\text{ ms}$) | Streaming API Token Gateway Engine |
| **Text-to-Speech (TTS)** | Kokoro TTS (`StyleTTS 2`) | Real-Time GPU/CPU generation | Local Stateless Microservice (`ghcr.io` Container) |

---

## 2. Infrastructure Setup & Environment Matrix

Enforce strict isolation principles across your build stack. Use `uv` for fast dependency verification.

### Dependencies Specifications
Initialize an isolated shell ecosystem with exact constraint matching:

```bash
# Initialize isolated python virtual ecosystem
uv venv ecosystem_env --python 3.11
source ecosystem_env/bin/activate

# Inject precise protocol dependency bindings
uv add "livekit-agents[openai]>=1.8.0" cactus-needle python-dotenv
```

### Host Environment Parameters (`.env.local`)
Create a strict environment allocation layer. Ensure no actual keys are passed to local-only engines.

```env
LIVEKIT_URL=wss://your-allocated-project.livekit.cloud
LIVEKIT_API_KEY=dev-api-key-hash-string
LIVEKIT_API_SECRET=dev-api-secret-secure-string
OPENAI_API_KEY=sk-proj-production-key-token
```

---

## 3. Modular Engine Implementations

### A. Custom Audio Transformer Adapter (`whistle_stt.py`)
Because Whistle STT executes synchronously via the native Cactus C++ runtime, its lifecycle must be isolated from the asynchronous event loop using Python’s `ThreadPoolExecutor`.

```python
import io
import wave
import asyncio
import logging
from livekit import agents

# Explicitly ensure native C++ engine is available inside target binary path
try:
    import needle
except ImportError as e:
    raise ImportError("CRITICAL: 'cactus-needle' library execution layer failed to initialize.") from e

logger = logging.getLogger("subsystem.stt.whistle")

class WhistleSTT(agents.stt.STT):
    def __init__(self):
        # Configure capability layer to handle complete conversational frames natively
        super().__init__(capabilities=agents.stt.STTCapabilities(streaming=False))

    async def _transcribe_impl(
        self, 
        buffer: agents.stt.AudioBuffer, 
        language: str | None = None
    ) -> agents.stt.SpeechEvent:
        """
        Intercepts incoming room track AudioBuffer arrays, marshals them into 
        uncompressed 16kHz linear PCM bytes, and triggers local model calculation.
        """
        try:
            io_buffer = io.BytesIO()
            with wave.open(io_buffer, "wb") as wav_writer:
                wav_writer.setnchannels(buffer.num_channels)
                wav_writer.setsampwidth(2)  # Fixed 16-bit encoding width
                wav_writer.setframerate(buffer.sample_rate)
                wav_writer.writeframes(buffer.data)
            
            io_buffer.seek(0)
            
            # Offload heavy synchronous model inference tasks to a background worker pool
            loop = asyncio.get_running_loop()
            inference_payload = await loop.run_in_executor(
                None, 
                lambda: needle.transcribe(io_buffer)
            )
            
            # Normalize strings and strip metadata out of band
            normalized_text = inference_payload.get("text", "").strip()
            detected_locale = inference_payload.get("audio_language", "en")
            
            return agents.stt.SpeechEvent(
                type=agents.stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    agents.stt.SpeechAlternative(
                        text=normalized_text,
                        language=detected_locale
                    )
                ]
            )
        except Exception as error_context:
            logger.error(f"Inference fault inside Whistle STT engine pipeline: {error_context}")
            return agents.stt.SpeechEvent(type=agents.stt.SpeechEventType.END_OF_SPEECH, alternatives=[])
```

### B. Monolithic Orchestrator Node (`agent.py`)
This main application handles WebRTC channel subscriptions, hooks the voice state machines together, and safely deploys the LiveKit worker.

```python
import asyncio
import logging
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai

# Inject localized dependencies
from whistle_stt import WhistleSTT

# Setup production logging standard outputs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("core.agent.orchestrator")

load_dotenv(dotenv_path=".env.local")

async def entrypoint(ctx: JobContext):
    """
    Main worker lifecycle loop. Triggers immediately upon allocation 
    by the upstream LiveKit room orchestrator.
    """
    logger.info(f"Establishing worker channel inside room allocation: {ctx.room.name}")
    
    # Restrict inbound track data to audio only to scale server efficiency
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize asynchronous functional components
    stt_subsystem = WhistleSTT()
    llm_subsystem = openai.LLM(model="gpt-4o-mini")
    
    # Initialize Kokoro adapter using OpenAI layout structures over local network transport
    tts_subsystem = openai.TTS(
        model="kokoro",
        voice="af_alloy",
        api_key="not-applicable",
        base_url="http://localhost:8880/v1",
        response_format="wav"
    )

    # Build Voice Assistant State Engine
    assistant = VoiceAssistant(
        vad=ctx.proc.get_vad(), # Silero VAD configured at worker level natively
        stt=stt_subsystem,
        llm=llm_subsystem,
        tts=tts_subsystem,
    )

    # Bind pipeline execution context directly into active WebRTC tracks
    assistant.start(ctx.room)
    
    # Enforce initial greetings with user interruption parameters permitted
    await assistant.say(
        "System online. Local Whistle STT engine and local Kokoro audio pipeline ready.", 
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

---

## 4. Run-Time Orchestration & Provisioning Steps

Follow these exact steps in sequence to boot up the system architecture cleanly:

1. **Deploy Microservice Containers:**
   Execute the Docker task detached to release the shell console. This initializes the 82M parameter StyleTTS 2 network model.
   ```bash
   docker run -d \
     --name kokoro-tts-service \
     -p 8880:8880 \
     --restart unless-stopped \
     ghcr.io/remsky/kokoro-fastapi:latest
   ```

2. **Verify Port Health Check Bindings:**
   Ensure the local OpenAI emulation routes respond cleanly before starting the main process.
   ```bash
   curl http://localhost:8880/v1/models
   ```

3. **Launch the LiveKit Worker Daemon:**
   Start the agent script using the integrated CLI runtime in developer watch mode.
   ```bash
   python agent.py dev
   ```

---

## 5. Failure Modes, Limits, & Verification Metrics

* **Audio Buffer Scaling Constraints:** Whistle native ingestion scales linearly with audio depth lengths. Ensure maximum user conversational clip frames are capped at $30\text{ seconds}$ inside the VAD config to mitigate thread lockup.
* **OpenAI Protocol Compliance:** Ensure your Kokoro wrapper handles voice string configuration mapping cleanly. If specific voice packs fail to load via `af_alloy`, verify image model files directly against the container volume mounts.
* **Concurrency Limitations:** When running on CPU architectures, multiple concurrent users can cause processing delays. If scaling for production environments, migrate the container image layers to an NVIDIA GPU host with standard CUDA container toolkits mapped.
