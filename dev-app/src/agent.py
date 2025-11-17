import ssl
ssl._create_default_https_context = ssl._create_unverified_context



# ⬇️ Вставляешь patch сразу сюда
import aiohttp
_old_init = aiohttp.ClientSession.__init__
def _new_init(self, *args, **kwargs):
    kwargs["connector"] = aiohttp.TCPConnector(ssl=False)
    return _old_init(self, *args, **kwargs)
aiohttp.ClientSession.__init__ = _new_init

import logging
import os
from datetime import datetime
from livekit.plugins import openai
from dotenv import load_dotenv
# from livekit.plugins import groq

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# -----------------------------------------------------
# STT LOGGING
# -----------------------------------------------------
def setup_stt_logging(stt_model: str):
    log_dir = "stt_logs"
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_dir}/{stt_model.replace('/', '_')}_{ts}.log"

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(fmt)

    logger_obj = logging.getLogger(f"stt_{stt_model}")
    logger_obj.setLevel(logging.INFO)
    logger_obj.addHandler(handler)

    return logger_obj


# -----------------------------------------------------
# STT FACTORY
# -----------------------------------------------------
def stt_factory(model_name: str):
    if model_name.startswith("assemblyai"):
        return inference.STT(model=model_name)

    if model_name.startswith("openai/whisper"):
        return inference.STT(model=model_name)

    if model_name.startswith("silero"):
        return silero.STT(model=model_name)
    
    # if model_name.startswith("groq"):
    #     return groq.STT(model=model_name)

    raise ValueError(f"Unknown STT model: {model_name}")


# -----------------------------------------------------
# AGENT (ONLY ONE CLASS!)
# -----------------------------------------------------
class Assistant(Agent):
    def __init__(self, stt_model: str):
        super().__init__(
            instructions="""You are a helpful voice AI assistant..."""
        )
        self.stt_logger = setup_stt_logging(stt_model)

    async def on_transcription(self, event):
        text = event.alternatives[0].text
        self.stt_logger.info(text)


# -----------------------------------------------------
# PREWARM VAD
# -----------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -----------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------
async def entrypoint(ctx: JobContext):

    ctx.log_context_fields = {"room": ctx.room.name}

    target_stt = os.getenv("TARGET_STT", "openai/whisper-tiny.en")

    session = AgentSession(
        # stt=stt_factory(target_stt),
        stt = openai.WhisperSTT(model="openai/whisper-tiny.en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    # start session ONCE
    await session.start(
        agent=Assistant(stt_model=target_stt),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# -----------------------------------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
