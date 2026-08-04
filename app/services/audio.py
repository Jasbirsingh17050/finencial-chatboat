import os
from faster_whisper import WhisperModel
import edge_tts

# Initialize STT model globally so it doesn't reload on every request
print("Loading Whisper STT model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(file_path: str) -> str:
    """Transcribes an audio file to text using faster-whisper."""
    segments, _ = whisper_model.transcribe(file_path, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text

async def synthesize_speech(text: str, output_path: str):
    """Converts text to speech and saves it to a file."""
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(output_path)