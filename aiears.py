# Standard library imports
import io
import logging
import os
import queue
import sys
import threading
import time

# Third-party imports - Core
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

# Third-party imports - Audio processing
import pyaudio
import soundfile as sf
from pydub import AudioSegment

# Third-party imports - System
import psutil
import socketserver

# Third-party imports - VAD
import webrtcvad

# Initialize logging for debugging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

sys.stdout.reconfigure(encoding='utf-8')

# Load the Whisper large-v3-turbo model using Transformers
device_str = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

try:
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.to(device_str)
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    sys.exit(1)  # Exit if Whisper model cannot be loaded

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device_str,
)

# Updated: audio_queue now holds transcription text
audio_queue = queue.Queue()

stop_recording_event = threading.Event()

# Noise gate to reduce whisper hallucinations - adjust this value based on your current environment background noise level 

ENERGY_THRESHOLD = 0  # Updated energy threshold (example value)

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

def transcribe_with_whisper_sync(audio_data):
    try:
        # Use the pipeline to transcribe the audio data
        result = pipe(audio_data, generate_kwargs={"language": "english"})
        transcription = result["text"].strip()

        # Handle hallucinated phrases
        whisper_hallucinated_phrases = [
            "Goodbye.", "Thanks for watching!", "Thank you for watching!",
            "I feel like I'm going to die.", "Thank you for watching.",
            "Transcription by CastingWords", "Thank you."
        ]
        if transcription in whisper_hallucinated_phrases:
            transcription = "."

        return transcription
    except Exception as e:
        logging.error(f"Error in transcribe_with_whisper: {e}")
        return "Error in transcription"

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Add the client to the list of clients
        with clients_lock:
            clients.append(self.request)
        try:
            while True:
                # Keep the connection open
                data = self.request.recv(1024)
                if not data:
                    break  # Client disconnected
        finally:
            # Remove the client from the list when they disconnect
            with clients_lock:
                if self.request in clients:
                    clients.remove(self.request)

# Global variables for TCP server
clients = []
clients_lock = threading.Lock()

def start_tcp_server(host='127.0.0.1', port=65432):
    server = socketserver.ThreadingTCPServer((host, port), ThreadedTCPRequestHandler)
    server.daemon_threads = True  # Ensure that server thread exits when main thread does
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Ensure that server thread exits when main thread does
    server_thread.start()
    logging.info(f"TCP server started on {host}:{port}")

def send_message_to_clients(message):
    with clients_lock:
        for client in clients[:]:  # Make a copy to avoid modification during iteration
            try:
                client.sendall((message + "\n").encode('utf-8'))
            except Exception as e:
                logging.error(f"Error sending message to client: {e}")
                clients.remove(client)

# New queues and events for continuous recording and transcription
transcription_audio_queue = queue.Queue()
stop_transcription_event = threading.Event()

def continuous_audio_recording():
    vad = webrtcvad.Vad(3)  # Voice Activity Detection with aggressiveness mode 3
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=320)  # 20 ms frames
    speech_buffer = []
    silence_duration = 0
    max_silence_duration = 0.5  # seconds
    is_speaking = False
    logging.info("Starting continuous audio recording...")
    while not stop_recording_event.is_set():
        try:
            data = stream.read(320, exception_on_overflow=False)
            # Use VAD to detect speech
            is_speech = vad.is_speech(data, sample_rate=16000)
            if is_speech:
                speech_buffer.append(data)
                silence_duration = 0
                if not is_speaking:
                    is_speaking = True
            else:
                if is_speaking:
                    silence_duration += 0.02  # 20 ms frames
                    if silence_duration > max_silence_duration:
                        # User stopped speaking
                        is_speaking = False
                        # Process the speech buffer
                        if speech_buffer:
                            audio_data = b''.join(speech_buffer)
                            speech_buffer = []
                            # Send audio data to transcription queue
                            transcription_audio_queue.put(audio_data)
        except Exception as e:
            logging.error(f"Error in continuous_audio_recording: {e}")
            break
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcription_worker():
    combined_audio_data = []
    combined_duration = 1.5  # in seconds
    target_min_duration = 0.3  # Minimum duration to consider processing
    target_max_duration = 2.0  # Maximum duration to prevent long waits
    min_transcribe_duration = 0.2  # Minimum duration after silence removal
    silence_timeout = 0  # Increased to 2.0 seconds
    last_voice_activity_time = time.time()

    while not stop_transcription_event.is_set():
        try:
            audio_data_received = False
            try:
                # Try to get new audio data from the queue
                audio_data = transcription_audio_queue.get(timeout=0.1)
                # Append audio data to combined_audio_data
                combined_audio_data.append(audio_data)
                # Calculate duration of the audio data
                audio_duration = len(audio_data) / (2 * 16000)  # 2 bytes per sample, 16000 samples per second
                combined_duration += audio_duration
                last_voice_activity_time = time.time()
                audio_data_received = True
            except queue.Empty:
                # No new audio data
                pass

            # Check if we have silence for longer than silence_timeout
            time_since_last_voice = time.time() - last_voice_activity_time
            if (combined_duration >= target_min_duration and time_since_last_voice >= silence_timeout) or combined_duration >= target_max_duration:
                if combined_audio_data:
                    # Combine audio data
                    all_audio_data = b''.join(combined_audio_data)
                    combined_audio_data = []
                    combined_duration = 0.0

                    # Convert audio_data to WAV format
                    audio_segment = AudioSegment(
                        data=all_audio_data,
                        sample_width=2,  # pyaudio.paInt16 is 2 bytes
                        frame_rate=16000,
                        channels=1
                    )

                    # Check duration after potential processing
                    if len(audio_segment) >= min_transcribe_duration * 1000:  # milliseconds
                        # Compute RMS energy
                        rms = audio_segment.rms
                        if rms >= ENERGY_THRESHOLD:
                            # Export to BytesIO
                            wav_io = io.BytesIO()
                            audio_segment.export(wav_io, format="wav")
                            wav_io.seek(0)
                            # Read audio data into NumPy array
                            audio_np, sample_rate = sf.read(wav_io)
                            # Transcribe with Whisper
                            transcription = transcribe_with_whisper_sync(audio_np)
                            if transcription and transcription != ".":
                                # Send transcription over TCP
                                send_message_to_clients(transcription)
                        else:
                            # Audio RMS below threshold, discard
                            logging.info("Audio RMS below threshold, discarding.")
                    else:
                        # Audio too short after potential processing, discard
                        logging.info("Audio too short after processing, discarding.")
                else:
                    # No audio data accumulated
                    pass

        except Exception as e:
            logging.error(f"Error in transcription_worker: {e}")

def main():
    # Initialize and start TCP server
    start_tcp_server()

    # Start continuous audio recording thread
    recording_thread = threading.Thread(target=continuous_audio_recording, daemon=True)
    recording_thread.start()

    # Start transcription worker thread
    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    transcription_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        stop_recording_event.set()
        stop_transcription_event.set()
        sys.exit(0)

if __name__ == "__main__":
    main()
