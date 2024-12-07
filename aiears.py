# HEARING-AID-VR - https://github.com/Vinventive/HEARING-AID-VR - 2024-12-07

# Standard library imports
import io
import logging
import warnings
import queue
import sys
import threading
import time
import re

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
    filename='app-debug.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Suppress future warnings and transformer logs to keep the console clean
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.stdout.reconfigure(encoding='utf-8')

# Load the Whisper large-v3-turbo model using Transformers
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

try:
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.to(device)
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
    device=device,
)

audio_queue = queue.Queue()

stop_recording_event = threading.Event()

# Noise gate threshold
ENERGY_THRESHOLD = 0

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

def limit_repetitions(text):
    text = re.sub(r'(.)\1{3,}', r'\1\1\1\1', text)

    words = text.split()
    limited_words = []
    last_word = None
    repeat_count = 0
    max_word_repetition = 3

    for w in words:
        if w == last_word:
            repeat_count += 1
        else:
            repeat_count = 1
            last_word = w
        if repeat_count <= max_word_repetition:
            limited_words.append(w)

    text = ' '.join(limited_words)

    parts = re.split(r'([.?!])', text)

    sentences = []
    for i in range(0, len(parts), 2):
        sentence_text = parts[i].strip()
        if not sentence_text:
            continue
        punctuation = parts[i+1].strip() if i+1 < len(parts) else ''
        full_sentence = (sentence_text + punctuation).strip()
        if full_sentence:
            sentences.append(full_sentence)

    # Limiting repeated identical sentences
    limited_sentences = []
    last_sentence = None
    for s in sentences:
        if s == last_sentence:
            continue
        limited_sentences.append(s)
        last_sentence = s

    cleaned_text = ' '.join(limited_sentences).strip()

    if len(limited_sentences) <= 1:
        words_all = cleaned_text.split()
        length = len(words_all)
        if length > 3:
            # Attempt to find a repeating word pattern
            for pattern_len in range(1, (length // 2) + 1):
                pattern = words_all[:pattern_len]
                repeat_count = 1
                idx = pattern_len
                while idx < length:
                    chunk = words_all[idx:idx+pattern_len]
                    if chunk == pattern:
                        repeat_count += 1
                        idx += pattern_len
                    else:
                        break
                if repeat_count > 2:
                    # Keep only 2 repeats and then remainder
                    remainder_start = pattern_len * repeat_count
                    remainder = words_all[remainder_start:]
                    new_words = pattern * 2 + remainder
                    cleaned_text = " ".join(new_words).strip()
                    break


    if len(cleaned_text) > 20:
        s = cleaned_text
        ss = (s+s)[1:-1]
        idx = ss.find(s)
        if idx != -1:
            unit = s[:idx+1]

            unit_len = len(unit)
            count = 1
            start_idx = unit_len
            while start_idx + unit_len <= len(s):
                chunk = s[start_idx:start_idx+unit_len]
                if chunk == unit:
                    count += 1
                    start_idx += unit_len
                else:
                    break
            if count > 2:
                remainder = s[start_idx:]
                cleaned_text = unit * 2 + remainder

    return cleaned_text.strip()

def transcribe_with_whisper_sync(audio_data):
    try:
        # Use the pipeline to transcribe the audio data
        result = pipe(audio_data, generate_kwargs={"language": "english", "return_timestamps": True})
        transcription = result["text"].strip()

        # Handle known hallucinated phrases
        whisper_hallucinated_phrases = [
            "Goodbye.", "Thanks for watching!", "Thank you for watching!",
            "I feel like I'm going to die.", "Thank you for watching.",
            "Transcription by CastingWords", "Thank you.", "I'm sorry.",
            "Okay.", "Bye."
        ]

        if transcription in whisper_hallucinated_phrases:
            transcription = "."

        # Apply repetition-limiting logic
        transcription = limit_repetitions(transcription)

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
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    logging.info(f"TCP server started on {host}:{port}")

def send_message_to_clients(message):
    with clients_lock:
        for client in clients[:]:
            try:
                client.sendall((message + "\n").encode('utf-8'))
            except Exception as e:
                logging.error(f"Error sending message to client: {e}")
                clients.remove(client)

# Queues and events for continuous recording and transcription
transcription_audio_queue = queue.Queue()
stop_transcription_event = threading.Event()

def continuous_audio_recording():
    vad = webrtcvad.Vad(3)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=320)
    speech_buffer = []
    silence_duration = 0
    max_silence_duration = 0.03
    is_speaking = False
    logging.info("Starting continuous audio recording...")
    while not stop_recording_event.is_set():
        try:
            data = stream.read(320, exception_on_overflow=False)
            is_speech = vad.is_speech(data, sample_rate=16000)
            if is_speech:
                speech_buffer.append(data)
                silence_duration = 0
                if not is_speaking:
                    is_speaking = True
            else:
                if is_speaking:
                    silence_duration += 0.02  # 20 ms increments
                    if silence_duration > max_silence_duration:
                        # User stopped speaking
                        is_speaking = False
                        if speech_buffer:
                            audio_data = b''.join(speech_buffer)
                            speech_buffer = []
                            transcription_audio_queue.put(audio_data)
        except Exception as e:
            logging.error(f"Error in continuous_audio_recording: {e}")
            break
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcription_worker():
    combined_audio_data = []
    combined_duration = 1.5
    target_min_duration = 0.3
    target_max_duration = 2.0
    min_transcribe_duration = 0.3
    silence_timeout = 0
    last_voice_activity_time = time.time()

    while not stop_transcription_event.is_set():
        try:
            try:
                audio_data = transcription_audio_queue.get(timeout=0.1)
                combined_audio_data.append(audio_data)
                audio_duration = len(audio_data) / (2 * 16000)
                combined_duration += audio_duration
                last_voice_activity_time = time.time()
            except queue.Empty:
                pass

            time_since_last_voice = time.time() - last_voice_activity_time
            if (combined_duration >= target_min_duration and time_since_last_voice >= silence_timeout) or combined_duration >= target_max_duration:
                if combined_audio_data:
                    all_audio_data = b''.join(combined_audio_data)
                    combined_audio_data = []
                    combined_duration = 0.0

                    audio_segment = AudioSegment(
                        data=all_audio_data,
                        sample_width=2,
                        frame_rate=16000,
                        channels=1
                    )

                    if len(audio_segment) >= min_transcribe_duration * 1000:
                        rms = audio_segment.rms
                        if rms >= ENERGY_THRESHOLD:
                            wav_io = io.BytesIO()
                            audio_segment.export(wav_io, format="wav")
                            wav_io.seek(0)
                            audio_np, sample_rate = sf.read(wav_io)
                            # Transcribe with Whisper
                            transcription = transcribe_with_whisper_sync(audio_np)
                            if transcription and transcription != ".":
                                send_message_to_clients(transcription)
                        else:
                            logging.info("Audio RMS below threshold, discarding.")
                    else:
                        logging.info("Audio too short after processing, discarding.")
                else:
                    pass
        except Exception as e:
            logging.error(f"Error in transcription_worker: {e}")

def main():
    # Initialize and start TCP server
    start_tcp_server()

    # ASCII art
    stay_comfy = """
    
                                         ...-====+***+++====-...   .-=++++=-..                      
                     .:---:....    ...-=+#%%@@%%%%##*###%%%@@%%#+=*%%%*++*####*-.                   
                   :+%%%#%%%%%%*--=#%%%#++=-                 -++%@@=..    ...-+%%*:                 
                 .*@#-:...::+=+@@@#*=:..                       .#@=            .-*%*:              
                :%%=.          :%@:                            .%%.               -#%=.            
               -@*.             -@=                             +*.                .=@*:           
             .+@+.              .:.   ..-==+*##****##*++==..    ..                   -%%=          
            .#%-                   :=+**                 *##+-.                       :%@+         
           -%%:                 .=*#*=:.                  .:=*%#-.                     -@@=        
          =@*.                -+#*-.   ..                    .:+%%=.                    =@%.       
        .+@+.               -*#=.     =%.                       .=##+.                  .#@+.      
       .*@*.              :##=.      -@+                          .-*#-.                 -@%.      
       -@%.             .+%+.       .%#.                             -##:                .%@-      
      .*@+             :##:         =@=                               .=#+.    ==.        *@=      
      .#@=       .   .=%+.         .#@:                    :+:          .*#-.  *@*       .%@-      
       +@*     .#*. :#%-           :@#.                    :@=            -%#: .%@*:.  .:+@%.      
       .%@=:..-#%- -%*.            +@=                     :@#.            :*@+..+%@%**%@@#-       
        :*@@#%@*..*%=              *@:                     .*%.              -%#: .=***%@+.        
         .-*@@+ :%#:              .*@=                     .#+      :=-.      .*@*.    =@#         
           *@#:+#=              .+%@@=                     :@+      .#@=        :#%=.  .#@-        
          =@@##+.        ..   .+#+=@@=                     =@@=.     .*%:        .-##=. *@=        
     .:::=%@%+:         :%=.:*%*: .@@=                    .#%+%#-     .%#.         .-%#-=@#.       
     +@%**+-.           *@=*#+:   .-@#.                   :@+ .+%*-.   -@+.          .+%*@@:       
     :*#=.             :@@*-..:=+: :@%.                   +@:   .=*#+=:.#@.            .=%@*.      
       -*#-:.         .#@-:+#@@@@= .*@-                  -@* -***+--+*%%@@=          .:-+#@%.      
       .*@@*.         =@#%@@@@%%*:  :@*.                .#@- =@@@@@%*-::=@*.  .==:.. -%@+=%@-      
       -@@*-..:-.    .%%=+-:::..     +@=       ...      =@+. .=++##@@@%: *@:  :@@###*+%%= *@=      
       =@*=**%@#:    :@-             .*@=.     .#%:   .=@#.      ..::::. -@=  :@%.:::::.  *@+      
      .%@- ..=@+    .*@.              .#@*      =@%=. .@%.               :@=  :@%.        +@#.     
      :@@.   .@+    :@#.               .#@+.     *@%#+=@%.               :@+  :@*.        :@@-     
      :@%.   .#+    :@+                 .=%#=.   +@.:*##*.               :@+  :@+         .@@=     
      +@+     *#.   :@+      .....        .*%#=:.:%.  ...  .::--::..     .@=  :@+         .#@+     
      *@+     -@-   :@= .:=+*#%%%#*-.       .=#%*+@=     -*@@@@@@@@#*=:. =@: .+@=          +@+     
     .#@=     .#%:  :@*+%@@@@@@@@@@@%:.       .:--=.   .+@@@@@@@@@@@@@@#+@#. .@%.          +@#.    
     .@@:      .%#. :@@@@@@@@%#*****#*.                .**======**#@@@@@@@=  -@-           +@%.    
     .@@:       -@#..%@@%*+=-.      ..                            .:-=*@@#.  *#.          .#@*.    
     .@@:        :*%+++                                                *@-  -@=           :@@+     
     .@@=         +@*                   .#%%%@@@@@@@@@#.              -@=  :%#.           +@@:     
     .%@+        .@@                    .%@:::::::::@@+             .+%=:-*@#.           .#@*      
     .+@*.       .@@.                   .*@.       :@@:           .=%@**#%@=.            -@@-      
      .@@.       .%@.                    :@+       +@#.           :##*--+#:             .#@#.      
       =@*.       =@=                    .#@-     .%@:      ...    .  -#+.              -@@:       
        +@+.      .+@=.                   .*@.....*@=   .-*#%@#*+=:.-*#-               .%@*        
         +@+.       =%#=:.       .:=++++:   -*###%#-  :*%@#++++%@@@#@=                .#@#.        
          =@#:.      :=#@#*+-:.=*%@@@@@@@#=          -@@%-.    ..-*@@*:              :#@%:         
           -%@#-       .-+#@@@@@@%#*+==-:+%@+.       -@@+.          :+@@+.          .+@@#:          
            .+@@#-.       -%@%*=:.       :%@#+++++++%@-              :%@%-       .=%@%=.           
              .=%@%+:.  :*@%-.            :%@@@####@@+                :#@@*.    .%@#+.             
                .-*%%*=*@@#.               -@@%...-@#.                 .+@@#.   .#@=               
                   .=%@@%=.                .#@@.  *@:                    -%@#.   .*@=              
                   -%@@%:                   +@@- .@#.                     -%@#.   .#@:             
                  +@@@#:                    :@@+ -@+                       -%@#:   :@%.            
                 +@@@+.                     .%@+ +@+                        .%@#.   +@+            
                -@@@=                       .*@* +@+                         +@@*.  .@@.           
               .%@@*                         =@#.-@+                         :%@@:   *@=           
               =@@+.                         :@%..%#.                         +@@+   -@#.           
"""  

    # Print the ASCII art at the start of the app
    print(stay_comfy)
    print("<<< This is the core component of the Hearing AID VR Overlay >>>\n<<< Please don't close this window while VR Overlay is running >>>\n\nLet this comfy lil fella support you in the background")

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
