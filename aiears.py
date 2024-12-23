# HEARING-AID-VR - https://github.com/Vinventive/HEARING-AID-VR - 2024-12-23

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

# Ensure UTF-8 encoding for stdout
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
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)  # Keep at most 3

    def limit_hyphenated_repeats(token):
        pattern = re.compile(r'(\b\w+)(?:-\1){3,}-?')
        while True:
            match = pattern.search(token)
            if not match:
                break
            sub_pattern = match.group(1)
            replacement = '-'.join([sub_pattern] * 3) + '-'
            token = pattern.sub(replacement, token, count=1)
        return token

    tokens = text.split()
    tokens = [limit_hyphenated_repeats(token) for token in tokens]
    text = ' '.join(tokens)

    def limit_repeated_syllables(token, max_repeats=3):
        for sub_len in range(2, 7):  # Adjust the range as needed
            pattern = re.compile(r'(\b\w{' + str(sub_len) + r'})\1{' + str(max_repeats) + r',}')
            while True:
                match = pattern.search(token)
                if not match:
                    break
                repeated_pattern = match.group(1)
                replacement = repeated_pattern * max_repeats
                token = pattern.sub(replacement, token, count=1)
        return token

    tokens = text.split()
    tokens = [limit_repeated_syllables(token) for token in tokens]
    text = ' '.join(tokens)

    filtered_tokens = []
    last_token = None
    repeat_count = 0

    for t in tokens:
        if t == last_token:
            repeat_count += 1
        else:
            repeat_count = 1
            last_token = t
        if repeat_count <= 3:
            filtered_tokens.append(t)

    tokens = filtered_tokens

    def remove_repeated_patterns(token_list, pattern_len, max_repeats=3):
        i = 0
        while i + pattern_len <= len(token_list):
            pattern = token_list[i:i + pattern_len]
            repeat_count = 1
            j = i + pattern_len
            while j + pattern_len <= len(token_list) and token_list[j:j + pattern_len] == pattern:
                repeat_count += 1
                j += pattern_len
            if repeat_count > max_repeats:
                keep_upto = i + (pattern_len * max_repeats)
                token_list = token_list[:keep_upto] + token_list[j:]
            else:
                i = j
        return token_list

    for pl in [3, 2, 1]:
        tokens = remove_repeated_patterns(tokens, pl, max_repeats=3)

    def limit_repeated_phrases(text, max_repeats=3, max_phrase_len=5):
        for phrase_len in range(2, max_phrase_len + 1):
            pattern = re.compile(r'(\b(?:\w+\s+){' + str(phrase_len - 1) + r'}\w+)(?:\s+\1){' + str(max_repeats) + r',}')
            while True:
                match = pattern.search(text)
                if not match:
                    break
                repeated_phrase = match.group(1)
                # Replace the repeated phrases with max_repeats occurrences
                replacement = ' '.join([repeated_phrase] * max_repeats)
                text = pattern.sub(replacement, text, count=1)
        return text

    text = limit_repeated_phrases(' '.join(tokens), max_repeats=3, max_phrase_len=5)
    tokens = text.split()

    def limit_concatenated_repeats(token, max_repeats=3):
        for sub_len in range(2, 7):  # Adjust based on expected syllable lengths
            pattern = re.compile(r'(\b\w{' + str(sub_len) + r'})\1{' + str(max_repeats) + r',}')
            while True:
                match = pattern.search(token)
                if not match:
                    break
                repeated_pattern = match.group(1)
                replacement = repeated_pattern * max_repeats
                token = pattern.sub(replacement, token, count=1)
        return token

    tokens = text.split()
    tokens = [limit_concatenated_repeats(token) for token in tokens]
    text = ' '.join(tokens)

    def detect_spammy_symbol_distribution(text, min_length=20, top_n=3, min_freq_percent=10, max_freq_diff_percent=20):
        if len(text) < min_length:
            return False

        # Count the frequency of each symbol
        symbol_counts = {}
        for char in text:
            symbol_counts[char] = symbol_counts.get(char, 0) + 1

        # Sort symbols by frequency
        sorted_symbols = sorted(symbol_counts.items(), key=lambda item: item[1], reverse=True)

        if len(sorted_symbols) < top_n:
            return False

        top_symbols = sorted_symbols[:top_n]
        total_length = len(text)

        # Check if each top symbol meets the minimum frequency percentage
        for symbol, count in top_symbols:
            freq_percent = (count / total_length) * 100
            if freq_percent < min_freq_percent:
                return False

        # Check if the frequencies are within the allowed difference
        counts = [count for symbol, count in top_symbols]
        max_count = max(counts)
        min_count = min(counts)
        if (max_count - min_count) / max_count * 100 > max_freq_diff_percent:
            return False

        return True

    # Apply the spammy symbol distribution check
    if detect_spammy_symbol_distribution(text):
        logging.info("Spammy symbol distribution detected. Transcription discarded.")
        return "."

    cleaned_text = text.strip()
    return cleaned_text

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
            "Okay.", "Bye.", "I'm out.", "I'm going to go.", "I'm going to go",
            "I'm going to go to the next one.", "I'm going to go to the next video.",
            "I'm going to go to the top.", "I'm going to go to the top of the top.", 
            "a little bit of"
        ]

        if transcription in whisper_hallucinated_phrases:
            transcription = "."

        # Apply repetition-limiting logic
        transcription = limit_repetitions(transcription)

        return transcription
    except Exception as e:
        logging.error(f"Error in transcribe_with_whisper_sync: {e}")
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
    max_silence_duration = 0.03  # 30 ms
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
                audio_duration = len(audio_data) / (2 * 16000)  # bytes / (bytes_per_sample * sample_rate)
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
           -@@*-..:-.    .%%=+-:::..     +@=.     ...      =@+. .=++##@@@%: *@:  :@@###*+%%= *@=      
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
