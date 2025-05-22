# live-captions-vr - https://github.com/Vinventive/live-captions-vr - 2025-05-22

import io
import logging
import warnings
import queue
import sys
import threading
import time
import re
import base64
import json
import asyncio
from websockets.asyncio.client import connect
import torch
import nemo.collections.asr as nemo_asr
import tempfile
import pyaudio
import soundfile as sf
from pydub import AudioSegment
import psutil
import socketserver
import webrtcvad
import requests
import os
import tkinter as tk
from tkinter import font as tkfont
import tkinter.ttk as ttk
from collections import deque

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False
    logging.warning("pystray or PIL not installed; system tray will not be available")
    logging.warning("To enable system tray, install with: pip install pystray pillow")

logging.basicConfig(filename='app-debug.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR) 
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
sys.stdout.reconfigure(encoding='utf-8')

try:
    parakeet_model_id = "nvidia/parakeet-tdt-0.6b-v2"
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=parakeet_model_id)
    if torch.cuda.is_available():
       asr_model = asr_model.to("cuda:0")
    else:
       asr_model = asr_model.to("cpu")
    logging.info(f"NVIDIA NeMo Parakeet TDT model ({parakeet_model_id}) loaded successfully.")
    logging.info("Ensure you have the NeMo toolkit installed: pip install nemo_toolkit['asr']")
except ImportError:
    logging.error("NVIDIA NeMo toolkit not found. Please install it: pip install nemo_toolkit['asr']")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to load the NVIDIA NeMo Parakeet TDT model: {e}")
    logging.error("This could be due to a missing NeMo installation or model download issues.")
    sys.exit(1)

audio_queue = queue.Queue()
stop_recording_event = threading.Event()

ENERGY_THRESHOLD = 0

parakeet_hallucinated_phrases = [
    "Yeah.", "Mm-hmm.", "Okay.", "Uh", "Mm", "Mm.",
]

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")


def limit_repetitions(text):
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
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
        for sub_len in range(2, 7):
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
    text = ' '.join(tokens)

    def limit_repeated_phrases(in_text, max_repeats=3, max_phrase_len=5):
        for phrase_len in range(2, max_phrase_len + 1):
            pattern = re.compile(
                r'(\b(?:\w+\s+){' + str(phrase_len - 1) + r'}\w+)(?:\s+\1){' + str(max_repeats) + r',}'
            )
            while True:
                match = pattern.search(in_text)
                if not match:
                    break
                repeated_phrase = match.group(1)
                replacement = ' '.join([repeated_phrase] * max_repeats)
                in_text = pattern.sub(replacement, in_text, count=1)
        return in_text
    text = limit_repeated_phrases(text, max_repeats=3, max_phrase_len=5)

    tokens = text.split()
    def limit_concatenated_repeats(token, max_repeats=3):
        for sub_len in range(2, 7):
            pattern = re.compile(r'(\b\w{' + str(sub_len) + r'})\1{' + str(max_repeats) + r',}')
            while True:
                match = pattern.search(token)
                if not match:
                    break
                repeated_pattern = match.group(1)
                replacement = repeated_pattern * max_repeats
                token = pattern.sub(replacement, token, count=1)
        return token
    tokens = [limit_concatenated_repeats(token) for token in tokens]
    text = ' '.join(tokens)

    def limit_long_substring_repeats(token, max_repeats=2):
        length = len(token)
        for sub_len in range(min(length // 2, 10), 1, -1):
            pattern = re.compile(r'(' + '.' * sub_len + r')(\1){2,}', flags=re.IGNORECASE)
            while True:
                match = pattern.search(token)
                if not match:
                    break
                repeated_pattern = match.group(1)
                replacement = repeated_pattern * max_repeats
                token = token[:match.start()] + replacement + token[match.end():]
        return token
    tokens = [limit_long_substring_repeats(t) for t in tokens]
    text = ' '.join(tokens)

    def detect_spammy_symbol_distribution(in_text, min_length=20, top_n=3, min_freq_percent=10, max_freq_diff_percent=20):
        if len(in_text) < min_length:
            return False
        symbol_counts = {}
        for char in in_text:
            symbol_counts[char] = symbol_counts.get(char, 0) + 1
        sorted_symbols = sorted(symbol_counts.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_symbols) < top_n:
            return False
        top_symbols = sorted_symbols[:top_n]
        total_length = len(in_text)
        for symbol, count in top_symbols:
            freq_percent = (count / total_length) * 100
            if freq_percent < min_freq_percent:
                return False
        counts = [count for symbol, count in top_symbols]
        max_count = max(counts)
        min_count = min(counts)
        if (max_count - min_count) / max_count * 100 > max_freq_diff_percent:
            return False
        return True

    if detect_spammy_symbol_distribution(text):
        logging.info("Spammy symbol distribution detected. Transcription discarded.")
        return "."

    cleaned_text = text.strip()
    return cleaned_text

def transcribe_with_parakeet(audio_data_np, sample_rate=16000):
    global asr_model
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            sf.write(tmp_audio_file.name, audio_data_np, sample_rate)
            temp_file_path = tmp_audio_file.name

        output = asr_model.transcribe([temp_file_path], verbose=False)

        if output and isinstance(output, list) and len(output) > 0 and hasattr(output[0], 'text'):
            transcription = output[0].text.strip()
            return transcription
        else:
            logging.warning("Parakeet transcription returned empty or unexpected result.")
            return ""
    except Exception as e:
        logging.error(f"Error in transcribe_with_parakeet: {e}")
        return "Error in transcription"
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_clean:
                logging.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")

def filter_transcription(transcription):
    if transcription in parakeet_hallucinated_phrases:
        logging.info(f"Transcription '{transcription}' found in hallucinated phrases, discarding.")
        return "."

    final = limit_repetitions(transcription)
    return final


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with clients_lock:
            clients.append(self.request)
        try:
            while True:
                data = self.request.recv(1024)
                if not data:
                    break
        finally:
            with clients_lock:
                if self.request in clients:
                    clients.remove(self.request)

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
    
    if caption_overlay:
        caption_overlay.add_message(message)


transcription_audio_queue = queue.Queue()
stop_transcription_event = threading.Event()

class TimedMessage:
    def __init__(self, text):
        self.text = text
        self.wrapped_text = []
        self.time_received = time.time()
        self.alpha = 1.0
        
    def update_alpha(self, fade_start_time, fade_duration):
        message_age = time.time() - self.time_received
        
        if message_age > fade_start_time:
            fade_progress = (message_age - fade_start_time) / fade_duration
            self.alpha = max(0.0, 1.0 - fade_progress)
        
        return self.alpha > 0

class SettingsWindow:
    def __init__(self, parent=None):
        self.parent = parent
        self.window = None
        self.settings = {
            "font_size": 18,
            "fade_start_time": 7.0,
            "fade_duration": 3.0,
            "max_messages": 10,
            "language": "english",
            "outline_thickness": 3
        }
        
        self.supported_languages = [
            "afrikaans", "arabic", "armenian", "azerbaijani", "belarusian", "bengali", "bosnian", "bulgarian", 
            "catalan", "chinese", "croatian", "czech", "danish", "dutch", "english", "estonian", "finnish", 
            "french", "galician", "german", "greek", "hebrew", "hindi", "hungarian", "icelandic", "indonesian", 
            "italian", "japanese", "kannada", "kazakh", "korean", "latvian", "lithuanian", "macedonian", 
            "malay", "marathi", "maori", "nepali", "norwegian", "persian", "polish", "portuguese", "romanian", 
            "russian", "serbian", "slovak", "slovenian", "spanish", "swahili", "swedish", "tagalog", "tamil", 
            "thai", "turkish", "ukrainian", "urdu", "vietnamese", "welsh"
        ]
        
    def show(self):
        if self.window is None:
            self.create_window()
        else: 
            self.window.deiconify()
            self.window.lift()
    
    def hide(self):
        if self.window:
            self.window.grab_release()
            self.window.withdraw()
            
            if self.parent and hasattr(self.parent, '_setup_drag_functionality'):
                self.window.after(100, self.parent._setup_drag_functionality)
    
    def create_window(self):
        self.window = tk.Toplevel()
        self.window.title("Live Captions VR Settings")
        self.window.geometry("400x450")
        self.window.resizable(False, False)
        
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            if os.path.exists(icon_path):
                icon_image = tk.PhotoImage(file=icon_path)
                self.window.iconphoto(True, icon_image)
                self.icon_image = icon_image
                logging.info(f"Applied custom icon to settings window from {icon_path}")
        except Exception as e:
            logging.error(f"Failed to set settings window icon: {e}")
        
        self.window.transient(self.parent.root if self.parent else None)
        self.window.grab_set()
        
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        settings_frame = ttk.LabelFrame(main_frame, text="Appearance")
        settings_frame.pack(fill=tk.X, pady=5)
        
        font_size_frame = ttk.Frame(settings_frame)
        font_size_frame.pack(fill=tk.X, pady=5)
        
        font_size_label = ttk.Label(font_size_frame, text="Font Size:")
        font_size_label.pack(side=tk.LEFT, padx=5)
        
        self.font_size_var = tk.IntVar(value=self.settings["font_size"])
        font_size_spinner = ttk.Spinbox(
            font_size_frame, 
            from_=8, 
            to=42, 
            width=5, 
            textvariable=self.font_size_var
        )
        font_size_spinner.pack(side=tk.LEFT, padx=5)
        
        outline_frame = ttk.Frame(settings_frame)
        outline_frame.pack(fill=tk.X, pady=5)
        
        outline_label = ttk.Label(outline_frame, text="Outline Thickness:")
        outline_label.pack(side=tk.LEFT, padx=5)
        
        self.outline_thickness_var = tk.IntVar(value=self.settings["outline_thickness"])
        outline_spinner = ttk.Spinbox(
            outline_frame, 
            from_=0, 
            to=5, 
            width=5, 
            textvariable=self.outline_thickness_var
        )
        outline_spinner.pack(side=tk.LEFT, padx=5)
        
        language_frame = ttk.LabelFrame(main_frame, text="Language")
        language_frame.pack(fill=tk.X, pady=5)
        
        lang_select_frame = ttk.Frame(language_frame)
        lang_select_frame.pack(fill=tk.X, pady=5)
        
        lang_label = ttk.Label(lang_select_frame, text="Transcription Language:")
        lang_label.pack(side=tk.LEFT, padx=5)
        
        display_languages = [lang.capitalize() for lang in self.supported_languages]
        
        self.language_var = tk.StringVar(value=self.settings["language"].capitalize())
        language_dropdown = ttk.Combobox(
            lang_select_frame,
            values=display_languages,
            textvariable=self.language_var,
            width=15,
            state="readonly"
        )
        language_dropdown.pack(side=tk.LEFT, padx=5)
        
        timing_frame = ttk.LabelFrame(main_frame, text="Timing")
        timing_frame.pack(fill=tk.X, pady=5)
        
        fade_start_frame = ttk.Frame(timing_frame)
        fade_start_frame.pack(fill=tk.X, pady=5)
        
        fade_start_label = ttk.Label(fade_start_frame, text="Fade Start (seconds):")
        fade_start_label.pack(side=tk.LEFT, padx=5)
        
        self.fade_start_var = tk.DoubleVar(value=self.settings["fade_start_time"])
        fade_start_spinner = ttk.Spinbox(
            fade_start_frame, 
            from_=1.0, 
            to=30.0, 
            increment=0.5,
            width=5, 
            textvariable=self.fade_start_var
        )
        fade_start_spinner.pack(side=tk.LEFT, padx=5)
        
        fade_duration_frame = ttk.Frame(timing_frame)
        fade_duration_frame.pack(fill=tk.X, pady=5)
        
        fade_duration_label = ttk.Label(fade_duration_frame, text="Fade Duration (seconds):")
        fade_duration_label.pack(side=tk.LEFT, padx=5)
        
        self.fade_duration_var = tk.DoubleVar(value=self.settings["fade_duration"])
        fade_duration_spinner = ttk.Spinbox(
            fade_duration_frame, 
            from_=0.5, 
            to=10.0, 
            increment=0.5,
            width=5, 
            textvariable=self.fade_duration_var
        )
        fade_duration_spinner.pack(side=tk.LEFT, padx=5)
        
        max_msg_frame = ttk.Frame(timing_frame)
        max_msg_frame.pack(fill=tk.X, pady=5)
        
        max_msg_label = ttk.Label(max_msg_frame, text="Maximum Messages:")
        max_msg_label.pack(side=tk.LEFT, padx=5)
        
        self.max_msg_var = tk.IntVar(value=self.settings["max_messages"])
        max_msg_spinner = ttk.Spinbox(
            max_msg_frame, 
            from_=1, 
            to=10, 
            width=5, 
            textvariable=self.max_msg_var
        )
        max_msg_spinner.pack(side=tk.LEFT, padx=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        apply_button = ttk.Button(button_frame, text="Apply", command=self.apply_settings)
        apply_button.pack(side=tk.RIGHT, padx=5)
        
        close_button = ttk.Button(button_frame, text="Close", command=self.hide)
        close_button.pack(side=tk.RIGHT, padx=5)
        
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"+{x}+{y}")
    
    def apply_settings(self):
        if not self.parent:
            return
            
        self.settings["font_size"] = self.font_size_var.get()
        self.settings["fade_start_time"] = self.fade_start_var.get()
        self.settings["fade_duration"] = self.fade_duration_var.get()
        self.settings["max_messages"] = self.max_msg_var.get()
        self.settings["language"] = self.language_var.get().lower()
        self.settings["outline_thickness"] = self.outline_thickness_var.get()
        
        self.parent.apply_settings(self.settings)
        
        self.window.bell()

class CaptionOverlay:
    def __init__(self):
        self.root = None
        self.messages = deque(maxlen=10)
        self.message_fade_start_time = 7.0
        self.message_fade_duration = 3.0
        self.language = "english"
        self.outline_thickness = 0
        
        self.message_queue = queue.Queue()
        
        self.stop_event = threading.Event()
        
        self.settings_window = SettingsWindow(self)
        
        self.tray_icon = None
        if HAS_TRAY:
            self._setup_tray_icon()
    
    def _setup_tray_icon(self):
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            
            if os.path.exists(icon_path):
                icon_image = Image.open(icon_path)
                logging.info(f"Loaded custom icon from {icon_path}")
            else:
                logging.info("Custom icon not found, using default")
                icon_size = 64
                icon_image = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(icon_image)
                draw.ellipse((4, 4, icon_size-4, icon_size-4), fill=(255, 255, 255))
            
            menu = (
                pystray.MenuItem('Settings', self._on_settings_clicked),
                pystray.MenuItem('Hide/Show Captions', self._on_toggle_visibility),
                pystray.MenuItem('Exit', self._on_exit_clicked)
            )
            
            self.tray_icon = pystray.Icon("LiveCaptions VR", icon_image, "Live Captions VR", menu)
            
            tray_thread = threading.Thread(target=self.tray_icon.run)
            tray_thread.daemon = True
            tray_thread.start()
        except Exception as e:
            logging.error(f"Failed to setup tray icon: {e}")
    
    def _on_settings_clicked(self, icon, item):
        if self.root:
            self.root.after(0, self.settings_window.show)
    
    def _on_toggle_visibility(self, icon, item):
        if not self.root:
            return
        
        try:
            if self.root.winfo_viewable():
                self.root.withdraw()
            else:
                self.root.deiconify()
                self.root.attributes("-topmost", True)
                
                if hasattr(self, 'drag_canvas') and self.drag_canvas and self.drag_canvas.winfo_exists():
                    self._setup_drag_functionality()
                else:
                    logging.info("Recreating UI after visibility toggle")
                    self._recreate_interface()
        except Exception as e:
            logging.error(f"Error toggling visibility: {e}")
    
    def _on_exit_clicked(self, icon, item):
        try:
            logging.info("Exit requested from system tray")
            
            self.stop_event.set()
            stop_recording_event.set()
            stop_transcription_event.set()
            
            if self.tray_icon:
                try:
                    self.tray_icon.stop()
                except Exception as e:
                    logging.error(f"Error stopping tray icon: {e}")
            
            if self.root:
                try:
                    self.root.destroy()
                except Exception as e:
                    logging.error(f"Error destroying root window: {e}")
            
            logging.info("Forcing application termination")
            
            if hasattr(os, 'kill'):
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
            
            os._exit(0)
        except Exception as e:
            logging.error(f"Error during exit: {e}")
            os._exit(1)

    def apply_settings(self, settings):
        if not self.root:
            return
            
        self.message_fade_start_time = settings["fade_start_time"]
        self.message_fade_duration = settings["fade_duration"]
        self.messages = deque(maxlen=settings["max_messages"])
        self.language = settings["language"]
        self.outline_thickness = settings["outline_thickness"]
        
        self._update_display()
        
        self._setup_drag_functionality()
    
    def _create_window(self):
        self.root = tk.Tk()
        self.root.title("Live Captions VR")
        
        self.root.attributes("-alpha", 1.0)
        self.root.attributes("-topmost", True)
        
        self.root.overrideredirect(True)
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.25) + 10
        x_position = (screen_width - window_width) // 2
        y_position = int(screen_height * 0.7)
        
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
   
        if sys.platform == 'win32':
            self.transparent_color = "#010101"
            self.root.configure(bg=self.transparent_color)
            self.root.wm_attributes("-transparentcolor", self.transparent_color)
        else:
            self.transparent_color = "black"
            self.root.configure(bg=self.transparent_color)
            self.root.attributes("-transparent", True)
        
        self.drag_container = tk.Frame(
            self.root,
            height=10,
            bg=self.transparent_color,
            highlightthickness=0,
            bd=0
        )
        self.drag_container.pack(side=tk.BOTTOM, fill=tk.X)
        
        drag_bar_width = int(window_width * 0.33)
        max_height = 5
        
        drag_bar_x = (window_width - drag_bar_width) // 2
        
        self.drag_canvas = tk.Canvas(
            self.drag_container,
            width=window_width,
            height=max_height + 4,
            bg=self.transparent_color,
            highlightthickness=0,
            bd=0
        )
        self.drag_canvas.pack(fill=tk.X)
        

        center_y = (max_height + 4) / 2
        
        self.drag_handle_segments = []
        
        num_segments = 10
        segment_width = drag_bar_width / (num_segments * 2)
        
        for i in range(num_segments):
            segment_height = 1 + (i * (max_height - 1) / num_segments)
            
            segment_y = center_y - (segment_height / 2)
            
            left_x = drag_bar_x + (i * segment_width)
            left_segment = self.drag_canvas.create_rectangle(
                left_x, segment_y, 
                left_x + segment_width, segment_y + segment_height,
                fill="white", outline="white"
            )
            self.drag_handle_segments.append(left_segment)
            
            right_x = drag_bar_x + drag_bar_width - ((i + 1) * segment_width)
            right_segment = self.drag_canvas.create_rectangle(
                right_x, segment_y, 
                right_x + segment_width, segment_y + segment_height,
                fill="white", outline="white"
            )
            self.drag_handle_segments.append(right_segment)
        
        center_x = drag_bar_x + (drag_bar_width / 2) - segment_width
        center_segment = self.drag_canvas.create_rectangle(
            center_x, center_y - (max_height / 2),
            center_x + (segment_width * 2), center_y + (max_height / 2),
            fill="white", outline="white"
        )
        self.drag_handle_segments.append(center_segment)
        
        self.main_frame = tk.Frame(
            self.root, 
            bg=self.transparent_color
        )
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.caption_canvas = tk.Canvas(
            self.main_frame,
            bg=self.transparent_color,
            highlightthickness=0,
            bd=0
        )
        self.caption_canvas.pack(fill=tk.BOTH, expand=True)
        
        self._setup_drag_functionality()
        

        if sys.platform == 'win32':
            try:
                import win32gui
                import win32api
                import win32con
                
                self._win32gui = win32gui
                self._win32con = win32con
                self._win32api = win32api
                
                hwnd = self.root.winfo_id()
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                style = style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
                
                self.root.after(100, lambda: self._make_widget_clickable(self.drag_canvas))
            except ImportError:
                logging.warning("pywin32 not installed, click-through will not work")
    
    def _start_move(self, event):
        self.x = event.x
        self.y = event.y
        
    def _stop_move(self, event):
        self.x = None
        self.y = None
    
    def _do_move(self, event):
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            return
            
        deltax = event.x - self.x
        deltay = event.y - self.y
        
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        
        self.root.geometry(f"+{x}+{y}")
    
    def _make_widget_clickable(self, widget):
        if sys.platform == 'win32' and hasattr(self, '_win32gui') and widget.winfo_ismapped():
            try:
                hwnd = widget.winfo_id()
                style = self._win32gui.GetWindowLong(hwnd, self._win32con.GWL_EXSTYLE)
                style = style & ~self._win32con.WS_EX_TRANSPARENT
                self._win32gui.SetWindowLong(hwnd, self._win32con.GWL_EXSTYLE, style)
                
                self._win32gui.SetWindowPos(
                    hwnd, 0, 0, 0, 0, 0,
                    self._win32con.SWP_NOMOVE | self._win32con.SWP_NOSIZE | 
                    self._win32con.SWP_NOZORDER | self._win32con.SWP_FRAMECHANGED
                )
                
                widget.update_idletasks()
            except Exception as e:
                logging.error(f"Error making widget clickable: {e}")
        
        if self.root and not self.stop_event.is_set():
            self.root.after(250, lambda: self._make_widget_clickable(widget))
    
    def _setup_drag_functionality(self):
        if not hasattr(self, 'drag_canvas') or not self.drag_canvas or not self.drag_canvas.winfo_exists():
            logging.warning("Drag canvas not available for setup")
            return False
        
        try:
            self.drag_canvas.unbind("<ButtonPress-1>")
            self.drag_canvas.unbind("<ButtonRelease-1>")
            self.drag_canvas.unbind("<B1-Motion>")
            
            self.drag_canvas.bind("<ButtonPress-1>", self._start_move)
            self.drag_canvas.bind("<ButtonRelease-1>", self._stop_move)
            self.drag_canvas.bind("<B1-Motion>", self._do_move)
            
            if sys.platform == 'win32' and hasattr(self, '_win32gui'):
                self._make_widget_clickable(self.drag_canvas)
            
            return True
        except Exception as e:
            logging.error(f"Error setting up drag functionality: {e}")
            return False
    
    def add_message(self, message):
        if message and message.strip():
            self.message_queue.put(message)
    
    def _update_display(self):
        if not self.root or not hasattr(self, 'caption_canvas'):
            return
            
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self.messages.append(TimedMessage(message))
                self.message_queue.task_done()
        except queue.Empty:
            pass
        
        try:
            canvas_width = self.caption_canvas.winfo_width()
            canvas_height = self.caption_canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = int(self.root.winfo_screenwidth() * 0.7)
            if canvas_height <= 1:
                canvas_height = int(self.root.winfo_screenheight() * 0.25)
            
            self.caption_canvas.delete("all")
            
            visible_messages = []
            
            for msg in list(self.messages):
                visible = msg.update_alpha(self.message_fade_start_time, self.message_fade_duration)
                if visible:
                    visible_messages.append(msg)
                else:
                    self.messages.remove(msg)
            
            font_size = self.settings_window.settings["font_size"]
            font_obj = tkfont.Font(family="Arial", size=font_size, weight="bold")
            
            effective_width = min(canvas_width * 0.8, canvas_width - (font_size * 4))
            
            for msg in visible_messages:
                text = msg.text
                wrapped_lines = []
                
                words = text.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if font_obj.measure(test_line) > effective_width:
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                
                if current_line:
                    wrapped_lines.append(current_line)
                
                msg.wrapped_text = wrapped_lines
            
            message_heights = []
            for msg in visible_messages:
                line_height = font_obj.metrics("linespace")
                msg_height = line_height * len(msg.wrapped_text) + 10
                message_heights.append(msg_height)
            
            y_positions = []
            remaining_height = canvas_height - 10
            for height in reversed(message_heights):
                remaining_height -= height
                y_positions.insert(0, remaining_height + 5)
            
            for i, msg in enumerate(visible_messages):
                y_pos = y_positions[i] if i < len(y_positions) else 10
                
                color_value = int(255 * msg.alpha)
                text_color = f"#{color_value:02x}{color_value:02x}{color_value:02x}"
                
                x_pos = canvas_width / 2
                
                line_y = y_pos
                line_height = font_obj.metrics("linespace")
                
                if self.outline_thickness > 0:
                    thickness = self.outline_thickness
                    
                    for line_idx, line in enumerate(msg.wrapped_text):
                        outline_directions = [
                            (-thickness, 0),
                            (thickness, 0),
                            (0, -thickness),
                            (0, thickness),
                            (-thickness, -thickness),
                            (thickness, -thickness),
                            (-thickness, thickness),
                            (thickness, thickness)
                        ]
                        
                        for x_offset, y_offset in outline_directions:
                            self.caption_canvas.create_text(
                                x_pos + x_offset,
                                line_y + (line_idx * line_height) + y_offset,
                                text=line,
                                fill="black",
                                font=font_obj,
                                anchor="n",
                                justify="center"
                            )
                
                for line_idx, line in enumerate(msg.wrapped_text):
                    self.caption_canvas.create_text(
                        x_pos,
                        line_y + (line_idx * line_height),
                        text=line,
                        fill=text_color,
                        font=font_obj,
                        anchor="n",
                        justify="center"
                    )
        except Exception as e:
            logging.error(f"Error in _update_display: {e}")
        
        if self.root and self.root.winfo_exists():
            self.root.after(50, self._update_display)
    
    def _recreate_interface(self):
        try:
            if hasattr(self, 'main_frame') and self.main_frame:
                try:
                    self.main_frame.destroy()
                except:
                    pass
            
            if hasattr(self, 'drag_container') and self.drag_container:
                try:
                    self.drag_container.destroy()
                except:
                    pass
            
            self._create_window()
        except Exception as e:
            logging.error(f"Error recreating interface: {e}")
    
    def start(self):
        self.overlay_thread = threading.Thread(target=self._run_overlay)
        self.overlay_thread.daemon = True
        self.overlay_thread.start()
        
    def _run_overlay(self):
        try:
            self._create_window()
            
            self._update_display()
            
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Error in caption overlay: {e}")
        finally:
            if self.tray_icon:
                try:
                    self.tray_icon.stop()
                except:
                    pass

caption_overlay = None

def continuous_audio_recording():
    vad = webrtcvad.Vad(2)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320)

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
                    silence_duration += 0.02
                    if silence_duration > max_silence_duration:
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
            if (
                (combined_duration >= target_min_duration and time_since_last_voice >= silence_timeout)
                or (combined_duration >= target_max_duration)
            ):
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
                            
                            current_language = "english"
                            if caption_overlay:
                                current_language = caption_overlay.language
                                
                            large_transcription = transcribe_with_parakeet(
                                audio_np, 
                                sample_rate=sample_rate
                            )

                            if large_transcription and large_transcription != "Error in transcription":
                                final_text = filter_transcription(large_transcription)
                                if final_text != ".":
                                    send_message_to_clients(final_text)
                            else:
                                logging.info("Large transcription was empty or error; discarding.")
                        else:
                            logging.info("Audio RMS below threshold, discarding.")
                    else:
                        logging.info("Audio too short after processing, discarding.")
                else:
                    pass

        except Exception as e:
            logging.error(f"Error in transcription_worker: {e}")


def main():
    global caption_overlay
    
    caption_overlay = CaptionOverlay()
    caption_overlay.start()
    
    start_tcp_server()

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
    print(stay_comfy)

    print("<<< This is the core component of the Live Captions VR Overlay >>>")
    print("<<< Please don't close this window while VR Overlay is running >>>")
    print("\nLive captions are now running. Check the system tray icon for settings.")

    recording_thread = threading.Thread(target=continuous_audio_recording, daemon=True)
    recording_thread.start()

    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    transcription_thread.start()
    
    def shutdown_application():
        logging.info("Application shutdown initiated")
        stop_recording_event.set()
        stop_transcription_event.set()
        
        if caption_overlay:
            caption_overlay.stop_event.set()
            if caption_overlay.tray_icon:
                try:
                    caption_overlay.tray_icon.stop()
                except:
                    pass
            
            if caption_overlay.root:
                try:
                    caption_overlay.root.destroy()
                except:
                    pass
        
        logging.info("Application shutdown complete")
        os._exit(0)

    try:
        while True:
            time.sleep(1)
            if caption_overlay and caption_overlay.stop_event.is_set():
                shutdown_application()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received...")
        shutdown_application()


if __name__ == "__main__":
    main()
