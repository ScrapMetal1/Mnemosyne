import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
from pynput import keyboard as kb
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import speech_recognition as sr
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ==== CONFIG ====
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

SAMPLE_RATE = 16000
CHANNELS = 1
FILENAME = "temp_recording.wav"

print("Push-to-Talk Assistant Ready ")
print("Hold 'P' to talk... release to stop.")
print("Press 'Q' to quit.")

# ==== AUDIO STREAM SETUP ====
audio_queue = queue.Queue()
current_stream = None

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def start_recording():
    """Begin capturing mic input continuously."""
    audio_queue.queue.clear()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback
    )
    stream.start()
    print("Recording started...")
    return stream

def stop_recording(stream):
    """Stop recording, save file, and return filename."""
    print("Recording stopped. Saving file...")
    stream.stop()
    stream.close()

    frames = []
    while not audio_queue.empty():
        frames.append(audio_queue.get())

    if not frames:
        print("No audio data captured.")
        return None

    audio = np.concatenate(frames, axis=0)
    sf.write(FILENAME, audio, SAMPLE_RATE, subtype='PCM_16')
    print("Saved to", FILENAME)
    return FILENAME

# ==== FAST TRANSCRIPTION ====
def transcribe_audio(filename):
    print("Transcribing (fast mode)...")
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = True

    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"Transcription complete: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Google Speech API error: {e}")
        return ""

# ==== AI + VOICE ====
def get_ai_response(prompt):
    print("Thinking...")
    response = client_ai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def speak(text):
    print("Speaking...")
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)

# ==== P KEY CONTROL ====
def process_audio(filename):
    if not filename:
        return
    user_text = transcribe_audio(filename)
    print(f"You said: {user_text}")
    if not user_text:
        return
    response = get_ai_response(user_text)
    print(f"AI: {response}")
    speak(response)

def on_press(key):
    global current_stream
    try:
        if key.char == 'p' and current_stream is None:
            current_stream = start_recording()
        elif key.char == 'q':
            print("\nExiting...")
            if current_stream:
                current_stream.stop()
                current_stream.close()
            exit()
    except AttributeError:
        pass

def on_release(key):
    global current_stream
    try:
        if key.char == 'p' and current_stream is not None:
            filename = stop_recording(current_stream)
            current_stream = None
            process_audio(filename)
    except AttributeError:
        pass

# ==== START LISTENER ====
try:
    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except KeyboardInterrupt:
    print("\nExiting voice recognition...")
    if current_stream:
        current_stream.stop()
        current_stream.close()
    print("Goodbye!")
