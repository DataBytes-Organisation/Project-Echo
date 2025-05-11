import os
import subprocess

AUDIO_DIR = "./local_audio"

def simulate():
    print("🔊 Starting local audio simulation using ffplay...")
    for file in os.listdir(AUDIO_DIR):
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            file_path = os.path.join(AUDIO_DIR, file)
            print(f"▶ Playing: {file}")
            subprocess.run(["ffplay", "-nodisp", "-autoexit", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    simulate()
