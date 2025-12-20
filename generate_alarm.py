import numpy as np
import wave
import struct

def generate_alarm(filename="alarm.wav", duration=1.0, freq=440.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a Square Wave for a harsh "Industrial" alarm sound
    # Modulate frequency to make it sound like a siren (440Hz to 880Hz)
    audio = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
    
    # Add a second tone for dissonance
    audio += 0.5 * np.sign(np.sin(2 * np.pi * (freq + 50) * t))
    audio = audio * 0.5 # Normalize
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_alarm()
