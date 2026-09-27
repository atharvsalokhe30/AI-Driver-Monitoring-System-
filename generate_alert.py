"""Generate a simple beep alert WAV file for the driver monitoring system."""
import struct, wave, math, os

os.makedirs("assets", exist_ok=True)
sample_rate = 44100
duration    = 0.3   # seconds
frequency   = 1000  # Hz
amplitude   = 28000

num_samples = int(sample_rate * duration)
samples = [
    int(amplitude * math.sin(2 * math.pi * frequency * t / sample_rate))
    for t in range(num_samples)
]

with wave.open("assets/alert.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(struct.pack(f"<{num_samples}h", *samples))

print("alert.wav generated successfully.")
