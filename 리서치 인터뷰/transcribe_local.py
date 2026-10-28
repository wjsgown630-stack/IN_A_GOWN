from pathlib import Path
import json
import sys

from faster_whisper import WhisperModel


audio_path = Path(__file__).with_name("인터뷰.m4a")
output_path = Path(__file__).with_name("전사_원본.jsonl")
model_name = sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(model_name, device=device, compute_type=compute_type)
segments, info = model.transcribe(
    str(audio_path),
    language="ko",
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False,
    word_timestamps=False,
)
print(f"language={info.language}, duration={info.duration:.2f}", flush=True)
with output_path.open("w", encoding="utf-8") as output:
    for segment in segments:
        item = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "avg_logprob": round(segment.avg_logprob, 3),
        }
        output.write(json.dumps(item, ensure_ascii=False) + "\n")
        output.flush()
        print(f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.text.strip()}", flush=True)
