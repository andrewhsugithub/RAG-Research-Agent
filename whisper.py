import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

device = "cuda:2" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=64,  # takes 34GB of VRAM beam 3, or 17GB of VRAM beam 1, takes 5928mb of VRAM if no generate_kwargs
    torch_dtype=torch_dtype,
    device=device,
)

generate_kwargs = {
    # "num_beams": 1,  # time taken: 639.6 seconds for 3 beams, 459 seconds for 1 beam
    # "condition_on_prev_tokens": False,
    # "compression_ratio_threshold": 1.35,
    # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    # "logprob_threshold": -1.0,
    # # "no_speech_threshold": 0.6, #! will error
}

# sample = "sample1.flac"
sample = "audio/audio.mp3"

import time

start = time.time()

result = pipe(
    sample, return_timestamps=True, generate_kwargs=generate_kwargs
)  # time taken: 168 seconds without generate_kwargs
print("time taken: ", time.time() - start)

docs = result["chunks"]
print("✅ Extracted Chunks")

file = "audio/audio.txt"
with open(file, "w") as f:
    f.write(json.dumps(docs, ensure_ascii=False))
print("✅ Saved Chunks to ", file)

from retriever import QdrantRetriever
from loader import Loader
from get_embedding import get_embedding_function

docs = Loader().load(documents=docs)
print("✅ Loaded Documents")

QdrantRetriever(
    collection_name="yt", embedding_func=get_embedding_function, device=device
).populate(documents=docs)
