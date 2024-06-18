from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio.transforms as T
import whisper
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger

from kxagent.voice.tts.core import TTSModel

MEMORY_LENGTH = 2
CONTEXT_SIZE = 2048
MAX_TOKENS = 512

asr = whisper.load_model("small")
llm = Llama.from_pretrained(
    repo_id="Spiral-AI/Anonymous-7b-gguf",
    filename="*.gguf",
    # n_gpu_layers=1,
)

repo_id = "litagin/style_bert_vits2_jvnv"

model_path = hf_hub_download(
    repo_id, filename="jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
)
config_path = hf_hub_download(repo_id, filename="jvnv-F1-jp/config.json")
style_vec_path = hf_hub_download(repo_id, filename="jvnv-F1-jp/style_vectors.npy")

tts = TTSModel(
    model_path=Path(model_path),
    config_path=Path(config_path),
    style_vec_path=Path(style_vec_path),
    device="mps",
)
tts.load()


def compose_prompt(history):
    prompt = "<|endoftext|>"
    for uttrs in history:
        prompt += f"USER: {uttrs[0]}\n"
        if uttrs[1] is not None:
            prompt += f"ASSISTANT: {uttrs[1]}<|endoftext|>\n"

    prompt += "ASSISTANT: "
    logger.debug(f"\nPrompt: {prompt}")

    return prompt


def text2speech(history):
    text = history[-1][1]
    sr, audio = tts.infer(text)
    return sr, audio


def speech2text(audio, history):
    sr, y = audio

    # 整数型から浮動小数点型へ変換
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # サンプルレートをWhisperが対応する16kHzへリサンプリング
    y_tensor = torch.from_numpy(y).clone()
    resample_rate = whisper.audio.SAMPLE_RATE
    resampler = T.Resample(sr, resample_rate, dtype=y_tensor.dtype)
    y2_tensor = resampler(y_tensor)
    y2_float = y2_tensor.to("cpu").detach().numpy().copy()

    # 音声認識
    result = asr.transcribe(y2_float, verbose=True, fp16=False, language="ja")
    text = result["text"]
    history += [[text, None]]

    return history


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history):
    # プロンプトを作成
    prompt = compose_prompt(history)
    print(prompt)

    # 推論
    streamer = llm.create_completion(prompt, max_tokens=MAX_TOKENS, stream=True)

    # 推論結果をストリーム表示
    history[-1][1] = ""
    for msg in streamer:
        message = msg["choices"][0]
        if "text" in message:
            new_token = message["text"]
            if new_token != "<":
                history[-1][1] += new_token
                yield history


def clear_audio_in():
    return None


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="チャット")
    msg = gr.Textbox("", label="あなたからのメッセージ")
    clear = gr.Button("チャット履歴の消去")
    audio_in = gr.Audio(sources=["microphone"], label="🎙️")
    audio_out = gr.Audio(type="numpy", label="AIからのメッセージ", autoplay=True)

    # テキスト入力時のイベントハンドリング
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    ).then(text2speech, chatbot, audio_out)

    # 音声入力時のイベントハンドリング
    audio_in.stop_recording(
        speech2text, [audio_in, chatbot], chatbot, queue=False
    ).then(bot, chatbot, chatbot).then(text2speech, chatbot, audio_out).then(
        clear_audio_in, outputs=[audio_in], queue=False
    )

    # チャット履歴の消去
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch()
