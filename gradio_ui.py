import gradio as gr
import threading
from threading import Event
from llama_cpp import Llama
import os


# -------------------------------
# Load Models ONCE at startup
# -------------------------------
def load_models():
    print("Loading local GGUF models...")

    mistral_path = "./mistral_model"
    qwen_path = "./qwen_model"

    mistral_model = Llama(
        model_path=os.path.join(mistral_path, "ggml-model-q4_k_m.gguf"),
        n_ctx=4096,
        n_threads=6,
        verbose=False,
    )

    qwen_model = Llama(
        model_path=os.path.join(qwen_path, "ggml-model-q4_k_m.gguf"),
        n_ctx=4096,
        n_threads=6,
        verbose=False,
    )

    return mistral_model, qwen_model


mistral, qwen = load_models()


# -------------------------------
# Your classifier
# -------------------------------
def classify_task(user_input):
    coding_keywords = [
        "code", "function", "python", "class", "debug",
        "java", "c++", "script", "error", "compile",
    ]
    math_keywords = ["solve", "equation", "calculate", "integral", "derivative"]

    input_lower = user_input.lower()

    if any(word in input_lower for word in coding_keywords):
        return "coding"
    if any(word in input_lower for word in math_keywords):
        return "math"

    return "general"


# -------------------------------
# Streaming LLM generator
# -------------------------------
def generate_stream(model, prompt, stop_event):
    full_output = ""
    for chunk in model(prompt, max_tokens=512, stream=True):
        if stop_event.is_set():
            break

        token = chunk["choices"][0]["text"]
        full_output += token
        yield full_output


# -------------------------------
# Chat handler
# -------------------------------
def chat_fn(message, history, stop_event):
    # HF ChatInterface gives history as list of [user, ai] pairs
    # Build prompt in simple concat format

    prompt = ""
    for user_msg, ai_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {ai_msg}\n"
    prompt += f"User: {message}\nAssistant: "

    # Choose model
    task = classify_task(message)
    model = qwen if task in ["coding", "math"] else mistral

    # Stream tokens
    for partial in generate_stream(model, prompt, stop_event):
        yield partial


# -------------------------------
# Stop Button (shared event)
# -------------------------------
stop_event = Event()

def stop_generation():
    stop_event.set()
    return "⛔ Stopped."


# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local Dual-Model LLM (Mistral + Qwen)\nStreaming · Stop Button · Auto Routing")

    with gr.Row():
        with gr.Column(scale=12):
            chat = gr.ChatInterface(
                lambda msg, history: chat_fn(msg, history, stop_event),
                type="messages",
                title="Your AI Assistant"
            )

        with gr.Column(scale=3):
            stop_btn = gr.Button("Stop Generation", variant="stop")
            stop_btn.click(stop_generation, None, chat)

    # Reset stop flag before each new request
    demo.load(lambda: stop_event.clear(), None, None)
    chat.clear(lambda: stop_event.clear(), None, None)


demo.queue()
demo.launch()
