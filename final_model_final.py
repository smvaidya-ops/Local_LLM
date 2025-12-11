# # model_final.py ← FINAL VERSION: No more echoing, no crashes, super fast
# from ctransformers import AutoModelForCausalLM
# from llama_cpp import Llama
# import gradio as gr
# import re
# import threading

# # ==============================
# # LOAD MODELS – OPTIMAL SPEED
# # ==============================
# print("Loading Mistral...")
# mistral_model = AutoModelForCausalLM.from_pretrained(
#     r"./quant_model.gguf",           # ← your Mistral GGUF file
#     n_ctx=8192,
#     n_threads=6,           # increase if you have more cores
#     n_batch=512,
#     n_gpu_layers=0,        # change to 35+ if you have GPU
#     use_mlock=True,
#     verbose=False
# )

# print("Loading Qwen2.5-Coder...")
# qwen_model = Llama(
#     r"./qwen2.5-coder-7b-instruct-q4_k_m.gguf",
#     n_ctx=8192,
#     n_threads=4,       # Fastest on CPU
#     n_batch=512,       # Fastest on CPU
#     n_gpu_layers=0,    # Change to 35–99 if GPU
#     use_mlock=True,
#     verbose=False
# )

# stop_event = threading.Event()

# # ==============================
# # BULLETPROOF DETECTION — MATH + CODE = ALWAYS QWEN
# # ==============================

# def is_coding_or_math(text: str) -> bool:
#     text = text.lower()
    
#     # Math & number series triggers
#     math_triggers = [
#     "next number", "series", "sequence", "pattern", "find the next", "what comes next",
#     "solve", "calculate", "equation", "math", "mathematics", "integral", "derivative",
#     "factorial", "prime", "geometry", "algebra", "probability", "statistics",
#     "seconds", "minutes", "hours", "number", "square", "squares", "sum of squares", "convert",
#     "decimal", "binary"
# ]

#     # Coding triggers
#     code_triggers = [
#         "code", "program", "write a", "implement", "function", "class", "python", "java",
#         "c++", "javascript", "sql", "debug", "algorithm", "leetcode", "binary search"
#     ]
    
#     # If any math or code keyword is found → Qwen
#     if any(trigger in text for trigger in math_triggers + code_triggers):
#         return True
        
#     # If contains numbers + math symbols → Qwen
#     if re.search(r'\d', text) and any(op in text for op in "+-*/=^()[]{}"):
#         return True
        
#     # If contains comma-separated numbers (like 2, 6, 12, 20) → Qwen
#     if re.search(r'\d+\s*[,]\s*\d+', text):
#         return True
        
#     return False

# # ==============================
# # API ANSWER FUNCTION (NO STREAMING)
# # ==============================
# def api_answer(prompt: str):
#     """
#     This function returns a clean **single-shot response**
#     (not streaming) from the correct model.
#     Useful for CSV generation, batch processing, etc.
#     """

#     # Detect which model to use
#     use_qwen = is_coding_or_math(prompt)

#     # --------------------------
#     # QWEN ANSWERING
#     # --------------------------
#     if use_qwen:
#         formatted = (
#             "System: You are a world-class math and coding assistant. "
#             "Always give clean, direct answers.\n\n"
#             "User: " + prompt + "\n"
#             "Assistant:"
#         )

#         output = qwen_model(formatted, max_tokens=800, temperature=0.1, stream=False)
#         return output["choices"][0]["text"].strip()

#     else:
#         # MISTRAL — now super fast with llama.cpp
#         system = "You are a helpful, concise assistant. Answer directly."
#         full_prompt = f"<s>[INST] <<SYS>>{system}<</SYS>>\n{prompt} [/INST]"
#         result = mistral_model(full_prompt, max_tokens=800, temperature=0.6, stream=False)
#         return result["choices"][0]["text"].strip()

#     # --------------------------
#     # MISTRAL ANSWERING
#     # --------------------------
#     system_prompt = (
#         "You are a helpful, concise assistant. "
#         "Do NOT repeat the user's question. "
#         "Answer directly and clearly."
#     )

#     formatted_prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {prompt} [/INST]"

#     result = mistral_model(
#         formatted_prompt,
#         max_new_tokens=800,
#         temperature=0.6,
#         top_p=0.9,
#         top_k=30,
#         repetition_penalty=1.1,
#         stream=False
#     )

#     return result.strip()

# # ==============================
# # FIXED STREAMING (NO ECHOING!)
# # ==============================
# def stream_mistral(prompt):
#     stop_event.clear()

#     system_prompt = (
#         "You are a helpful, concise assistant. "
#         "Do NOT repeat the user's question. "
#         "Answer directly and clearly."
#         "Give math output in proper chatgpt style format"
#         "You are a helpful AI. Do NOT prefix your answers with any labels such as 'Assistant:' or 'AI:'. "
#         "Answer directly and clearly."
#     )

#     formatted_prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {prompt} [/INST]"

#     yield "**[Mistral]**\n\n"

#     full = ""
#     for chunk in mistral_model(
#         formatted,
#         max_tokens=800,
#         temperature=0.6,
#         top_p=0.9,
#         top_k=30,
#         repeat_penalty=1.1,
#         stream=True,
#         stop=["</s>", "<s>"]
#     ):
#         if stop_event.is_set():
#             break
#         token = chunk["choices"][0]["text"]
#         full += token
#         yield "**[Mistral]**\n\n" + full.strip()

# def stream_qwen(prompt):
#     stop_event.clear()
#     resp = ""

#     yield [{"role": "assistant", "content": "**[Qwen2.5-Coder]**\n\n"}]

#     formatted = (
#         "System: You are a world-class math and coding assistant. "
#         "Always give clean, direct answers.\n\n"
#         "User: " + prompt + "\n"
#         "Assistant:"
#     )

#     for chunk in qwen_model(
#         formatted,
#         stream=True,
#         max_tokens=800,
#         temperature=0.1,
#         top_p=0.9,
#         top_k=20,
#         repeat_penalty=1.05
#     ):
#         if stop_event.is_set():
#             break

#         choice = chunk["choices"][0]
#         token = (
#             choice.get("text") or
#             choice.get("delta", {}).get("content", "") or
#             ""
#         )

#         resp += token
#         yield [{"role": "assistant", "content": f"**[Qwen2.5-Coder]**\n\n{resp}"}]

# # ==============================
# # MAIN CHAT — WORKS WITH MESSAGES FORMAT
# # ==============================
# def chat(message, history):
#     stop_event.clear()

#     # Handle history as list of dicts (Gradio's type="messages")
#     messages = []
#     for msg in history:
#         if isinstance(msg, dict) and "role" in msg:
#             messages.append(msg)
#         else:
#             # Fallback for tuples (old format)
#             for u, a in msg if isinstance(msg, (list, tuple)) else []:
#                 if u: messages.append({"role": "user", "content": u})
#                 if a: messages.append({"role": "assistant", "content": a})
#     messages.append({"role": "user", "content": message})

#     streamer = stream_qwen(message) if is_coding_or_math(message) else stream_mistral(message)

#     partial = messages.copy()
#     first = True
#     for chunk in streamer:
#         if stop_event.is_set(): break
#         if first:
#             partial.append(chunk[0])
#             first = False
#         else:
#             partial[-1] = chunk[0]
#         yield partial

# def stop():
#     stop_event.set()

# # ------------------------------------------------------
# # STREAMLIT UI STARTS HERE
# # ------------------------------------------------------

# import streamlit as st

# st.set_page_config(page_title="Dual Local AI", layout="wide")
# st.title("Dual Local AI — Qwen for Code/Math | Mistral for Chat")

# # SESSION STATE
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# tab1, tab2 = st.tabs(["💬 Chatbot", "🧪 API Tester"])

# # ------------------------------------------------------
# # TAB 1 — LIVE CHATBOT (streaming)
# # ------------------------------------------------------
# with tab1:
#     st.subheader("Chat with your local models")

#     # Display previous messages
#     for m in st.session_state.messages:
#         st.chat_message(m["role"]).write(m["content"])

#     user_input = st.chat_input("Type your message...")

#     if user_input:
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         st.chat_message("user").write(user_input)

#         is_qwen = is_coding_or_math(user_input)

#         # Choose model
#         streamer = stream_qwen(user_input) if is_qwen else stream_mistral(user_input)

#         model_name = "Qwen2.5-Coder" if is_qwen else "Mistral"
#         bot_placeholder = st.chat_message("assistant").container()
#         full_response = ""

#         with bot_placeholder:
#             st.write(f"**[{model_name}]**")
#             response_box = st.empty()

#             for chunk in streamer:
#                 full_response = chunk
#                 response_box.write(full_response)

#         st.session_state.messages.append({"role": "assistant", "content": full_response})

# # ------------------------------------------------------
# # TAB 2 — API TESTER (Simple)
# # ------------------------------------------------------
# with tab2:
#     st.subheader("API Tester — Same as Gradio API")

#     prompt = st.text_area("Enter prompt:", height=120)
#     if st.button("Send"):
#         resp = api_answer(prompt)
#         st.text_area("Response:", value=resp, height=300)


# app.py — YOUR FINAL DUAL AI IN STREAMLIT (MISTRAL NOW FLYING FAST)
from llama_cpp import Llama
import streamlit as st
import re
import threading
import os
st.set_page_config(page_title="Dual Local AI", layout="centered")

# ==============================
# LOAD BOTH MODELS — BOTH NOW SUPER FAST
# ==============================
@st.cache_resource
def load_models():
    import huggingface_hub

    hf_token = st.secrets["HF_TOKEN"]

    # import os
    # import 
    huggingface_hub.login(token=hf_token)
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)


    # --- download mistral model ---
    mistral_path = huggingface_hub.hf_hub_download(
        repo_id="sana123456/Mistral",
        filename="quant_model.gguf",
        repo_type="model",
        cache_dir=cache_dir,
        token=hf_token,
    )
    
    with st.spinner("Loading Mistral (fast llama.cpp)..."):
        mistral = Llama(
            # r"./quant_model.gguf",
            model_path=mistral_path,
            n_ctx=8192,
            n_threads=8,
            n_batch=512,
            n_gpu_layers=0,
            verbose=False
        )
    with st.spinner("Loading Qwen2.5-Coder..."):
        qwen = Llama(
            model_path=qwen_path,
            # r"./qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            n_ctx=8192,
            n_threads=8,
            n_batch=512,
            n_gpu_layers=0,
            verbose=False
        )
    return mistral, qwen

mistral_model, qwen_model = load_models()

stop_event = threading.Event()

# ==============================
# BULLETPROOF DETECTION — MATH + CODE = ALWAYS QWEN
# ==============================
def is_coding_or_math(text: str) -> bool:
    text = text.lower()
    greeting_triggers = [
        "hi", "hello", "hey", "heyy", "hii", "hiii",
        "good morning", "good afternoon", "good evening",
        "how are you", "whats up", "what's up", "sup",
        "yo", "hola", "namaste",
        "start chat", "can we talk", "i want to chat",
        "introduce yourself", "who are you"]
    if any(g in text for g in greeting_triggers):
        return True
    math_triggers = [
        "next number", "series", "pattern", "solve", "calculate", "calculation",
        "equation", "integral", "derivative", "differentiate", "limit",
        "geometry", "algebra", "trigonometry", "matrix", "probability",
        "statistics", "mean", "median", "mode", "variance",
        "gcd", "lcm", "prime", "factorial", "percentage",
        "convert", "binary", "hex", "octal", "decimal"
    ]

    # CODING triggers
    code_triggers = [
        "code", "program", "script", "function", "class",
        "python", "java", "javascript", "typescript", "c++", "c#", "php", "sql",
        "debug", "error", "bug", "exception",
        "api", "endpoint", "json", "website",
        "algorithm", "data structure", "leetcode",
        "write a", "generate code", "fix this code",
        "regex", "loop", "list comprehension",
        "html", "css", "react", "nextjs", "node", "express", "django", "flask"
    ]
    if any(t in text for t in math_triggers + code_triggers):
        return True
    if re.search(r'\d', text) and any(op in text for op in "+-*/=^()[]{}"):
        return True
    if re.search(r'\d+\s*[,]\s*\d+', text):
        return True
    return False

# ==============================
# SAFETY FILTER — BLOCK PERSONAL / HARMFUL QUESTIONS
# ==============================
def is_forbidden_question(text: str) -> bool:
    text = text.lower()
    forbidden_keywords = [
        "suicide", "kill myself", "hurt myself", "self-harm", "die", "depression help",
        "how to make bomb", "how to make drug", "illegal", "hack", "steal", "murder",
        "my location", "my name", "my phone", "my address", "who am i", "where do i live",
        "girlfriend", "boyfriend", "crush", "love life", "break up", "divorce",
        "i'm sad", "i feel lonely", "nobody loves me", "i want to die"
    ]
    return any(kw in text for kw in forbidden_keywords)
# ==============================
# FINAL STREAMING — STOPS CORRECTLY AND KEEPS TEXT
# ==============================
def stream_response(prompt: str):
    stop_event.clear()
    use_qwen = is_coding_or_math(prompt)
    model = qwen_model if use_qwen else mistral_model

    if use_qwen:
        full_prompt = f"System: You are a world-class coding/math assistant. Answer directly.\nUser: {prompt}\nAssistant:"
    else:
        full_prompt = (
            "<|system|>\nYou are a helpful, concise assistant. "
            "Do NOT repeat the user's question. Answer directly.\n"
            "<|user|>\n" + prompt + "\n"
            "<|assistant|>"
        )

    response = ""
    try:
        for chunk in model(full_prompt, max_tokens=800, temperature=0.7, stream=True, stop=["</s>"]):
            if stop_event.is_set():
                if response.strip():
                    yield response.strip()
                else:
                    yield "[Stopped — no output yet]"
                return  # ← stops the generator completely

            token = chunk["choices"][0]["text"]
            response += token
            yield response + "▌"   # live typing effect
    except Exception:
        yield response.strip() if response.strip() else "[Stopped]"

    yield response.strip()  # final clean version

# ==============================
# API ANSWER FUNCTION (NO STREAMING) — EXACT SAME AS YOURS
# ==============================
def api_answer(prompt: str):
    use_qwen = is_coding_or_math(prompt)
    if use_qwen:
        formatted = (
            "System: You are a world-class math and coding assistant. "
            "Always give clean, direct answers.\n\n"
            "User: " + prompt + "\n"
            "Assistant:"
        )
        output = qwen_model(formatted, max_tokens=800, temperature=0.1, stream=False)
        return output["choices"][0]["text"].strip()
    else:
        system = "You are a helpful, concise assistant. Answer directly."
        full_prompt = f"<s>[INST] <<SYS>>{system}<</SYS>>\n{prompt} [/INST]"
        result = mistral_model(full_prompt, max_tokens=800, temperature=0.6, stream=False)
        return result["choices"][0]["text"].strip()

# ==============================
# STREAMING — BOTH MODELS ULTRA FAST
# ==============================
def stream_mistral(prompt):
    stop_event.clear()
    system_prompt = (
        "You are a helpful, concise assistant. "
        "Do NOT repeat the user's question. "
        "Answer directly and clearly."
    )
    formatted = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {prompt} [/INST]"
    yield "**[Mistral]**\n\n"
    full = ""
    for chunk in mistral_model(formatted, max_tokens=800, temperature=0.6, top_p=0.9, top_k=30, repeat_penalty=1.1, stream=True, stop=["</s>"]):
        if stop_event.is_set(): break
        token = chunk["choices"][0]["text"]
        full += token
        yield "**[Mistral]**\n\n" + full.strip()

def stream_qwen(prompt):
    stop_event.clear()
    yield "**[Qwen2.5-Coder]**\n\n"
    formatted = (
        "System: You are a world-class math and coding assistant. "
        "Always give clean, direct answers.\n\n"
        "User: " + prompt + "\n"
        "Assistant:"
    )
    resp = ""
    for chunk in qwen_model(formatted, stream=True, max_tokens=800, temperature=0.1, top_p=0.9, top_k=20, repeat_penalty=1.05):
        if stop_event.is_set(): break
        token = chunk["choices"][0].get("text", "")
        resp += token
        yield "**[Qwen2.5-Coder]**\n\n" + resp

# ==============================
# STREAMLIT UI — CLEAN & EXACT SAME BEHAVIOR
# ==============================
st.markdown("Local LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input + Send/Stop
# Input + Send/Stop
# Input + Send/Stop
col1, col2 = st.columns([8, 1])
prompt = col1.chat_input("Ask anything...")
stop_btn = col2.button("Stop", type="secondary", use_container_width=True)

# ==============================
# MAIN INPUT HANDLER WITH SAFETY
# ==============================
if prompt:
    # === SAFETY FILTER FIRST ===
    if is_forbidden_question(prompt):
        reply = "I'm sorry, but I can't assist with personal, emotional, or harmful requests. I'm here to help with coding, math, science, and general knowledge questions."
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": reply})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        # === NORMAL CHAT FLOW ===
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for chunk in stream_response(prompt):
                if stop_event.is_set():
                    full_response += "\n\n[Stopped]"
                    break
                full_response = chunk
                placeholder.markdown(full_response + "[cursor]")

            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==============================
# STOP BUTTON
# ==============================
if stop_btn:
    stop_event.set()
    st.rerun()
