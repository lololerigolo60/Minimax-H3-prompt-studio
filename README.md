MINIMAX H3 Prompt Studio

A local desktop app (Python / CustomTkinter) that helps you build structured MiniMax H3 video-generation prompts, powered by a local LLM — no cloud, no API keys, no subscription.

It walks you through your scene mode by mode (T2VA, I2VA, FL2VA, L2VA, Ref2VA), collects your raw description, reference images, and camera notes, then delegates the final formatting to a local model so the output matches the MiniMax H3 structured prompt spec.

Features
All 5 MiniMax H3 modes: Text-to-Video-Audio, Image-to-Video-Audio, First/Last-Frame-to-Video-Audio, Last-frame-to-Video-Audio, and Reference-to-Video-Audio.
Two local LLM backends, switchable in Settings:
Ollama (native /api/chat)
LM Studio (OpenAI-compatible /v1/chat/completions)
Vision-assisted reference description: point the app at a reference image and let a local vision model (qwen2.5vl, llava, etc.) describe it for you automatically — works with either backend.
Streaming output panel: watch the structured prompt get generated token by token.
Persistent settings: server URLs, active backend, chat/vision model, temperature, and default duration are saved to ~/.h3_prompt_studio_config.json.
100% local: no data leaves your machine.
Requirements
pip install customtkinter requests

Plus one of the following, running locally:

Backend	Setup
Ollama	Install Ollama, run ollama serve, then pull a chat model: ollama pull qwen2.5:14b-instruct (and optionally a vision model: ollama pull qwen2.5vl:7b)
LM Studio	Install LM Studio, load a chat model (and optionally a vision model), then start the local server (Developer → Start Server, default port 1234)

A model in the 14B+ range is strongly recommended — small models tend to drift from the structured H3 format.

Usage
python main.py
Go to the ⚙ Settings tab, pick your backend (Ollama or LM Studio), set its server URL, then click Test connection / refresh models to populate the model list.
Choose a chat model (required) and, optionally, a vision model for auto-describing reference images.
Pick a mode tab (T2VA / I2VA / FL2VA / L2VA / Ref2VA), fill in your scenario, style, camera notes, and reference assets.
Click Generate — the structured H3 prompt streams into the right-hand panel, ready to Copy or Save as .txt.
Project structure
main.py             # CustomTkinter UI, app state, generation logic
ollama_client.py     # Unified client for Ollama / LM Studio (chat, vision, model listing)
system_prompts.py    # H3 mode system prompts, style/camera/language option lists
Notes
Switching backends in Settings keeps both server URLs in memory — you can flip between Ollama and LM Studio without re-typing the other one's address.
All backend differences (native vs. OpenAI-compatible chat, streaming format, vision payload shape) are absorbed inside ollama_client.py; the rest of the app only ever calls client.chat(...), client.list_models(), and client.describe_image(...).
