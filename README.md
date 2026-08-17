# H3 Prompt Studio

A desktop application (Python / CustomTkinter) for building structured prompts compliant with the **MiniMax H3** format (open-source audio-video model) through guided forms instead of hand-writing the prompt. Runs 100% locally via an LLM (Ollama, LM Studio, or llama.cpp) — no data is sent to any external service.

H3 expects a rigid, structured, labelled prompt format that is very different from a natural-language description. The app does the round-trip: you fill in simple fields (scenario, style, dialogue, references...), and the local LLM turns that into a full, compliant H3 prompt.

---

## Table of contents

- [Features](#features)
- [The 5 generation modes](#the-5-generation-modes)
- [Story → Sequences tab](#story--sequences-tab)
- [Supported LLM backends](#supported-llm-backends)
- [Installation](#installation)
- [Configuration](#configuration)
- [Code architecture](#code-architecture)
- [Known limitations](#known-limitations)

---

## Features

- **5 H3 generation modes** covering every use case of the model (text only, first-frame image, first+last frame, last frame only, multi-reference).
- **Storyboard mode** inside Ref2VA: splits a scene into several chained scenes, each with its own dialogue and its own references, generated in a single sequence.
- **Story → Sequences tab**: starting from a library of reference images, generates a short story and then automatically splits it into N video sequences, each expanded into a full Ref2VA prompt.
- **Automatic image description** via a local vision model (e.g. `qwen2.5vl`), to avoid describing every reference by hand.
- **Streaming** generation, token by token, into the output pane.
- **3 interchangeable LLM backends**: Ollama, LM Studio, llama.cpp (llama-server) — all fully local.
- **Persistent settings** (backend, URLs, models, temperature, default duration) automatically saved to `~/.h3_prompt_studio_config.json`.
- **Session save/load** for the Story → Sequences tab (references, story, sequences, and settings) as a JSON file.
- **Export** of the generated prompt to clipboard or to a `.txt` file.

---

## The 5 generation modes

Each mode has its own tab, with a form tailored to it (scenario, visual style, camera notes, dialogue, soundscape, music), producing a compliant H3 prompt (fields `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`, or Ref2VA's 6-section structure).

| Mode | Description | Input |
|---|---|---|
| **T2VA** | Text → Video+Audio | No image; everything is built from a text scenario. |
| **I2VA** | Image → Video+Audio | One image anchors the first frame (0.00s); the LLM describes what happens next. |
| **FL2VA** | First + Last frame → Video+Audio | Two images anchor the start and the end; the LLM describes the motion connecting them. |
| **L2VA** | Last frame only → Video+Audio | One image anchors the end; the LLM invents a plausible starting state. |
| **Ref2VA** | Multi-reference (full mode) | Up to 9 images / 3 videos / 3 audio clips (12 references max); the LLM assigns `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>` labels and writes the 6 prompt sections (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music). |

Ref2VA also has an optional **storyboard mode**: define N scenes, each with its own brief, dialogue, and selection of references from the shared library; every scene is expanded into its own standalone H3 prompt.

Fields shared across all modes ("Dialogue / voice" + "Sound & music" blocks):
- Dialogue with speaker description, language, exact text, voiceover option.
- On-screen text (signs, subtitles...).
- Soundscape and music (or explicit silence).
- Free extra instructions for the LLM.

---

## Story → Sequences tab

Builds one Ref2VA prompt per sequence from a single reference library, in 3 LLM passes:

1. **References**: add images/videos/audio with a role and a description (same as Ref2VA), with automatic description via a vision model.
2. **Story generation** (Pass A): the LLM writes a short narrative (~200-500 words) that uses the provided references, based on an optional premise and a target language.
3. **Sequence breakdown** (Pass B): the LLM receives the story, the reference library, and the allowed camera-motion vocabulary, and returns a JSON array of N sequences — for each one it picks the relevant references, whether dialogue is present, and the camera movement itself. A reconciliation step automatically re-adds any reference the LLM forgot to list (keyword overlap between a sequence's brief and the reference descriptions).
4. **Final prompt generation**: each sequence is expanded into a full, isolated Ref2VA prompt (strict isolation rule: content from one sequence must never leak into another).

Robustness: repair of truncated JSON (a common failure mode caused by local models' `max_tokens` limits), with automatic retry on parsing failure.

**Save/load**: dedicated buttons to save the entire session (references, premise, language, story, sequence count/duration, style, extra instructions, generated sequences) to a `.json` file, and reload it later to resume work.

---

## Supported LLM backends

| Backend | API | Default host |
|---|---|---|
| **Ollama** | native (`/api/tags`, `/api/chat`) | `http://localhost:11434` |
| **LM Studio** | OpenAI-compatible (`/v1/models`, `/v1/chat/completions`) | `http://localhost:1234` |
| **llama.cpp** (llama-server) | OpenAI-compatible (`/v1/models`, `/v1/chat/completions`) | `http://localhost:8080` |

The backend is selectable in the Settings tab; all three URLs are kept independently (switching backends doesn't lose the other ones' config). Chat and image description (vision) both work across all three backends, including streaming.

---

## Installation

```bash
pip install customtkinter requests
```

One of the three backends must be running locally:

```bash
# Ollama
ollama serve
ollama pull qwen2.5:14b-instruct     # chat model
ollama pull qwen2.5vl:7b             # vision model (optional, for automatic image description)
```

Then launch the app:

```bash
python main.py
```

**Recommendation**: a 14B+ chat model follows the structured H3 format far better than a small model.

---

## Configuration

In the **⚙ Settings** tab:
- Backend selection and its URLs.
- "Test connection / refresh models" button to list the models available on the active backend.
- Choice of chat model (text) and vision model (optional).
- Temperature and default duration.

Settings are automatically saved to `~/.h3_prompt_studio_config.json` when the app closes.

---

## Code architecture

```
main.py               # CustomTkinter UI: per-mode tabs, Ref2VA + storyboard,
                       # Story→Sequences, Settings, output/streaming pane
system_prompts.py      # H3 rules (camera, dialogue, on-screen text, sound, duration,
                       # timestamps) assembled into per-mode system prompts
sequence_pipeline.py    # Story→Sequences pipeline: story generation, JSON sequence
                       # breakdown, truncated-JSON repair, reference reconciliation,
                       # brief construction
llm_client.py          # Multi-backend abstraction (Ollama / LM Studio / llama.cpp):
                       # streaming chat, model listing, image description
```

---

## Known limitations

- Ref2VA / Story→Sequences: maximum 9 images, 3 videos, 3 audio clips, 12 references total (H3 model limit).
- The quality of structured JSON output (sequence breakdown, H3 prompts) depends heavily on the local chat model's ability to follow strict formatting instructions — smaller models (<7B) are prone to truncation or malformed JSON, partially mitigated by the built-in repair mechanisms.
- The app does not perform any actual video/audio generation: it only produces the text prompt meant to be fed into H3 (via ComfyUI or another compatible runtime).
