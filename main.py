"""
H3 Prompt Studio
================
Application locale (CustomTkinter) qui interroge l'utilisateur, mode par mode
(T2VA / I2VA / FL2VA / L2VA / Ref2VA), pour construire la description brute
d'une scène, puis délègue à un LLM local via Ollama la mise en forme finale
au format de prompt structuré MiniMax H3 (tel que défini dans les guides
Drive minimaxh3/*).

Lancement :
    python main.py

Dépendances :
    pip install customtkinter requests

Prérequis :
    - Ollama installé et lancé (`ollama serve`), avec au moins un modèle de
      chat pullé (ex: `ollama pull qwen2.5:14b-instruct`) et, en option, un
      modèle de vision (ex: `ollama pull qwen2.5vl:7b`) pour la description
      automatique des images de référence.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from llm_client import LLMClient, LLMError
from system_prompts import (
    CAMERA_MOTIONS,
    LANGUAGES,
    REF2VA_SYSTEM_PROMPT,
    STYLE_OPTIONS,
    build_base_system_prompt,
)
from sequence_pipeline import generate_sequence_breakdown, generate_story, sequence_to_brief

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CONFIG_PATH = Path.home() / ".h3_prompt_studio_config.json"

DEFAULT_SETTINGS = {
    "llm_backend": "ollama",        # "ollama", "lmstudio" ou "llamacpp"
    "ollama_url": "http://localhost:11434",
    "lmstudio_url": "http://localhost:1234",
    "llamacpp_url": "http://localhost:8080",
    "chat_model": "",
    "vision_model": "",
    "temperature": 0.6,
    "default_duration": 10,
}

NO_CHAT_MODEL_MARKER = "—"
NO_VISION_MODEL_MARKER = "(none)"

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIO = 3
MAX_REFS = 12


# ------------------------------------------------------------------------- #
# Persistance des réglages
# ------------------------------------------------------------------------- #
def load_settings() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            return merged
        except (OSError, json.JSONDecodeError):
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------------- #
# Petits widgets réutilisables
# ------------------------------------------------------------------------- #
class LabeledEntry(ctk.CTkFrame):
    def __init__(self, master, label, placeholder="", width=400, **kw):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, anchor="w").pack(fill="x")
        self.entry = ctk.CTkEntry(self, placeholder_text=placeholder, width=width, **kw)
        self.entry.pack(fill="x", pady=(2, 8))

    def get(self):
        return self.entry.get().strip()

    def set(self, value):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)


class LabeledText(ctk.CTkFrame):
    def __init__(self, master, label, height=90, placeholder="", **kw):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, anchor="w").pack(fill="x")
        self.box = ctk.CTkTextbox(self, height=height, wrap="word", **kw)
        self.box.pack(fill="x", pady=(2, 8))
        self._placeholder = placeholder
        if placeholder:
            self.box.insert("1.0", placeholder)
            self.box.configure(text_color="gray50")
            self.box.bind("<FocusIn>", self._clear_placeholder)

    def _clear_placeholder(self, _evt=None):
        if self.box.get("1.0", "end-1c") == self._placeholder:
            self.box.delete("1.0", "end")
            self.box.configure(text_color=("black", "white"))

    def get(self):
        text = self.box.get("1.0", "end-1c").strip()
        return "" if text == self._placeholder else text

    def set(self, value):
        self.box.delete("1.0", "end")
        self.box.insert("1.0", value)
        self.box.configure(text_color=("black", "white"))


class LabeledCombo(ctk.CTkFrame):
    def __init__(self, master, label, values, default=None, **kw):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, anchor="w").pack(fill="x")
        self.combo = ctk.CTkComboBox(self, values=values, **kw)
        if default:
            self.combo.set(default)
        self.combo.pack(fill="x", pady=(2, 8))

    def get(self):
        return self.combo.get().strip()


class ImageRefPicker(ctk.CTkFrame):
    """Champ 'chemin image' + bouton parcourir + bouton description auto (vision)
    + zone de texte pour la description finale (modifiable à la main)."""

    def __init__(self, master, app: "H3StudioApp", label: str):
        super().__init__(master, fg_color="transparent")
        self.app = app
        ctk.CTkLabel(self, text=label, anchor="w", font=ctk.CTkFont(weight="bold")).pack(fill="x")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(2, 4))
        self.path_entry = ctk.CTkEntry(row, placeholder_text="image path (optional)")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(row, text="Browse…", width=100, command=self._browse).pack(side="left", padx=(0, 6))
        ctk.CTkButton(row, text="Describe (vision)", width=130, command=self._describe).pack(side="left")

        self.desc_box = ctk.CTkTextbox(self, height=70, wrap="word")
        self.desc_box.pack(fill="x", pady=(0, 8))
        self.desc_box.insert(
            "1.0",
            "Describe the image here (subjects, appearance, pose, environment, lighting, "
            "style) or click 'Describe (vision)' if a vision model is configured.",
        )
        self.desc_box.configure(text_color="gray50")
        self.desc_box.bind("<FocusIn>", self._clear_placeholder)
        self._placeholder = True

    def _clear_placeholder(self, _evt=None):
        if self._placeholder:
            self.desc_box.delete("1.0", "end")
            self.desc_box.configure(text_color=("black", "white"))
            self._placeholder = False

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)

    def _describe(self):
        path = self.path_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Missing image", "Please select a valid image file first.")
            return
        vision_model = self.app.settings.get("vision_model", "").strip()
        if not vision_model:
            messagebox.showwarning(
                "Vision model not configured",
                "Set an Ollama vision model in the Settings tab (e.g. qwen2.5vl:7b).",
            )
            return

        def work():
            try:
                desc = self.app.client.describe_image(vision_model, path)
            except LLMError as e:
                err_msg = str(e)
                self.app.after(0, lambda err_msg=err_msg: messagebox.showerror("Vision error", err_msg))
                return
            def apply():
                self.desc_box.delete("1.0", "end")
                self.desc_box.insert("1.0", desc)
                self.desc_box.configure(text_color=("black", "white"))
                self._placeholder = False
            self.app.after(0, apply)

        threading.Thread(target=work, daemon=True).start()

    def get_description(self) -> str:
        if self._placeholder:
            return ""
        return self.desc_box.get("1.0", "end-1c").strip()

    def get_path(self) -> str:
        return self.path_entry.get().strip()


class SharedFields(ctk.CTkFrame):
    """Bloc de champs communs à tous les modes : dialogue, ambiance sonore,
    musique, notes libres."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")

        ctk.CTkLabel(self, text="Dialogue / voice", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(4, 2)
        )
        self.has_dialogue = ctk.CTkCheckBox(self, text="There is dialogue / a voice in the scene")
        self.has_dialogue.pack(anchor="w", pady=(0, 6))

        self.speaker_desc = LabeledEntry(
            self, "Speaker description",
            placeholder="e.g. young woman, soft and slightly raspy voice",
        )
        self.speaker_desc.pack(fill="x")

        self.language = LabeledCombo(self, "Dialogue language", LANGUAGES, default="French")
        self.language.pack(fill="x")

        self.dialogue_text = LabeledText(
            self, "Dialogue text (exact, will be preserved word for word)", height=70,
            placeholder="e.g. I'm getting off at the next station.",
        )
        self.dialogue_text.pack(fill="x")

        self.voiceover = ctk.CTkCheckBox(self, text="This is a voiceover, not an on-screen character speaking")
        self.voiceover.pack(anchor="w", pady=(0, 10))

        self.onscreen_text = LabeledEntry(
            self, "On-screen text (sign, screen, subtitle…)",
            placeholder='e.g. a red neon sign reading "OPEN"',
        )
        self.onscreen_text.pack(fill="x")

        ctk.CTkLabel(self, text="Sound & music", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(10, 2)
        )
        self.soundscape = LabeledText(
            self, "Desired soundscape (noises, texture) — the LLM will expand on it",
            height=70, placeholder="e.g. light rain on a window, wet footsteps",
        )
        self.soundscape.pack(fill="x")

        self.no_music = ctk.CTkCheckBox(self, text="No music (silent score)")
        self.no_music.pack(anchor="w", pady=(2, 4))

        self.music_notes = LabeledText(
            self, "Desired music (instruments, tempo, mood)", height=70,
            placeholder="e.g. sparse piano, slow tempo, low strings in the background",
        )
        self.music_notes.pack(fill="x")

        ctk.CTkLabel(self, text="Free notes", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(10, 2)
        )
        self.extra_notes = LabeledText(
            self, "Extra instructions for the LLM (optional)", height=60
        )
        self.extra_notes.pack(fill="x")

    def to_brief(self, include_dialogue: bool = True) -> str:
        lines = []
        if include_dialogue:
            if self.has_dialogue.get():
                lines.append("DIALOGUE: yes")
                if self.speaker_desc.get():
                    lines.append(f"  Speaker description: {self.speaker_desc.get()}")
                lines.append(f"  Language: {self.language.get() or 'English'}")
                if self.dialogue_text.get():
                    lines.append(f"  Exact line(s) to preserve verbatim: {self.dialogue_text.get()}")
                if self.voiceover.get():
                    lines.append("  This is an off-screen voiceover, not on-screen speech - remember to state the speaking character's lips stay closed.")
            else:
                lines.append("DIALOGUE: none - do not invent spoken dialogue.")

            if self.onscreen_text.get():
                lines.append(f"ON-SCREEN TEXT: {self.onscreen_text.get()}")

        if self.no_music.get():
            lines.append("MUSIC: explicitly none, non_diegetic_music must be N/A.")
        elif self.music_notes.get():
            lines.append(f"MUSIC HINTS: {self.music_notes.get()}")

        if self.soundscape.get():
            lines.append(f"SOUNDSCAPE HINTS: {self.soundscape.get()}")

        if self.extra_notes.get():
            lines.append(f"EXTRA INSTRUCTIONS: {self.extra_notes.get()}")

        return "\n".join(lines)


# ------------------------------------------------------------------------- #
# Onglets par mode
# ------------------------------------------------------------------------- #
class BaseModeTab(ctk.CTkScrollableFrame):
    MODE = "T2VA"
    TITLE = ""
    INTRO = ""

    def __init__(self, master, app: "H3StudioApp"):
        super().__init__(master, fg_color="transparent")
        self.app = app
        if self.INTRO:
            ctk.CTkLabel(
                self, text=self.INTRO, anchor="w", justify="left", wraplength=680, text_color="gray70"
            ).pack(fill="x", pady=(0, 10))

        self.scenario = LabeledText(
            self, "Scenario / what should happen", height=110,
            placeholder="Describe the scene: who, where, what, what mood…",
        )
        self.scenario.pack(fill="x")

        self.style = LabeledCombo(self, "Visual style", STYLE_OPTIONS, default="Cinematic")
        self.style.pack(fill="x")

        self.build_mode_fields()

        self.camera_notes = LabeledText(
            self, "Desired camera movements (optional — otherwise the LLM chooses)",
            height=60, placeholder=f"e.g. {CAMERA_MOTIONS[2]} low amplitude, slow speed",
        )
        self.camera_notes.pack(fill="x")

        self.shared = SharedFields(self)
        self.shared.pack(fill="x", pady=(6, 0))

        ctk.CTkButton(
            self, text=f"🎬 Generate H3 prompt ({self.MODE})", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.on_generate,
        ).pack(fill="x", pady=(16, 20))

    def build_mode_fields(self):
        """To override in a subclass."""
        pass

    def duration_value(self) -> str:
        raw = self.app.settings.get("default_duration", 10)
        try:
            float(raw)
        except (TypeError, ValueError):
            return "10"
        return str(raw)

    def build_brief(self) -> str:
        """To override: builds the text sent as the user message."""
        raise NotImplementedError

    def on_generate(self):
        if not self.scenario.get() and self.MODE not in ("FL2VA", "L2VA"):
            messagebox.showwarning("Missing scenario", "Please describe the scenario, at least briefly.")
            return
        brief = self.build_brief()
        system_prompt = build_base_system_prompt(self.MODE)
        self.app.run_generation(system_prompt, brief, mode_label=self.MODE)


class T2VATab(BaseModeTab):
    MODE = "T2VA"
    TITLE = "Text → Video+Audio"
    INTRO = (
        "No reference image: everything is built from text. "
        "Ideal for starting from scratch on a narrative idea."
    )

    def build_mode_fields(self):
        self.duration = LabeledEntry(self, "Target duration (seconds)", placeholder="e.g. 10")
        self.duration.set(self.duration_value())
        self.duration.pack(fill="x")

        self.shots = LabeledCombo(
            self, "Desired number of shots",
            ["auto (let the LLM decide)", "1 (single shot)", "2", "3", "4", "5+"],
            default="auto (let the LLM decide)",
        )
        self.shots.pack(fill="x")

    def build_brief(self) -> str:
        lines = [
            f"SCENARIO: {self.scenario.get()}",
            f"VISUAL STYLE: {self.style.get()}",
            f"TARGET DURATION: {self.duration.get() or self.duration_value()} seconds",
            f"DESIRED NUMBER OF SHOTS: {self.shots.get()}",
        ]
        if self.camera_notes.get():
            lines.append(f"CAMERA NOTES: {self.camera_notes.get()}")
        lines.append("")
        lines.append(self.shared.to_brief())
        return "\n".join(lines)


class I2VATab(BaseModeTab):
    MODE = "I2VA"
    TITLE = "Image → Video+Audio (first frame)"
    INTRO = (
        "An image serves as the exact first frame of shot 1 (0.00s). Describe it "
        "(or use automatic description via an Ollama vision model), "
        "then say what should happen next."
    )

    def build_mode_fields(self):
        self.first_frame = ImageRefPicker(self, self.app, "First-frame image (<Picture 1>)")
        self.first_frame.pack(fill="x", pady=(4, 10))

        self.duration = LabeledEntry(self, "Target duration (seconds)", placeholder="e.g. 8")
        self.duration.set(self.duration_value())
        self.duration.pack(fill="x")

    def build_brief(self) -> str:
        desc = self.first_frame.get_description()
        lines = [
            f"FIRST-FRAME IMAGE (<Picture 1>) DESCRIPTION: {desc or '(not described - infer from scenario)'}",
            f"SCENARIO / WHAT HAPPENS NEXT: {self.scenario.get()}",
            f"VISUAL STYLE: {self.style.get()} (should match the image if described)",
            f"TARGET DURATION: {self.duration.get() or self.duration_value()} seconds",
        ]
        if self.camera_notes.get():
            lines.append(f"CAMERA NOTES: {self.camera_notes.get()}")
        lines.append("")
        lines.append(self.shared.to_brief())
        return "\n".join(lines)


class FL2VATab(BaseModeTab):
    MODE = "FL2VA"
    TITLE = "First + Last frame → Video+Audio"
    INTRO = (
        "Two images anchor the start (0.00s) and the end (target duration) of the shot. "
        "The LLM describes the motion path connecting the two, usually in a single shot."
    )

    def build_mode_fields(self):
        self.first_frame = ImageRefPicker(self, self.app, "First-frame image (<Picture 1>)")
        self.first_frame.pack(fill="x", pady=(4, 10))

        self.last_frame = ImageRefPicker(self, self.app, "Last-frame image (<Picture 2>)")
        self.last_frame.pack(fill="x", pady=(4, 10))

        self.duration = LabeledEntry(self, "Exact target duration (seconds, e.g. 8.00)", placeholder="8.00")
        self.duration.set(f"{float(self.duration_value()):.2f}")
        self.duration.pack(fill="x")

    def build_brief(self) -> str:
        d1 = self.first_frame.get_description()
        d2 = self.last_frame.get_description()
        lines = [
            f"FIRST-FRAME IMAGE (<Picture 1>) DESCRIPTION: {d1 or '(not described)'}",
            f"LAST-FRAME IMAGE (<Picture 2>) DESCRIPTION: {d2 or '(not described)'}",
            f"SCENARIO / WHAT HAPPENS BETWEEN THE TWO FRAMES: {self.scenario.get()}",
            f"VISUAL STYLE: {self.style.get()}",
            f"EXACT TARGET DURATION (use as S.SS in the alignment line): {self.duration.get() or self.duration_value()} seconds",
        ]
        if self.camera_notes.get():
            lines.append(f"CAMERA / TRANSITION NOTES: {self.camera_notes.get()}")
        lines.append("")
        lines.append(self.shared.to_brief())
        return "\n".join(lines)


class L2VATab(BaseModeTab):
    MODE = "L2VA"
    TITLE = "Last frame only → Video+Audio"
    INTRO = (
        "A single image anchors the very last frame of the final shot. The LLM invents "
        "a plausible starting state and converges the action toward this image."
    )

    def build_mode_fields(self):
        self.last_frame = ImageRefPicker(self, self.app, "Last-frame image (<Picture 1>)")
        self.last_frame.pack(fill="x", pady=(4, 10))

        self.duration = LabeledEntry(self, "Exact target duration (seconds, e.g. 6.00)", placeholder="6.00")
        self.duration.set(f"{float(self.duration_value()):.2f}")
        self.duration.pack(fill="x")

    def build_brief(self) -> str:
        d1 = self.last_frame.get_description()
        lines = [
            f"LAST-FRAME IMAGE (<Picture 1>) DESCRIPTION: {d1 or '(not described)'}",
            f"SCENARIO / WHAT SHOULD LEAD UP TO THIS FINAL IMAGE: {self.scenario.get()}",
            f"VISUAL STYLE: {self.style.get()}",
            f"EXACT TARGET DURATION (use as S.SS in the alignment line): {self.duration.get() or self.duration_value()} seconds",
        ]
        if self.camera_notes.get():
            lines.append(f"CAMERA / CONVERGENCE NOTES: {self.camera_notes.get()}")
        lines.append("")
        lines.append(self.shared.to_brief())
        return "\n".join(lines)


# ------------------------------------------------------------------------- #
# Ref2VA : références multiples
# ------------------------------------------------------------------------- #
class AddReferenceDialog(ctk.CTkToplevel):
    def __init__(self, master, app: "H3StudioApp", on_add):
        super().__init__(master)
        self.title("Add a reference")
        self.geometry("520x480")
        self.app = app
        self.on_add = on_add
        self.grab_set()

        self.ref_type = LabeledCombo(self, "Reference type", ["Picture (image)", "Video", "Audio"])
        self.ref_type.pack(fill="x", padx=16, pady=(16, 4))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=16)
        self.path_entry = ctk.CTkEntry(row, placeholder_text="file path (optional)")
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(row, text="Browse…", width=100, command=self._browse).pack(side="left")

        self.role = LabeledEntry(
            self, "Role of this reference",
            placeholder="e.g. first-frame anchor, character style, voice timbre…",
        )
        self.role.pack(fill="x", padx=16, pady=(8, 0))

        self.description = LabeledText(
            self, "Detailed description (subject, appearance, content…)", height=140,
        )
        self.description.pack(fill="x", padx=16, pady=(8, 0))

        vision_btn = ctk.CTkButton(self, text="Describe the image with the vision model", command=self._describe)
        vision_btn.pack(fill="x", padx=16, pady=(8, 0))

        ctk.CTkButton(self, text="Add the reference", height=38, command=self._confirm).pack(
            fill="x", padx=16, pady=16
        )

    def _browse(self):
        path = filedialog.askopenfilename(title="Choose a reference file")
        if path:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)

    def _describe(self):
        ref_type = self.ref_type.get().split(" ")[0]  # "Picture"/"Video"/"Audio"
        if ref_type != "Picture":
            messagebox.showwarning(
                "Not an image",
                "Vision description only works for a Picture reference (not Video/Audio).",
            )
            return
        path = self.path_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Missing file", "Please select a valid image first.")
            return
        vision_model = self.app.settings.get("vision_model", "").strip()
        if not vision_model:
            messagebox.showwarning("Vision model not configured", "Set it in the Settings tab.")
            return

        def work():
            try:
                desc = self.app.client.describe_image(vision_model, path)
            except LLMError as e:
                err_msg = str(e)
                self.after(0, lambda err_msg=err_msg: messagebox.showerror("Vision error", err_msg))
                return
            self.after(0, lambda: self.description.set(desc))

        threading.Thread(target=work, daemon=True).start()

    def _confirm(self):
        ref_type = self.ref_type.get().split(" ")[0]  # "Picture"/"Video"/"Audio"
        role = self.role.get()
        desc = self.description.get()
        path = self.path_entry.get().strip()
        if not role and not desc:
            messagebox.showwarning("Incomplete reference", "Please provide at least a role or a description.")
            return
        self.on_add({"type": ref_type, "path": path, "role": role, "description": desc})
        self.destroy()


class SceneDialogueFields(ctk.CTkFrame):
    """Sous-ensemble 'Dialogue / voice' de SharedFields, dupliqué par scène
    (chaque scène du storyboard a son propre dialogue/voix/texte à l'écran)."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")

        self.has_dialogue = ctk.CTkCheckBox(self, text="There is dialogue / a voice in this scene")
        self.has_dialogue.pack(anchor="w", pady=(0, 4))

        self.speaker_desc = LabeledEntry(
            self, "Speaker description", placeholder="e.g. young woman, soft and slightly raspy voice",
        )
        self.speaker_desc.pack(fill="x")

        self.language = LabeledCombo(self, "Dialogue language", LANGUAGES, default="French")
        self.language.pack(fill="x")

        self.dialogue_text = LabeledText(
            self, "Dialogue text (exact, will be preserved word for word)", height=60,
            placeholder="e.g. I'm getting off at the next station.",
        )
        self.dialogue_text.pack(fill="x")

        self.voiceover = ctk.CTkCheckBox(self, text="This is a voiceover, not an on-screen character speaking")
        self.voiceover.pack(anchor="w", pady=(0, 6))

        self.onscreen_text = LabeledEntry(
            self, "On-screen text (sign, screen, subtitle…)",
            placeholder='e.g. a red neon sign reading "OPEN"',
        )
        self.onscreen_text.pack(fill="x")

    def to_brief(self) -> str:
        lines = []
        if self.has_dialogue.get():
            lines.append("DIALOGUE: yes")
            if self.speaker_desc.get():
                lines.append(f"  Speaker description: {self.speaker_desc.get()}")
            lines.append(f"  Language: {self.language.get() or 'English'}")
            if self.dialogue_text.get():
                lines.append(f"  Exact line(s) to preserve verbatim: {self.dialogue_text.get()}")
            if self.voiceover.get():
                lines.append("  This is an off-screen voiceover, not on-screen speech - remember to state the speaking character's lips stay closed.")
        else:
            lines.append("DIALOGUE: none - do not invent spoken dialogue.")

        if self.onscreen_text.get():
            lines.append(f"ON-SCREEN TEXT: {self.onscreen_text.get()}")

        return "\n".join(lines)


class StoryboardScenePanel(ctk.CTkFrame):
    """Panneau d'une scène en mode storyboard Ref2VA : brief court à
    développer par le LLM + sélection des références à privilégier."""

    def __init__(self, master, index: int):
        super().__init__(master, corner_radius=8, fg_color=("gray92", "gray17"))
        self.index = index
        ctk.CTkLabel(self, text=f"Scene {index}", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=10, pady=(8, 2)
        )

        self.brief = LabeledText(
            self, "Scene description (brief — the LLM will develop it into a full prompt)",
            height=60,
            placeholder="e.g. the character from <Picture 1> enters the room and sits down",
        )
        self.brief.pack(fill="x", padx=10)

        ctk.CTkLabel(self, text="Dialogue / voice", text_color="gray60").pack(
            anchor="w", padx=10, pady=(4, 2)
        )
        self.dialogue = SceneDialogueFields(self)
        self.dialogue.pack(fill="x", padx=10)

        ctk.CTkLabel(self, text="References used in this scene", text_color="gray60").pack(
            anchor="w", padx=10, pady=(4, 2)
        )
        self.ref_checks_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ref_checks_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._checkboxes: dict[str, ctk.CTkCheckBox] = {}

    def refresh_refs(self, refs: list[dict]):
        """refs: [{'tag': '<Picture 1>', 'label': '<Picture 1> - role'}]. Rebuilds
        checkboxes, keeping the checked state of tags that still exist."""
        previous = {tag: cb.get() for tag, cb in self._checkboxes.items()}
        for w in self.ref_checks_frame.winfo_children():
            w.destroy()
        self._checkboxes = {}
        if not refs:
            ctk.CTkLabel(
                self.ref_checks_frame, text="(no reference defined yet)", text_color="gray50"
            ).pack(anchor="w")
            return
        for r in refs:
            cb = ctk.CTkCheckBox(self.ref_checks_frame, text=r["label"])
            if previous.get(r["tag"]):
                cb.select()
            cb.pack(anchor="w", pady=1)
            self._checkboxes[r["tag"]] = cb

    def selected_tags(self) -> list[str]:
        return [tag for tag, cb in self._checkboxes.items() if cb.get()]

    def get_brief(self) -> str:
        return self.brief.get()


class Ref2VATab(ctk.CTkScrollableFrame):
    MODE = "Ref2VA"

    def __init__(self, master, app: "H3StudioApp"):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.references: list[dict] = []

        ctk.CTkLabel(
            self,
            text=(
                "Full multi-reference mode (up to 9 images / 3 videos / 3 audio, "
                "12 total). Ideal for composing a scene from several reference "
                "characters, settings, or audio tracks."
            ),
            anchor="w", justify="left", wraplength=680, text_color="gray70",
        ).pack(fill="x", pady=(0, 10))

        self.scenario = LabeledText(
            self, "Target scenario (what the final video should show)", height=110,
            placeholder="e.g. the character from image 2 eats a cookie in the setting of image 1…",
        )
        self.scenario.pack(fill="x")

        self.style = LabeledCombo(self, "Visual style", STYLE_OPTIONS, default="Cinematic")
        self.style.pack(fill="x")

        self.duration = LabeledEntry(self, "Target duration (seconds, ~10 by default)", placeholder="10")
        self.duration.set(str(app.settings.get("default_duration", 10)))
        self.duration.pack(fill="x")

        ctk.CTkLabel(self, text="References", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(10, 4)
        )
        self.ref_count_label = ctk.CTkLabel(self, text="", text_color="gray60")
        self.ref_count_label.pack(anchor="w")

        self.ref_list_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ref_list_frame.pack(fill="x", pady=(4, 6))

        ctk.CTkButton(self, text="➕ Add a reference", command=self.add_reference_dialog).pack(
            fill="x", pady=(0, 12)
        )

        ctk.CTkLabel(self, text="Storyboard mode", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(6, 2)
        )
        self.storyboard_enabled = ctk.CTkCheckBox(
            self,
            text="Generate several sequential scenes instead of a single prompt",
            command=self._toggle_storyboard,
        )
        self.storyboard_enabled.pack(anchor="w", pady=(0, 6))

        sb_row = ctk.CTkFrame(self, fg_color="transparent")
        sb_row.pack(fill="x", pady=(0, 6))
        self.scene_count_entry = ctk.CTkEntry(sb_row, placeholder_text="Number of scenes", width=140)
        self.scene_count_entry.insert(0, "3")
        self.scene_count_entry.pack(side="left", padx=(0, 8))
        self.build_scenes_btn = ctk.CTkButton(
            sb_row, text="Build scenes", width=140, command=self._build_scenes, state="disabled"
        )
        self.build_scenes_btn.pack(side="left")

        self.scenes_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.scenes_frame.pack(fill="x", pady=(0, 6))
        self.scene_panels: list[StoryboardScenePanel] = []

        self.camera_notes = LabeledText(
            self, "Camera / shot notes (optional)", height=60,
        )
        self.camera_notes.pack(fill="x")

        self.shared = SharedFields(self)
        self.shared.pack(fill="x", pady=(6, 0))

        ctk.CTkButton(
            self, text="🎬 Generate H3 prompt (Ref2VA)", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.on_generate,
        ).pack(fill="x", pady=(16, 20))

        self._refresh_ref_list()

    def add_reference_dialog(self):
        counts = self._counts()
        if counts["total"] >= MAX_REFS:
            messagebox.showwarning("Limit reached", f"Maximum {MAX_REFS} references in total.")
            return
        AddReferenceDialog(self, self.app, self._add_reference)

    def _counts(self):
        c = {"Picture": 0, "Video": 0, "Audio": 0}
        for r in self.references:
            c[r["type"]] = c.get(r["type"], 0) + 1
        c["total"] = len(self.references)
        return c

    def _add_reference(self, ref: dict):
        counts = self._counts()
        limits = {"Picture": MAX_IMAGES, "Video": MAX_VIDEOS, "Audio": MAX_AUDIO}
        if counts.get(ref["type"], 0) >= limits[ref["type"]]:
            messagebox.showwarning(
                "Limit reached", f"Maximum {limits[ref['type']]} references of type {ref['type']}."
            )
            return
        self.references.append(ref)
        self._refresh_ref_list()

    def _remove_reference(self, idx: int):
        del self.references[idx]
        self._refresh_ref_list()

    def _refresh_ref_list(self):
        for w in self.ref_list_frame.winfo_children():
            w.destroy()

        label_prefix = {"Picture": "Picture", "Video": "Video", "Audio": "Audio"}
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        for i, ref in enumerate(self.references):
            counters[ref["type"]] += 1
            tag = f"<{label_prefix[ref['type']]} {counters[ref['type']]}>"

            row = ctk.CTkFrame(self.ref_list_frame, corner_radius=6)
            row.pack(fill="x", pady=3)
            text = f"{tag}  —  {ref['role'] or '(no role specified)'}"
            ctk.CTkLabel(row, text=text, anchor="w", justify="left", wraplength=520).pack(
                side="left", fill="x", expand=True, padx=8, pady=6
            )
            ctk.CTkButton(
                row, text="✕", width=28, fg_color="transparent", hover_color="#7a2020",
                command=lambda idx=i: self._remove_reference(idx),
            ).pack(side="right", padx=6)

        counts = self._counts()
        self.ref_count_label.configure(
            text=(
                f"{counts['total']}/{MAX_REFS} total references — "
                f"images {counts['Picture']}/{MAX_IMAGES}, "
                f"videos {counts['Video']}/{MAX_VIDEOS}, "
                f"audio {counts['Audio']}/{MAX_AUDIO}"
            )
        )

        if self.scene_panels:
            refs = self._current_ref_options()
            for panel in self.scene_panels:
                panel.refresh_refs(refs)

    # -- Storyboard mode --------------------------------------------------- #
    def _toggle_storyboard(self):
        enabled = bool(self.storyboard_enabled.get())
        self.build_scenes_btn.configure(state="normal" if enabled else "disabled")

    def _current_ref_options(self) -> list[dict]:
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        options = []
        for ref in self.references:
            counters[ref["type"]] += 1
            tag = f"<{ref['type']} {counters[ref['type']]}>"
            options.append({"tag": tag, "label": f"{tag} — {ref['role'] or '(no role)'}"})
        return options

    def _build_scenes(self):
        try:
            n = int(self.scene_count_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid number", "Enter a whole number of scenes.")
            return
        if n < 1 or n > 30:
            messagebox.showwarning("Invalid number", "Choose between 1 and 30 scenes.")
            return
        for w in self.scenes_frame.winfo_children():
            w.destroy()
        self.scene_panels = []
        refs = self._current_ref_options()
        for i in range(1, n + 1):
            panel = StoryboardScenePanel(self.scenes_frame, i)
            panel.pack(fill="x", pady=4)
            panel.refresh_refs(refs)
            self.scene_panels.append(panel)

    def _reference_library_block(self) -> str:
        lines = ["REFERENCE ASSETS (full shared library — assign labels in this order, per category):"]
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        if not self.references:
            lines.append("  (none provided - rely only on the scene description)")
        for ref in self.references:
            counters[ref["type"]] += 1
            tag = f"<{ref['type']} {counters[ref['type']]}>"
            path_info = f" [file: {ref['path']}]" if ref["path"] else ""
            lines.append(f"  {tag}{path_info}")
            if ref["role"]:
                lines.append(f"    role: {ref['role']}")
            if ref["description"]:
                lines.append(f"    description: {ref['description']}")
        return "\n".join(lines)

    def build_brief(self) -> str:
        lines = [
            f"TARGET SCENARIO: {self.scenario.get()}",
            f"VISUAL STYLE: {self.style.get()}",
            f"TARGET DURATION: {self.duration.get() or self.app.settings.get('default_duration', 10)} seconds",
        ]
        if self.camera_notes.get():
            lines.append(f"CAMERA NOTES: {self.camera_notes.get()}")

        lines.append("")
        lines.append("REFERENCE ASSETS (assign labels in this order, per category):")
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        if not self.references:
            lines.append("  (none provided - rely only on the scenario text)")
        for ref in self.references:
            counters[ref["type"]] += 1
            tag = f"<{ref['type']} {counters[ref['type']]}>"
            path_info = f" [file: {ref['path']}]" if ref["path"] else ""
            lines.append(f"  {tag}{path_info}")
            if ref["role"]:
                lines.append(f"    role: {ref['role']}")
            if ref["description"]:
                lines.append(f"    description: {ref['description']}")

        lines.append("")
        lines.append(self.shared.to_brief())
        return "\n".join(lines)

    def on_generate(self):
        if self.storyboard_enabled.get():
            self._on_generate_storyboard()
            return
        if not self.scenario.get():
            messagebox.showwarning("Missing scenario", "Please describe the target scenario, at least briefly.")
            return
        brief = self.build_brief()
        self.app.run_generation(REF2VA_SYSTEM_PROMPT, brief, mode_label="Ref2VA")

    def _on_generate_storyboard(self):
        if not self.scene_panels:
            messagebox.showwarning("No scenes", "Set the number of scenes and click 'Build scenes' first.")
            return
        missing = [p.index for p in self.scene_panels if not p.get_brief()]
        if missing:
            messagebox.showwarning(
                "Missing scene description",
                f"Scene(s) {', '.join(str(i) for i in missing)} have no description.",
            )
            return

        ref_library = self._reference_library_block()
        n = len(self.scene_panels)
        briefs = []
        for panel in self.scene_panels:
            tags = panel.selected_tags()
            lines = [
                f"STORYBOARD SCENE {panel.index}/{n} — treat as its own standalone H3 Ref2VA prompt "
                "(consistent with the same reference library across all scenes).",
                f"OVERALL STORYBOARD CONCEPT (context only): {self.scenario.get() or '(none)'}",
                f"BRIEF FOR THIS SCENE (develop into the full prompt): {panel.get_brief()}",
                f"VISUAL STYLE: {self.style.get()}",
                f"TARGET DURATION: {self.duration.get() or self.app.settings.get('default_duration', 10)} seconds",
            ]
            if self.camera_notes.get():
                lines.append(f"CAMERA NOTES: {self.camera_notes.get()}")
            lines.append("")
            lines.append(ref_library)
            lines.append(
                f"PRIMARY REFERENCES FOR THIS SCENE: "
                f"{', '.join(tags) if tags else '(none selected - use scene description only)'}"
            )
            lines.append("")
            lines.append(panel.dialogue.to_brief())
            lines.append(self.shared.to_brief(include_dialogue=False))
            briefs.append((f"Scene {panel.index}", "\n".join(lines)))

        self.app.run_generation_sequence(REF2VA_SYSTEM_PROMPT, briefs, mode_label="Ref2VA storyboard")


# ------------------------------------------------------------------------- #
# Story -> Sequences : construit l'histoire depuis les images de référence,
# la découpe en N séquences, chaque séquence recevant du LLM ses propres
# références / dialogue / mouvement de caméra, puis génère un prompt H3
# Ref2VA par séquence (réutilise run_generation_sequence).
# ------------------------------------------------------------------------- #
class SequenceCard(ctk.CTkFrame):
    """Affichage en lecture seule d'une séquence décidée par le LLM."""

    def __init__(self, master, seq: dict):
        super().__init__(master, corner_radius=8, fg_color=("gray92", "gray17"))
        title = f"Scene {seq.get('index', '?')} — {seq.get('title', '')}"
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=10, pady=(8, 2)
        )
        ctk.CTkLabel(
            self, text=seq.get("brief", ""), anchor="w", justify="left",
            wraplength=620, text_color="gray80",
        ).pack(anchor="w", padx=10, pady=(0, 4))

        refs = ", ".join(seq.get("references") or []) or "(none)"
        dlg = seq.get("dialogue", {}) or {}
        if dlg.get("present"):
            dlg_text = f"\"{dlg.get('text', '')}\" — {dlg.get('speaker', '')} ({dlg.get('language', '')})"
            if dlg.get("voiceover"):
                dlg_text += " [voiceover]"
        else:
            dlg_text = "none"

        for label, value in (
            ("References", refs),
            ("Dialogue", dlg_text),
            ("Camera", seq.get("camera_movement", "")),
        ):
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=10)
            ctk.CTkLabel(row, text=f"{label}: ", text_color="gray60", width=90, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, anchor="w", justify="left", wraplength=520).pack(
                side="left", fill="x", expand=True
            )
        ctk.CTkFrame(self, fg_color="transparent", height=6).pack()


class SequenceTab(ctk.CTkScrollableFrame):
    """Story -> Sequences : réutilise la bibliothèque de références de
    Ref2VA (AddReferenceDialog) et enchaîne 2 passes LLM (histoire, puis
    découpage en séquences) avant de générer les prompts H3 finaux."""

    MODE = "Story→Sequences"

    def __init__(self, master, app: "H3StudioApp"):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.references: list[dict] = []
        self.sequences: list[dict] = []
        self.story_text: str = ""

        ctk.CTkLabel(
            self,
            text=(
                "Build a short story from your reference images, split it into N "
                "successive sequences, then generate one H3 Ref2VA prompt per "
                "sequence — the LLM picks the references, dialogue, and camera "
                "movement for each sequence itself."
            ),
            anchor="w", justify="left", wraplength=680, text_color="gray70",
        ).pack(fill="x", pady=(0, 10))

        # -- 1. Reference library ------------------------------------------------
        ctk.CTkLabel(self, text="1. References", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(6, 4)
        )
        self.ref_count_label = ctk.CTkLabel(self, text="", text_color="gray60")
        self.ref_count_label.pack(anchor="w")
        self.ref_list_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ref_list_frame.pack(fill="x", pady=(4, 6))
        ctk.CTkButton(self, text="➕ Add a reference", command=self.add_reference_dialog).pack(
            fill="x", pady=(0, 12)
        )

        # -- 2. Story --------------------------------------------------------
        ctk.CTkLabel(self, text="2. Story", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(6, 4))
        self.premise = LabeledText(
            self, "Premise / notes (optional — the LLM invents freely if left blank)", height=70,
        )
        self.premise.pack(fill="x")
        self.story_language = LabeledCombo(self, "Story language", LANGUAGES, default="English")
        self.story_language.pack(fill="x")
        ctk.CTkButton(self, text="📝 Generate story from references", command=self.on_generate_story).pack(
            fill="x", pady=(4, 6)
        )
        self.story_box = ctk.CTkTextbox(self, height=160, wrap="word")
        self.story_box.pack(fill="x", pady=(0, 12))

        # -- 3. Sequencing -----------------------------------------------------
        ctk.CTkLabel(self, text="3. Split into sequences", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(6, 4)
        )
        seq_row = ctk.CTkFrame(self, fg_color="transparent")
        seq_row.pack(fill="x", pady=(0, 6))
        self.n_sequences_entry = ctk.CTkEntry(seq_row, placeholder_text="Number of sequences", width=160)
        self.n_sequences_entry.insert(0, "5")
        self.n_sequences_entry.pack(side="left", padx=(0, 8))
        self.seq_duration_entry = ctk.CTkEntry(seq_row, placeholder_text="Seconds / sequence", width=160)
        self.seq_duration_entry.insert(0, str(app.settings.get("default_duration", 10)))
        self.seq_duration_entry.pack(side="left")

        self.style = LabeledCombo(self, "Visual style", STYLE_OPTIONS, default="Cinematic")
        self.style.pack(fill="x")

        self.extra_instructions = LabeledText(
            self, "Extra instructions (optional — applied to every sequence)", height=60,
            placeholder="e.g. always keep the camera at eye level; avoid text overlays…",
        )
        self.extra_instructions.pack(fill="x")

        ctk.CTkButton(self, text="✂ Break story into sequences", command=self.on_breakdown).pack(
            fill="x", pady=(4, 6)
        )

        self.sequences_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.sequences_frame.pack(fill="x", pady=(0, 6))

        ctk.CTkButton(
            self, text="🎬 Generate H3 prompts for all sequences", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.on_generate_prompts,
        ).pack(fill="x", pady=(16, 6))

        save_row = ctk.CTkFrame(self, fg_color="transparent")
        save_row.pack(fill="x", pady=(0, 20))
        ctk.CTkButton(save_row, text="💾 Sauvegarder", command=self.on_save).pack(
            side="left", expand=True, fill="x", padx=(0, 6)
        )
        ctk.CTkButton(save_row, text="📂 Load", command=self.on_load).pack(
            side="left", expand=True, fill="x", padx=(6, 0)
        )

        self._refresh_ref_list()

    # -- reference library (same shape/behaviour as Ref2VATab) -------------- #
    def add_reference_dialog(self):
        counts = self._counts()
        if counts["total"] >= MAX_REFS:
            messagebox.showwarning("Limit reached", f"Maximum {MAX_REFS} references in total.")
            return
        AddReferenceDialog(self, self.app, self._add_reference)

    def _counts(self):
        c = {"Picture": 0, "Video": 0, "Audio": 0}
        for r in self.references:
            c[r["type"]] = c.get(r["type"], 0) + 1
        c["total"] = len(self.references)
        return c

    def _add_reference(self, ref: dict):
        counts = self._counts()
        limits = {"Picture": MAX_IMAGES, "Video": MAX_VIDEOS, "Audio": MAX_AUDIO}
        if counts.get(ref["type"], 0) >= limits[ref["type"]]:
            messagebox.showwarning(
                "Limit reached", f"Maximum {limits[ref['type']]} references of type {ref['type']}."
            )
            return
        self.references.append(ref)
        self._refresh_ref_list()

    def _remove_reference(self, idx: int):
        del self.references[idx]
        self._refresh_ref_list()

    def _refresh_ref_list(self):
        for w in self.ref_list_frame.winfo_children():
            w.destroy()
        label_prefix = {"Picture": "Picture", "Video": "Video", "Audio": "Audio"}
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        for i, ref in enumerate(self.references):
            counters[ref["type"]] += 1
            tag = f"<{label_prefix[ref['type']]} {counters[ref['type']]}>"
            row = ctk.CTkFrame(self.ref_list_frame, corner_radius=6)
            row.pack(fill="x", pady=3)
            text = f"{tag}  —  {ref['role'] or '(no role specified)'}"
            ctk.CTkLabel(row, text=text, anchor="w", justify="left", wraplength=520).pack(
                side="left", fill="x", expand=True, padx=8, pady=6
            )
            ctk.CTkButton(
                row, text="✕", width=28, fg_color="transparent", hover_color="#7a2020",
                command=lambda idx=i: self._remove_reference(idx),
            ).pack(side="right", padx=6)
        counts = self._counts()
        self.ref_count_label.configure(
            text=(
                f"{counts['total']}/{MAX_REFS} total references — "
                f"images {counts['Picture']}/{MAX_IMAGES}, "
                f"videos {counts['Video']}/{MAX_VIDEOS}, "
                f"audio {counts['Audio']}/{MAX_AUDIO}"
            )
        )

    def _reference_library_block(self) -> str:
        lines = ["REFERENCE ASSETS (full shared library — assign labels in this order, per category):"]
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        if not self.references:
            lines.append("  (none provided - rely only on the scene description)")
        for ref in self.references:
            counters[ref["type"]] += 1
            tag = f"<{ref['type']} {counters[ref['type']]}>"
            path_info = f" [file: {ref['path']}]" if ref["path"] else ""
            lines.append(f"  {tag}{path_info}")
            if ref["role"]:
                lines.append(f"    role: {ref['role']}")
            if ref["description"]:
                lines.append(f"    description: {ref['description']}")
        return "\n".join(lines)

    # -- model helper --------------------------------------------------------- #
    def _require_model(self) -> str | None:
        model = self.app.settings.get("chat_model", "").strip()
        if not model or model == NO_CHAT_MODEL_MARKER or model.startswith("("):
            messagebox.showwarning(
                "Model not configured",
                "Choose a chat model in the Settings tab (and click "
                "'Test connection / refresh models').",
            )
            return None
        return model

    # -- 2. story ------------------------------------------------------------- #
    def on_generate_story(self):
        if not self.references:
            messagebox.showwarning("No references", "Add at least one reference image first.")
            return
        model = self._require_model()
        if not model:
            return

        self.story_box.delete("1.0", "end")
        self.app.status.configure(text="Generating story…", text_color="#e0b055")

        def on_token(piece: str):
            def apply():
                self.story_box.insert("end", piece)
                self.story_box.see("end")
            self.app.safe_after(apply)

        def work():
            try:
                text = generate_story(
                    self.app.client, model, self.references,
                    self.premise.get(), self.story_language.get(),
                    on_token=on_token,
                )
                self.story_text = text
                self.app.safe_after(lambda: self.app.status.configure(text="Story ready ✓", text_color="#55c07a"))
            except LLMError as e:
                err_msg = str(e)
                self.app.safe_after(lambda err_msg=err_msg: messagebox.showerror("Story generation error", err_msg))
                self.app.safe_after(lambda: self.app.status.configure(text="Error", text_color="#e05555"))

        threading.Thread(target=work, daemon=True).start()

    # -- 3. breakdown ----------------------------------------------------------- #
    def _n_sequences(self) -> int | None:
        try:
            n = int(self.n_sequences_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid number", "Enter a whole number of sequences.")
            return None
        if n < 1 or n > 30:
            messagebox.showwarning("Invalid number", "Choose between 1 and 30 sequences.")
            return None
        return n

    def _duration_per_sequence(self) -> int:
        try:
            return int(self.seq_duration_entry.get().strip())
        except ValueError:
            return int(self.app.settings.get("default_duration", 10))

    def on_breakdown(self):
        story_text = self.story_box.get("1.0", "end-1c").strip()
        if not story_text:
            messagebox.showwarning("No story", "Generate (or write) a story first.")
            return
        n = self._n_sequences()
        if n is None:
            return
        model = self._require_model()
        if not model:
            return

        self.story_text = story_text
        for w in self.sequences_frame.winfo_children():
            w.destroy()
        self.sequences = []
        self.app.status.configure(text=f"Breaking story into {n} sequences…", text_color="#e0b055")

        def work():
            try:
                seqs = generate_sequence_breakdown(
                    self.app.client, model, story_text, self.references, n,
                    CAMERA_MOTIONS, self.extra_instructions.get(),
                    duration_per_sequence=self._duration_per_sequence(),
                )
            except (LLMError, ValueError, Exception) as e:  # noqa: BLE001 - surface any parsing failure
                err_msg = str(e)
                self.app.safe_after(lambda err_msg=err_msg: messagebox.showerror("Breakdown error", err_msg))
                self.app.safe_after(lambda: self.app.status.configure(text="Error", text_color="#e05555"))
                return

            def apply():
                self.sequences = seqs
                for seq in seqs:
                    SequenceCard(self.sequences_frame, seq).pack(fill="x", pady=4)
                self.app.status.configure(text=f"{len(seqs)} sequences ready ✓", text_color="#55c07a")
            self.app.safe_after(apply)

        threading.Thread(target=work, daemon=True).start()

    # -- 4. final H3 prompts -------------------------------------------------- #
    def on_generate_prompts(self):
        if not self.sequences:
            messagebox.showwarning("No sequences", "Break the story into sequences first.")
            return
        ref_block = self._reference_library_block()
        n_total = len(self.sequences)
        briefs = [
            (
                f"Scene {seq['index']}",
                sequence_to_brief(
                    seq, n_total, ref_block, self.story_text,
                    self._duration_per_sequence(), self.style.get(),
                    self.extra_instructions.get(),
                ),
            )
            for seq in self.sequences
        ]
        self.app.run_generation_sequence(REF2VA_SYSTEM_PROMPT, briefs, mode_label="Story sequences")

    # -- save / load (settings + story + sequences + prompts) ---------------- #
    def _to_dict(self) -> dict:
        return {
            "references": self.references,
            "premise": self.premise.get(),
            "story_language": self.story_language.get(),
            "story_text": self.story_box.get("1.0", "end-1c"),
            "n_sequences": self.n_sequences_entry.get().strip(),
            "seq_duration": self.seq_duration_entry.get().strip(),
            "style": self.style.get(),
            "extra_instructions": self.extra_instructions.get(),
            "sequences": self.sequences,
        }

    def on_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="story_sequences.json",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:
            messagebox.showerror("Save error", str(e))
            return
        self.app.status.configure(text=f"Saved: {os.path.basename(path)}", text_color="#55c07a")

    def on_load(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Load error", str(e))
            return

        self.references = data.get("references", [])
        self._refresh_ref_list()

        self.premise.set(data.get("premise", ""))
        if data.get("story_language"):
            self.story_language.combo.set(data["story_language"])

        self.story_text = data.get("story_text", "")
        self.story_box.delete("1.0", "end")
        self.story_box.insert("1.0", self.story_text)

        self.n_sequences_entry.delete(0, "end")
        self.n_sequences_entry.insert(0, data.get("n_sequences") or "5")
        self.seq_duration_entry.delete(0, "end")
        self.seq_duration_entry.insert(
            0, data.get("seq_duration") or str(self.app.settings.get("default_duration", 10))
        )

        if data.get("style"):
            self.style.combo.set(data["style"])
        self.extra_instructions.set(data.get("extra_instructions", ""))

        self.sequences = data.get("sequences", [])
        for w in self.sequences_frame.winfo_children():
            w.destroy()
        for seq in self.sequences:
            SequenceCard(self.sequences_frame, seq).pack(fill="x", pady=4)

        self.app.status.configure(text=f"Loaded: {os.path.basename(path)}", text_color="#55c07a")


# ------------------------------------------------------------------------- #
# Onglet Réglages
# ------------------------------------------------------------------------- #
class SettingsTab(ctk.CTkScrollableFrame):
    def __init__(self, master, app: "H3StudioApp"):
        super().__init__(master, fg_color="transparent")
        self.app = app

        ctk.CTkLabel(self, text="LLM backend", anchor="w").pack(fill="x")
        self.backend = ctk.CTkSegmentedButton(
            self, values=["ollama", "lmstudio", "llamacpp"], command=self._on_backend_change
        )
        self.backend.set(app.settings.get("llm_backend", "ollama"))
        self.backend.pack(fill="x", pady=(2, 10))

        self.url = LabeledEntry(self, "Ollama server URL", placeholder="http://localhost:11434")
        self.url.set(app.settings["ollama_url"])
        self.url.pack(fill="x")

        self.lmstudio_url = LabeledEntry(self, "LM Studio server URL", placeholder="http://localhost:1234")
        self.lmstudio_url.set(app.settings.get("lmstudio_url", DEFAULT_SETTINGS["lmstudio_url"]))
        self.lmstudio_url.pack(fill="x")

        self.llamacpp_url = LabeledEntry(self, "llama.cpp server URL", placeholder="http://localhost:8080")
        self.llamacpp_url.set(app.settings.get("llamacpp_url", DEFAULT_SETTINGS["llamacpp_url"]))
        self.llamacpp_url.pack(fill="x")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))
        ctk.CTkButton(row, text="Test connection / refresh models", command=self.refresh_models).pack(
            side="left"
        )
        self.status_label = ctk.CTkLabel(row, text="", text_color="gray60")
        self.status_label.pack(side="left", padx=10)

        self.chat_model = LabeledCombo(self, "Chat model (text)", [app.settings["chat_model"]] if app.settings["chat_model"] else [NO_CHAT_MODEL_MARKER])
        self.chat_model.pack(fill="x")

        self.vision_model = LabeledCombo(
            self, "Vision model (to describe reference images, optional)",
            [app.settings["vision_model"]] if app.settings["vision_model"] else [NO_VISION_MODEL_MARKER],
        )
        self.vision_model.pack(fill="x")

        self.temperature = LabeledEntry(self, "Temperature", placeholder="0.6")
        self.temperature.set(str(app.settings["temperature"]))
        self.temperature.pack(fill="x")

        self.default_duration = LabeledEntry(self, "Default duration (seconds)", placeholder="10")
        self.default_duration.set(str(app.settings["default_duration"]))
        self.default_duration.pack(fill="x")

        ctk.CTkButton(self, text="💾 Save settings", height=38, command=self.save).pack(
            fill="x", pady=16
        )

        ctk.CTkLabel(
            self,
            text=(
                "Tip: a strong chat model (14B+) follows the structured H3 format "
                "much better than a small model. A vision model (qwen2.5vl, llava, etc.) "
                "lets you automatically describe first/last-frame images."
            ),
            wraplength=680, justify="left", text_color="gray60",
        ).pack(fill="x", pady=(0, 10))

        self.after(200, self.refresh_models)

    def _on_backend_change(self, value: str):
        self.app.settings["llm_backend"] = value
        self.app.client.backend = value
        self.status_label.configure(text="", text_color="gray60")

    def refresh_models(self):
        self.app.settings["llm_backend"] = self.backend.get() or "ollama"
        self.app.settings["ollama_url"] = self.url.get() or DEFAULT_SETTINGS["ollama_url"]
        self.app.settings["lmstudio_url"] = self.lmstudio_url.get() or DEFAULT_SETTINGS["lmstudio_url"]
        self.app.settings["llamacpp_url"] = self.llamacpp_url.get() or DEFAULT_SETTINGS["llamacpp_url"]
        self.app.client.backend = self.app.settings["llm_backend"]
        self.app.client.ollama_url = self.app.settings["ollama_url"]
        self.app.client.lmstudio_url = self.app.settings["lmstudio_url"]
        self.app.client.llamacpp_url = self.app.settings["llamacpp_url"]

        def work():
            try:
                models = self.app.client.list_models()
                no_model_hints = {
                    "lmstudio": "(no model found — load one in LM Studio)",
                    "llamacpp": "(no model found — check llama-server is running with a loaded model)",
                }
                no_model_hint = no_model_hints.get(
                    self.app.client.backend, "(no model found — run `ollama pull ...`)"
                )
                names = [m.name for m in models] or [no_model_hint]
            except LLMError as e:
                err_msg = str(e)
                self.app.after(0, lambda err_msg=err_msg: self.status_label.configure(text=f"✕ {err_msg}", text_color="#e05555"))
                return

            def apply():
                self.chat_model.combo.configure(values=names)
                self.vision_model.combo.configure(values=[NO_VISION_MODEL_MARKER] + names)
                if self.app.settings.get("chat_model") in names:
                    self.chat_model.combo.set(self.app.settings["chat_model"])
                elif names:
                    self.chat_model.combo.set(names[0])
                if self.app.settings.get("vision_model") in names:
                    self.vision_model.combo.set(self.app.settings["vision_model"])
                self.status_label.configure(text=f"✓ {len(names)} model(s) found", text_color="#55c07a")

            self.app.after(0, apply)

        threading.Thread(target=work, daemon=True).start()

    def save(self):
        try:
            temp = float(self.temperature.get() or 0.6)
        except ValueError:
            temp = 0.6
        try:
            dur = int(float(self.default_duration.get() or 10))
        except ValueError:
            dur = 10

        self.app.settings.update(
            {
                "llm_backend": self.backend.get() or "ollama",
                "ollama_url": self.url.get() or DEFAULT_SETTINGS["ollama_url"],
                "lmstudio_url": self.lmstudio_url.get() or DEFAULT_SETTINGS["lmstudio_url"],
                "llamacpp_url": self.llamacpp_url.get() or DEFAULT_SETTINGS["llamacpp_url"],
                "chat_model": self.chat_model.get(),
                "vision_model": "" if self.vision_model.get() == NO_VISION_MODEL_MARKER else self.vision_model.get(),
                "temperature": temp,
                "default_duration": dur,
            }
        )
        self.app.client.backend = self.app.settings["llm_backend"]
        self.app.client.ollama_url = self.app.settings["ollama_url"]
        self.app.client.lmstudio_url = self.app.settings["lmstudio_url"]
        self.app.client.llamacpp_url = self.app.settings["llamacpp_url"]
        save_settings(self.app.settings)
        messagebox.showinfo("Settings", "Settings saved.")


# ------------------------------------------------------------------------- #
# Application principale
# ------------------------------------------------------------------------- #
class H3StudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("H3 Prompt Studio — MiniMax H3 (T2VA / I2VA / FL2VA / L2VA / Ref2VA)")
        self.geometry("1440x900")
        self.minsize(1100, 700)

        self.settings = load_settings()
        self.client = LLMClient(
            base_url=self.settings["ollama_url"],
            backend=self.settings.get("llm_backend", "ollama"),
            lmstudio_url=self.settings.get("lmstudio_url", DEFAULT_SETTINGS["lmstudio_url"]),
            llamacpp_url=self.settings.get("llamacpp_url", DEFAULT_SETTINGS["llamacpp_url"]),
        )

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- colonne gauche : formulaires par mode ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        self.tabview.add("T2VA")
        self.tabview.add("I2VA")
        self.tabview.add("FL2VA")
        self.tabview.add("L2VA")
        self.tabview.add("Ref2VA")
        self.tabview.add("📖 Story→Seq")
        self.tabview.add("⚙ Settings")

        self.tab_t2va = T2VATab(self.tabview.tab("T2VA"), self)
        self.tab_t2va.pack(fill="both", expand=True)
        self.tab_i2va = I2VATab(self.tabview.tab("I2VA"), self)
        self.tab_i2va.pack(fill="both", expand=True)
        self.tab_fl2va = FL2VATab(self.tabview.tab("FL2VA"), self)
        self.tab_fl2va.pack(fill="both", expand=True)
        self.tab_l2va = L2VATab(self.tabview.tab("L2VA"), self)
        self.tab_l2va.pack(fill="both", expand=True)
        self.tab_ref2va = Ref2VATab(self.tabview.tab("Ref2VA"), self)
        self.tab_ref2va.pack(fill="both", expand=True)
        self.tab_sequence = SequenceTab(self.tabview.tab("📖 Story→Seq"), self)
        self.tab_sequence.pack(fill="both", expand=True)
        self.tab_settings = SettingsTab(self.tabview.tab("⚙ Settings"), self)
        self.tab_settings.pack(fill="both", expand=True)

        # --- colonne droite : sortie ---
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(right, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        self.output_title = ctk.CTkLabel(
            header, text="Generated H3 prompt", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.output_title.pack(side="left")
        self.status = ctk.CTkLabel(header, text="", text_color="gray60")
        self.status.pack(side="right")

        self.output_box = ctk.CTkTextbox(right, wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
        self.output_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        ctk.CTkButton(btn_row, text="📋 Copy", command=self.copy_output).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="💾 Save .txt", command=self.save_output).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="🗑 Clear", fg_color="#7a2020", hover_color="#992828", command=self.clear_output).pack(
            side="left"
        )

        self._generating = False
        self._closing = threading.Event()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    def safe_after(self, fn):
        """Comme self.after(0, fn), mais n'agit plus une fois la fenêtre fermée
        (évite les erreurs Tk quand un thread de fond termine après coup)."""
        if not self._closing.is_set():
            self.after(0, fn)

    def _on_close(self):
        self._closing.set()
        save_settings(self.settings)
        self.destroy()

    def clear_output(self):
        self.output_box.delete("1.0", "end")

    def copy_output(self):
        text = self.output_box.get("1.0", "end-1c")
        if not text.strip():
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status.configure(text="Copied to clipboard ✓", text_color="#55c07a")

    def save_output(self):
        text = self.output_box.get("1.0", "end-1c")
        if not text.strip():
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")], initialfile="h3_prompt.txt"
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self.status.configure(text=f"Saved: {os.path.basename(path)}", text_color="#55c07a")

    # ------------------------------------------------------------------ #
    def run_generation(self, system_prompt: str, user_brief: str, mode_label: str):
        model = self.settings.get("chat_model", "").strip()
        if not model or model == NO_CHAT_MODEL_MARKER or model.startswith("("):
            messagebox.showwarning(
                "Model not configured",
                "Choose a chat model in the Settings tab (and click "
                "'Test connection / refresh models').",
            )
            return

        if self._generating:
            messagebox.showwarning(
                "Generation in progress",
                "Please wait for the current generation to finish before starting another one.",
            )
            return

        self._generating = True
        self.clear_output()
        self.status.configure(text=f"Generating ({mode_label}, {model})…", text_color="#e0b055")
        self.output_title.configure(text=f"H3 prompt — {mode_label}")

        def on_token(piece: str):
            def apply():
                self.output_box.insert("end", piece)
                self.output_box.see("end")
            self.safe_after(apply)

        def work():
            try:
                self.client.chat(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_brief,
                    temperature=self.settings.get("temperature", 0.6),
                    on_token=on_token,
                )
                self.safe_after(lambda: self.status.configure(text="Done ✓", text_color="#55c07a"))
            except LLMError as e:
                err_msg = str(e)
                self.safe_after(lambda: self.status.configure(text="Error", text_color="#e05555"))
                self.safe_after(lambda err_msg=err_msg: messagebox.showerror("Generation error", err_msg))
            except Exception as e:  # filet de sécurité : ne jamais laisser le thread mourir en silence
                err_msg = str(e)
                self.safe_after(lambda: self.status.configure(text="Error", text_color="#e05555"))
                self.safe_after(lambda err_msg=err_msg: messagebox.showerror("Unexpected error", err_msg))
            finally:
                def clear_flag():
                    self._generating = False
                self.safe_after(clear_flag)

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ #
    def run_generation_sequence(self, system_prompt: str, briefs: list[tuple[str, str]], mode_label: str):
        """Comme run_generation, mais enchaîne plusieurs (label, brief) dans
        le même thread — utilisé par le mode storyboard de Ref2VA."""
        model = self.settings.get("chat_model", "").strip()
        if not model or model == NO_CHAT_MODEL_MARKER or model.startswith("("):
            messagebox.showwarning(
                "Model not configured",
                "Choose a chat model in the Settings tab (and click "
                "'Test connection / refresh models').",
            )
            return

        if self._generating:
            messagebox.showwarning(
                "Generation in progress",
                "Please wait for the current generation to finish before starting another one.",
            )
            return

        self._generating = True
        self.clear_output()
        self.output_title.configure(text=f"H3 prompt — {mode_label}")

        def on_token(piece: str):
            def apply():
                self.output_box.insert("end", piece)
                self.output_box.see("end")
            self.safe_after(apply)

        def work():
            try:
                total = len(briefs)
                for i, (label, brief) in enumerate(briefs, start=1):
                    if self._closing.is_set():
                        return

                    def set_status(label=label, i=i, total=total):
                        self.status.configure(
                            text=f"Generating ({mode_label} {i}/{total} — {label})…",
                            text_color="#e0b055",
                        )
                    self.safe_after(set_status)

                    def insert_header(label=label, i=i):
                        prefix = "" if i == 1 else "\n\n"
                        self.output_box.insert("end", f"{prefix}{'=' * 10} {label} {'=' * 10}\n\n")
                        self.output_box.see("end")
                    self.safe_after(insert_header)

                    self.client.chat(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=brief,
                        temperature=self.settings.get("temperature", 0.6),
                        on_token=on_token,
                    )
                self.safe_after(lambda: self.status.configure(text="Done ✓", text_color="#55c07a"))
            except LLMError as e:
                err_msg = str(e)
                self.safe_after(lambda: self.status.configure(text="Error", text_color="#e05555"))
                self.safe_after(lambda err_msg=err_msg: messagebox.showerror("Generation error", err_msg))
            except Exception as e:  # filet de sécurité : ne jamais laisser le thread mourir en silence
                err_msg = str(e)
                self.safe_after(lambda: self.status.configure(text="Error", text_color="#e05555"))
                self.safe_after(lambda err_msg=err_msg: messagebox.showerror("Unexpected error", err_msg))
            finally:
                def clear_flag():
                    self._generating = False
                self.safe_after(clear_flag)

        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    app = H3StudioApp()
    app.mainloop()
