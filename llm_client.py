"""
llm_client.py
-----------------
Petite couche d'abstraction autour de l'API d'un backend LLM local.
Ne dépend que de `requests`. Pensée pour tourner 100% en local.

Trois backends supportés, réglables via `backend` ("ollama", "lmstudio" ou "llamacpp") :
  - Ollama (natif)                : /api/tags, /api/chat             (host par défaut http://localhost:11434)
  - LM Studio (compatible OpenAI) : /v1/models, /v1/chat/completions (host par défaut http://localhost:1234)
  - llama.cpp (llama-server, compatible OpenAI) : /v1/models, /v1/chat/completions (host par défaut http://localhost:8080)

LM Studio et llama.cpp exposent la même API "OpenAI-compatible" ; ils partagent
donc le même code d'appel (_openai_chat / _openai_describe_image), seul le
host et le libellé changent. OPENAI_COMPATIBLE_BACKENDS liste ces backends.

Le reste de l'appli (main.py) ne connaît pas ces détails : il appelle
list_models() / chat() / describe_image() / ping() sur un LLMClient,
qui dispatche en interne vers le bon backend selon `self.backend`.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from typing import Callable, List, Optional

import requests


class LLMError(RuntimeError):
    pass


# Backends qui parlent l'API "OpenAI-compatible" (/v1/models, /v1/chat/completions).
# Ollama reste à part car il a sa propre API native (/api/tags, /api/chat).
OPENAI_COMPATIBLE_BACKENDS = ("lmstudio", "llamacpp")

BACKEND_LABELS = {
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
    "llamacpp": "llama.cpp",
}


@dataclass
class LLMModel:
    name: str
    size_bytes: int = 0
    family: str = ""


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        backend: str = "ollama",
        lmstudio_url: str = "http://localhost:1234",
        llamacpp_url: str = "http://localhost:8080",
    ):
        self.backend = backend  # "ollama", "lmstudio" ou "llamacpp"
        self._ollama_url = base_url.rstrip("/")
        self._lmstudio_url = lmstudio_url.rstrip("/")
        self._llamacpp_url = llamacpp_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # base_url pointe toujours sur le host du backend actif. Les trois hosts
    # sont conservés séparément (self._ollama_url / self._lmstudio_url /
    # self._llamacpp_url) pour ne pas en perdre un quand on bascule sur un
    # autre dans les réglages.
    # ------------------------------------------------------------------ #
    @property
    def base_url(self) -> str:
        if self.backend == "lmstudio":
            return self._lmstudio_url
        if self.backend == "llamacpp":
            return self._llamacpp_url
        return self._ollama_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        value = (value or "").rstrip("/")
        if self.backend == "lmstudio":
            self._lmstudio_url = value
        elif self.backend == "llamacpp":
            self._llamacpp_url = value
        else:
            self._ollama_url = value

    # Accès direct à chaque host, indépendamment du backend actif (utile pour
    # que l'UI des réglages puisse mettre à jour les trois champs sans avoir
    # à basculer temporairement le backend).
    @property
    def ollama_url(self) -> str:
        return self._ollama_url

    @ollama_url.setter
    def ollama_url(self, value: str) -> None:
        self._ollama_url = (value or "").rstrip("/")

    @property
    def lmstudio_url(self) -> str:
        return self._lmstudio_url

    @lmstudio_url.setter
    def lmstudio_url(self, value: str) -> None:
        self._lmstudio_url = (value or "").rstrip("/")

    @property
    def llamacpp_url(self) -> str:
        return self._llamacpp_url

    @llamacpp_url.setter
    def llamacpp_url(self, value: str) -> None:
        self._llamacpp_url = (value or "").rstrip("/")

    # ------------------------------------------------------------------ #
    # Découverte des modèles
    # ------------------------------------------------------------------ #
    def list_models(self) -> List[LLMModel]:
        backend_label = BACKEND_LABELS.get(self.backend, self.backend)
        try:
            if self.backend in OPENAI_COMPATIBLE_BACKENDS:
                r = requests.get(f"{self.base_url}/v1/models", timeout=10)
                r.raise_for_status()
                data = r.json()
                return [LLMModel(name=m.get("id", "")) for m in data.get("data", []) if m.get("id")]

            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            hints = {
                "lmstudio": "Vérifiez que le serveur local LM Studio est lancé (onglet Developer > Start Server).",
                "llamacpp": (
                    "Vérifiez que llama-server est lancé (ex: `llama-server -m model.gguf --port 8080`) "
                    "et que l'URL est correcte. Note : certains anciens builds de llama.cpp n'exposent pas "
                    "/v1/models — mettez à jour llama.cpp si cette erreur persiste."
                ),
            }
            hint = hints.get(self.backend, "Vérifiez qu'Ollama tourne (`ollama serve`) et que l'URL est correcte.")
            raise LLMError(
                f"Impossible de contacter {backend_label} sur {self.base_url}.\n{hint}\nDétail : {e}"
            ) from e

        data = r.json()
        models = []
        for m in data.get("models", []):
            models.append(
                LLMModel(
                    name=m.get("name") or m.get("model", ""),
                    size_bytes=m.get("size", 0),
                    family=(m.get("details") or {}).get("family", ""),
                )
            )
        return models

    def ping(self) -> bool:
        try:
            if self.backend in OPENAI_COMPATIBLE_BACKENDS:
                requests.get(f"{self.base_url}/v1/models", timeout=3)
            else:
                requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------ #
    # Chat / génération de texte
    # ------------------------------------------------------------------ #
    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        num_ctx: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Envoie system+user au backend actif. Si `on_token` est fourni, la réponse
        est streamée token par token (utile pour l'affichage progressif dans
        l'UI) et la fonction est appelée à chaque fragment reçu.
        Retourne toujours le texte complet à la fin.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.backend in OPENAI_COMPATIBLE_BACKENDS:
            return self._openai_chat(model, messages, temperature, on_token)
        return self._ollama_chat(model, messages, temperature, num_ctx, on_token)

    def _ollama_chat(self, model, messages, temperature, num_ctx, on_token):
        url = f"{self.base_url}/api/chat"

        # `temperature`/`num_ctx` peuvent venir directement d'un fichier de config
        # potentiellement corrompu : on retombe sur des valeurs sûres plutôt que
        # de laisser un ValueError remonter non catché.
        try:
            temp_value = float(temperature)
        except (TypeError, ValueError):
            temp_value = 0.7
        options = {"temperature": temp_value}
        if num_ctx:
            try:
                options["num_ctx"] = int(num_ctx)
            except (TypeError, ValueError):
                pass

        payload = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": bool(on_token),
        }

        try:
            if on_token:
                full = []
                with requests.post(url, json=payload, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line.decode("utf-8"))
                        if "message" in chunk:
                            piece = chunk["message"].get("content", "")
                            if piece:
                                full.append(piece)
                                on_token(piece)
                        if chunk.get("done"):
                            break
                return "".join(full)
            else:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "")
        except requests.RequestException as e:
            raise LLMError(f"Erreur pendant l'appel à Ollama ({model}) : {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise LLMError(
                f"Réponse invalide reçue d'Ollama ({model}) : flux interrompu ou JSON malformé. Détail : {e}"
            ) from e

    def _openai_chat(self, model, messages, temperature, on_token):
        """Appel chat pour tout backend "OpenAI-compatible" (LM Studio, llama.cpp)."""
        backend_label = BACKEND_LABELS.get(self.backend, self.backend)
        url = f"{self.base_url}/v1/chat/completions"
        try:
            temp_value = float(temperature)
        except (TypeError, ValueError):
            temp_value = 0.7
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp_value,
            "stream": bool(on_token),
        }

        try:
            if on_token:
                full = []
                with requests.post(url, json=payload, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        if raw == "[DONE]":
                            break
                        chunk = json.loads(raw)
                        choices = chunk.get("choices") or []
                        if choices:
                            piece = choices[0].get("delta", {}).get("content", "")
                            if piece:
                                full.append(piece)
                                on_token(piece)
                return "".join(full)
            else:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.RequestException as e:
            raise LLMError(f"Erreur pendant l'appel à {backend_label} ({model}) : {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise LLMError(
                f"Réponse invalide reçue de {backend_label} ({model}) : flux interrompu ou JSON malformé. Détail : {e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Vision (description automatique d'image de référence)
    # ------------------------------------------------------------------ #
    def describe_image(
        self,
        vision_model: str,
        image_path: str,
        instruction: str = (
            "Describe this image in dense, literal, observable detail for use as a video "
            "generation reference: subjects and their exact appearance (hair, clothing, "
            "colors, pose, expression), composition and framing, environment, lighting, "
            "color palette, and overall visual style. Do not interpret emotions or story; "
            "describe only what is visible. Write in English, 3-6 sentences."
        ),
    ) -> str:
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except OSError as e:
            raise LLMError(f"Impossible de lire l'image {image_path} : {e}") from e

        # Devine le vrai type MIME (png/webp/jpeg…) au lieu de forcer jpeg,
        # certains serveurs de vision refusant ou mal interprétant une image
        # dont le MIME déclaré ne correspond pas aux données.
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        try:
            if self.backend in OPENAI_COMPATIBLE_BACKENDS:
                url = f"{self.base_url}/v1/chat/completions"
                payload = {
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                            ],
                        }
                    ],
                    "stream": False,
                }
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            url = f"{self.base_url}/api/chat"
            payload = {
                "model": vision_model,
                "messages": [{"role": "user", "content": instruction, "images": [b64]}],
                "stream": False,
            }
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.RequestException as e:
            backend_label = BACKEND_LABELS.get(self.backend, self.backend)
            raise LLMError(
                f"Erreur pendant la description d'image ({backend_label}, {vision_model}) : {e}"
            ) from e
