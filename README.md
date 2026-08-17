# H3 Prompt Studio

Application desktop (Python / CustomTkinter) pour construire des prompts structurés et conformes au format **MiniMax H3** (modèle audio-vidéo open-source), à partir de formulaires guidés plutôt que d'écrire le prompt à la main. Tourne 100% en local via un LLM (Ollama, LM Studio ou llama.cpp) — aucune donnée n'est envoyée à un service externe.

H3 attend un format de prompt rigide, structuré et étiqueté, très différent d'une description en langage naturel. L'app fait l'aller-retour : vous remplissez des champs simples (scénario, style, dialogue, références...), le LLM local transforme ça en prompt H3 complet et conforme.

---

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Les 5 modes de génération](#les-5-modes-de-génération)
- [Onglet Story → Sequences](#onglet-story--sequences)
- [Backends LLM supportés](#backends-llm-supportés)
- [Installation](#installation)
- [Configuration](#configuration)
- [Architecture du code](#architecture-du-code)
- [Limites connues](#limites-connues)

---

## Fonctionnalités

- **5 modes de génération H3** couvrant tous les cas d'usage du modèle (texte seul, image de première frame, première+dernière frame, dernière frame seule, multi-références).
- **Mode storyboard** dans Ref2VA : découpe une scène en plusieurs scènes chaînées, chacune avec son propre dialogue et ses propres références, générées en une seule séquence.
- **Onglet Story → Sequences** : à partir d'une bibliothèque d'images de référence, génère une courte histoire puis la découpe automatiquement en N séquences vidéo, chacune développée en prompt Ref2VA complet.
- **Description automatique d'images** via un modèle de vision local (ex. `qwen2.5vl`), pour éviter de décrire chaque référence à la main.
- **Streaming** de la génération token par token dans la zone de sortie.
- **3 backends LLM interchangeables** : Ollama, LM Studio, llama.cpp (llama-server) — tous 100% locaux.
- **Réglages persistants** (backend, URLs, modèles, température, durée par défaut) sauvegardés automatiquement dans `~/.h3_prompt_studio_config.json`.
- **Sauvegarde / chargement de session** pour l'onglet Story → Sequences (références, histoire, séquences et réglages) au format JSON.
- **Export** du prompt généré en presse-papier ou en fichier `.txt`.

---

## Les 5 modes de génération

Chaque mode correspond à un onglet, avec un formulaire adapté (scénario, style visuel, notes caméra, dialogue, ambiance sonore, musique) et produit un prompt H3 conforme (champs `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`, ou la structure 6-sections de Ref2VA).

| Mode | Description | Entrée |
|---|---|---|
| **T2VA** | Texte → Vidéo+Audio | Aucune image, tout est construit depuis un scénario texte. |
| **I2VA** | Image → Vidéo+Audio | Une image ancre la première frame (0.00s) ; le LLM décrit la suite. |
| **FL2VA** | Première + Dernière frame → Vidéo+Audio | Deux images ancrent le début et la fin ; le LLM décrit le mouvement entre les deux. |
| **L2VA** | Dernière frame seule → Vidéo+Audio | Une image ancre la fin ; le LLM invente un état de départ plausible. |
| **Ref2VA** | Multi-références (mode complet) | Jusqu'à 9 images / 3 vidéos / 3 audios (12 références max) ; le LLM assigne `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>` et écrit les 6 sections du prompt (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music). |

Ref2VA a également un **mode storyboard** activable : on définit N scènes, chacune avec son propre brief, dialogue et sélection de références parmi la bibliothèque partagée ; chaque scène est développée en un prompt H3 indépendant.

Champs communs à tous les modes (bloc "Dialogue / voice" + "Sound & music") :
- Dialogue avec speaker, langue, texte exact, option voiceover.
- Texte à l'écran (enseignes, sous-titres...).
- Ambiance sonore et musique (ou silence explicite).
- Notes libres additionnelles pour le LLM.

---

## Onglet Story → Sequences

Construit un prompt Ref2VA par séquence à partir d'une seule bibliothèque de références, en 3 passes LLM :

1. **Références** : on ajoute des images/vidéos/audios avec rôle et description (comme dans Ref2VA), avec description automatique via modèle de vision.
2. **Génération de l'histoire** (Pass A) : le LLM écrit une courte narration (~200-500 mots) qui utilise les références fournies, à partir d'une prémisse optionnelle et d'une langue cible.
3. **Découpage en séquences** (Pass B) : le LLM reçoit l'histoire, la bibliothèque de références et le vocabulaire de mouvements de caméra, et retourne un tableau JSON de N séquences — pour chacune il choisit lui-même les références concernées, la présence/absence de dialogue, et le mouvement de caméra. Un mécanisme de réconciliation ajoute automatiquement les références omises par erreur (recoupement par mots-clés entre le brief de la séquence et la description des références).
4. **Génération des prompts finaux** : chaque séquence est développée en un prompt Ref2VA complet et isolé (règle d'isolation stricte : le contenu d'une séquence ne doit jamais fuiter dans une autre).

Robustesse : réparation de JSON tronqué (troncature liée au `max_tokens` des modèles locaux) avec retry automatique en cas d'échec de parsing.

**Sauvegarde / chargement** : boutons dédiés pour enregistrer toute la session (références, prémisse, langue, histoire, nombre/durée de séquences, style, instructions additionnelles, séquences générées) dans un fichier `.json`, et la recharger plus tard pour reprendre le travail.

---

## Backends LLM supportés

| Backend | API | Host par défaut |
|---|---|---|
| **Ollama** | native (`/api/tags`, `/api/chat`) | `http://localhost:11434` |
| **LM Studio** | compatible OpenAI (`/v1/models`, `/v1/chat/completions`) | `http://localhost:1234` |
| **llama.cpp** (llama-server) | compatible OpenAI (`/v1/models`, `/v1/chat/completions`) | `http://localhost:8080` |

Le backend est sélectionnable dans l'onglet Réglages ; les trois URLs sont conservées indépendamment (basculer de backend ne perd pas la config des autres). Le chat et la description d'image (vision) fonctionnent avec les trois backends, y compris en streaming.

---

## Installation

```bash
pip install customtkinter requests
```

Un des trois backends doit tourner localement :

```bash
# Ollama
ollama serve
ollama pull qwen2.5:14b-instruct     # modèle de chat
ollama pull qwen2.5vl:7b             # modèle de vision (optionnel, pour la description auto d'images)
```

Puis lancer l'application :

```bash
python main.py
```

**Recommandation** : un modèle de chat 14B+ suit beaucoup mieux le format H3 structuré qu'un petit modèle.

---

## Configuration

Dans l'onglet **⚙ Settings** :
- Choix du backend et de ses URLs.
- Bouton "Test connection / refresh models" pour lister les modèles disponibles sur le backend actif.
- Choix du modèle de chat (texte) et du modèle de vision (optionnel).
- Température et durée par défaut.

Les réglages sont sauvegardés automatiquement dans `~/.h3_prompt_studio_config.json` à la fermeture de l'app.

---

## Architecture du code

```
main.py               # UI CustomTkinter : onglets par mode, Ref2VA + storyboard,
                       # Story→Sequences, Réglages, zone de sortie/streaming
system_prompts.py      # Règles H3 (caméra, dialogue, texte à l'écran, son, durée,
                       # timestamps) assemblées en system prompts par mode
sequence_pipeline.py    # Pipeline Story→Sequences : génération d'histoire,
                       # découpage JSON en séquences, réparation JSON tronqué,
                       # réconciliation des références, construction des briefs
llm_client.py          # Abstraction multi-backend (Ollama / LM Studio / llama.cpp) :
                       # chat streaming, listing des modèles, description d'image
```

---

## Limites connues

- Ref2VA / Story→Sequences : maximum 9 images, 3 vidéos, 3 audios, 12 références au total (limite du modèle H3).
- La qualité du JSON structuré (découpage en séquences, prompts H3) dépend fortement de la capacité du modèle de chat local à suivre des instructions de format strict — les petits modèles (<7B) sont sujets à des troncatures ou des JSON malformés, partiellement compensées par les mécanismes de réparation intégrés.
- Aucune génération vidéo/audio réelle n'est effectuée par l'application : elle produit uniquement le prompt texte destiné à être soumis à H3 (via ComfyUI ou autre runtime compatible).
