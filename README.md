# Vosk voice command tester

A Jupyter notebook for testing offline speech recognition of short spoken game commands
in English, Spanish and Russian with [Vosk](https://alphacephei.com/vosk/). It lets you
tune the filters that separate commands spoken *to* the app from background speech.

- **Language modes.** Only the current language's commands are recognized. Switch by
  saying the language's name in that language ("english", "español", "русский"), or from
  a menu or text box.
- **Silence gates.** A command must have a quiet gap before it, and optionally after it.
  Threshold, gap lengths and tolerance are all sliders.
- **Two command sets**, switchable live: `original`, and `distinct`, whose words were
  picked to sound unlike each other and unlike everyday speech (see below).
- **Confidence threshold**, with optional *decoy words* added to the grammar so that
  Vosk's confidence becomes meaningful.
- **Double-check.** Each candidate is re-transcribed with Vosk's full vocabulary, or
  with Whisper, and compared to the command by sound-alike similarity.
- A live waveform and dB plot with the threshold line. Each candidate is shaded:
  pending, accepted or rejected.
- A log with the reason for every decision.
- **Show vocabulary** lists exactly what can be said in the current mode.
- A device scan and a mic tester with a gain control, for finding the right input and level.

## Setup

Requires Python 3.9+ and a microphone. Tested with WinPython on Windows. Plain
`python -m venv` works the same way.

```bash
git clone https://github.com/ticohannan/vosk-voice-command-tester.git
cd vosk-voice-command-tester
pip install -r requirements.txt
python setup_models.py          # extracts the bundled model zips (downloads any that are missing)
jupyter lab voice_commands.ipynb
```

**WinPython.** Open the *WinPython Command Prompt*, `cd` to the cloned folder, and run the
same commands. If JupyterLab is already set up, open the notebook from there.

**Optional Whisper double-check.** This is more accurate, especially for Russian, but takes
about 0.3 to 1.5 s per command on a CPU.

```bash
pip install faster-whisper
```

Restart the kernel, then choose *Whisper* under **Double-check**. The first use downloads
the model set by `WHISPER_MODEL` in the setup cell (`small` is about 0.5 GB).

## Using the notebook

Run the cells from top to bottom:

1. **Setup.** Commands, language names, decoy words, and starting slider values. Edit
   `commands` to change what can be said.
2. **Load models.** Warns about any command word a model doesn't know. Such a word can
   never be recognized.
3. **Devices / Scan.** While you talk, the scan records from every input and marks which
   ones actually pick up sound.
4. **Mic tester.** Pick a device, a duration and a gain, then click Record. **Use these
   settings** passes the device and gain to the listener.
5. **Listener / Run.** Click Start and speak.

### Tuning

1. Stay quiet for a few seconds, then click **Set threshold from noise floor**.
2. Say commands with a short pause before them. Read the log for the reason behind each
   rejection, and loosen only that setting.
3. Background conversation getting through: raise **Quiet before**, turn on **Quiet
   after**, raise **Min confidence**, and keep **decoy words** on.
4. Real commands rejected by the double-check: the log shows what it heard and the
   similarity score. Lower **Match similarity** a little (0.65 to 0.7), add a variant
   to `commands`, or replace a word the model handles badly.

### Command sets

| Command | English (original / distinct) | Spanish (original / distinct) | Russian (original / distinct) |
|---|---|---|---|
| start | start / **begin** | empezar, iniciar / **comenzar**, empezar | начать, старт / **старт, поехали** |
| done | done / **finished** | listo, terminado / **terminado** | готово, закончил / **готово** |
| quit | quit / **goodbye** | salir / **abandonar** | выход, выйти / **завершить** |
| reset | reset / **restart** | reiniciar / **reiniciar** | сброс, сбросить / **перезапуск** |
| your turn (hand the turn to the computer) | your turn / **your move** | tu turno, es tu turno / **tu jugada** | твоя очередь, твой ход / **твоя очередь** |

The `distinct` set avoids:
- Short words ("done", "quit").
- Phrases contained in another phrase ("iniciar" inside "reiniciar").
- Shared syllables ("выход" vs. "твой ход"; "ход" was also misheard as "год"/"код").
- Words common in ordinary conversation ("listo", "your turn").

Within each language, no two distinct-set phrases score above 0.44 on the notebook's
sound-alike similarity measure. Edit `COMMAND_SETS` in the setup cell to add your own set.

### Hooking up a game

```python
listener.on_command  = lambda cmd, lang, phrase: game.handle(cmd)
listener.on_language = lambda lang: game.set_language(lang)
listener.set_language("es")
```

## Tests

The listener logic has simulation tests that need no microphone or models. They use fake
`sounddevice` and `vosk` modules and synthetic audio.

```bash
pip install pytest
pytest -q
```

## Models and licenses

The three small Vosk models (about 40 MB each) are bundled as zip files in `models/`.
They are unmodified copies from alphacephei.com, licensed under Apache 2.0. See
[`models/NOTICE.md`](models/NOTICE.md) and
[`models/LICENSE-Apache-2.0.txt`](models/LICENSE-Apache-2.0.txt).

`setup_models.py` extracts them. The extracted folders are git-ignored.

The notebook, scripts and tests are © 2026 Tico Hannan ([@ticohannan](https://github.com/ticohannan)), under the BSD 3-Clause license (see `LICENSE`).
