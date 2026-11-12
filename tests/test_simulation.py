"""Simulation tests for the notebook's listener logic.

No microphone or real models are needed: sounddevice and vosk are replaced by fakes.
Synthetic audio has quiet background noise plus tone bursts wherever a scripted word
is "spoken". The fake recognizer reports scripted words with timings, and respects
its grammar and language the same way Vosk does.

    pip install pytest numpy scipy matplotlib ipywidgets
    pytest -q
"""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

R = 16000
NOTEBOOK = Path(__file__).resolve().parents[1] / "voice_commands.ipynb"
CELLS = (1, 2, 3, 6)          # setup, models, devices, listener (skip mic scan / tester)

VOCAB = {
    "en": ("start done quit reset your turn english spanish russian hello the and game "
           "begin finished goodbye restart move").split(),
    "es": ("empezar iniciar listo terminado salir reiniciar tu turno es español inglés ruso hola "
           "comenzar abandonar jugada").split(),
    "ru": ("начать старт готово закончил выход выйти сброс сбросить твоя очередь твой ход "
           "русский английский испанский и в не год что поехали завершить перезапуск "
           "очередь").split(),
}
PATHS = {"en": "models/vosk-model-small-en-us-0.15",
         "es": "models/vosk-model-small-es-0.42",
         "ru": "models/vosk-model-small-ru-0.22"}


class World:
    """Shared state between the fakes and the test."""
    def __init__(self):
        self.sched = []            # [(spoken_lang, [(word, start_s, end_s, conf)])]
        self.relative = False      # word times per utterance instead of per stream
        self.total = 0             # listener.total when a recognizer is created
        self.open_text = {}        # fake full-vocabulary result: {"text":..., "conf":...}


W = World()


class FakeModel:
    def __init__(self, path):
        self.lang = next(l for l, p in PATHS.items() if p in path.replace("\\", "/"))


class FakeRecognizer:
    def __init__(self, model, rate, grammar=None):
        self.lang = model.lang
        self.phrases = None if grammar is None else [p.split() for p in json.loads(grammar)]
        self.origin, self.fed, self.done = W.total, 0, set()
        self.utt_start, self.last = self.origin, ""

    def SetWords(self, v):
        pass

    def AcceptWaveform(self, data):
        self.fed += len(data) // 2
        if self.phrases is None:
            return False
        now = (self.origin + self.fed) / R
        for k, (lang, words) in enumerate(W.sched):
            if k in self.done:
                continue
            if words[0][1] * R < self.origin:
                self.done.add(k)
                continue
            if now >= words[-1][2] + 0.5:          # endpoint after 0.5 s of silence
                self.done.add(k)
                base = (self.utt_start if W.relative else self.origin) / R
                out = [{"word": w if (lang == self.lang and any(w in p for p in self.phrases)) else "[unk]",
                        "start": s - base, "end": e - base, "conf": c} for (w, s, e, c) in words]
                self.utt_start = self.origin + self.fed
                self.last = json.dumps({"result": out, "text": " ".join(o["word"] for o in out)})
                return True
        return False

    def Result(self):
        return self.last

    def PartialResult(self):
        return '{"partial": ""}'

    def FinalResult(self):
        t, c = W.open_text.get("text", ""), W.open_text.get("conf", 0.9)
        return json.dumps({"text": t, "result": [
            {"word": w, "start": 0, "end": 0.3, "conf": c} for w in t.split()]})


@pytest.fixture(scope="module")
def ns(tmp_path_factory):
    root = tmp_path_factory.mktemp("models_root")
    for lang, p in PATHS.items():
        (root / p / "graph").mkdir(parents=True)
        (root / p / "graph" / "words.txt").write_text(
            "\n".join(f"{w} {i}" for i, w in enumerate(VOCAB[lang])), encoding="utf-8")
    vosk = types.ModuleType("vosk")
    vosk.Model, vosk.KaldiRecognizer, vosk.SetLogLevel = FakeModel, FakeRecognizer, lambda x: None
    devs = [{"name": "Test mic", "hostapi": 0, "max_input_channels": 1,
             "default_samplerate": 16000.0}] * 3
    sd = types.ModuleType("sounddevice")
    sd.query_hostapis = lambda: [{"name": "MME"}]
    sd.query_devices = lambda dev=None, kind=None: devs if dev is None else devs[dev]
    sd.default = types.SimpleNamespace(device=[1, 3])
    saved = {k: sys.modules.get(k) for k in ("vosk", "sounddevice", "faster_whisper")}
    sys.modules.update(vosk=vosk, sounddevice=sd)
    sys.modules["faster_whisper"] = None           # behave as "not installed"
    os.environ.setdefault("MPLBACKEND", "Agg")
    cwd = os.getcwd()
    os.chdir(root)
    try:
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        g = {}
        for k in CELLS:
            exec(nb["cells"][k]["source"], g)
    finally:
        os.chdir(cwd)
    yield g
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def utter(lang, t0, text, conf=0.95, wdur=0.35):
    words, t = [], t0
    for w in text.split():
        words.append((w, t, t + wdur, conf))
        t += wdur + 0.05
    W.sched.append((lang, words))


def run_until(L, t_end, extra_bursts=(), rate=16000):
    """Feed audio up to t_end seconds: -60 dBFS noise, -20 dBFS tone for every word."""
    rng = np.random.default_rng(0)
    blk = int(rate * 0.1)
    while L.total / R < t_end:
        t = L.total / R
        spans = [(s, e) for _, ws in W.sched for (_, s, e, _) in ws] + list(extra_bursts)
        tt = t + np.arange(blk) / rate
        x = rng.normal(0, 1e-3, blk)
        for s, e in spans:
            m = (tt >= s) & (tt < e)
            x[m] += 0.14 * np.sin(2 * np.pi * 300 * tt[m])
        W.total = L.total
        L.process_block(x.astype(np.float32))


@pytest.fixture
def L(ns):
    W.__init__()
    ns["use_command_set"]("original")
    lis = ns["CommandListener"]()
    lis.verify_dd.value = "off"
    lis.pre_slider.value, lis.post_slider.value = 1.0, 0.5
    lis.conf_slider.value, lis.sil_slider.value = 0.6, -45
    lis.events = []
    lis.on_command = lambda cmd, lang, phrase: lis.events.append(("cmd", cmd, lang))
    lis.on_language = lambda lang: lis.events.append(("lang", lang))

    def start(rate=16000, lang="en"):
        lis.in_rate, lis.blocks, lis.dropped, lis.last_status = rate, 0, 0, ""
        lis._reset_buffers()
        lis._prepare_dsp()
        lis.lang = lang
        lis._sync_lang_widgets()
        W.total = 0
        lis._build_recognizers()
    lis.begin = start
    return lis


def test_gates_confidence_and_phrases(L):
    L.begin()
    utter("en", 2.0, "start")                                     # clean: accept
    W.sched.append(("en", [("hello", 4.0, 4.5, .9)]))
    utter("en", 4.7, "done")                                      # speech just before: reject
    utter("en", 7.0, "quit")                                      # speech just after: reject
    utter("en", 10.0, "reset", conf=0.4)                          # low confidence: reject
    utter("en", 12.0, "your turn")                                # two-word phrase: accept
    run_until(L, 14, extra_bursts=[(7.55, 8.2)])
    assert L.events == [("cmd", "start", "en"), ("cmd", "your turn", "en")]
    log = "\n".join(L.log_lines)
    assert "not quiet before" in log and "not quiet after" in log and "confidence 0.40" in log


def test_language_switching_by_voice(L):
    L.begin()
    utter("es", 2.0, "listo")              # Spanish command in English mode: ignored
    utter("es", 4.5, "español")            # switch, said in Spanish
    utter("en", 7.0, "start")              # English in Spanish mode: ignored
    utter("es", 9.5, "es tu turno")        # Spanish command
    utter("es", 12.0, "inglés")            # back to English, named in Spanish
    utter("en", 14.5, "done")
    run_until(L, 17)
    assert L.events == [("lang", "es"), ("cmd", "your turn", "es"),
                        ("lang", "en"), ("cmd", "done", "en")]


def test_menu_and_typed_language(L, ns):
    L.begin()
    L.lang_text.value = "Русский"
    assert L.lang == "ru" and L.lang_dd.value == "ru"
    assert [ns["name_to_lang"](t) for t in ["espanol", "russian", "ES", "klingon"]] == ["es", "ru", "es", None]
    L.worker = object()                     # pretend the worker thread is running
    L.set_language("es")
    L.worker = None
    run_until(L, 0.3)
    assert L.lang == "es"


@pytest.mark.parametrize("rate,relative", [(48000, True), (44100, False), (16000, True)])
def test_word_times_and_resampling(L, rate, relative):
    W.relative = relative
    L.begin(rate=rate, lang="es")
    seen = []
    orig = L._decide
    L._decide = lambda c: (seen.append(round(c["start"] / R, 2)), orig(c))
    for t0 in (2.0, 6.0, 10.0):
        utter("es", t0, "listo")
    run_until(L, 12, rate=rate)
    assert seen == [2.0, 6.0, 10.0]
    assert [e[1] for e in L.events] == ["done", "done", "done"]   # 4 s apart: past the 1 s debounce


def test_verify_sound_alike(L):
    L.begin(lang="ru")
    L.verify_dd.value = "vosk"
    L.sim_slider.value = 0.75
    cases = [("твой ход", "твой год", True), ("выход", "вы год", True),
             ("начать", "начат", True), ("сбросить", "спросит", True),
             ("выйти", "мы видели его", False), ("начать", "что это хорошо", False)]
    t = 2.0
    for said, heard, _ in cases:
        utter("ru", t, said)
        run_until(L, t + 0.4)
        W.open_text = {"text": heard, "conf": 0.9}
        run_until(L, t + 2.5)
        t += 3.0
    accepted = [e[1] for e in L.events]
    assert accepted == ["your turn", "quit", "start", "reset"], (accepted, L.log_lines)


def test_decoys_and_vocabulary_panel(L, ns):
    assert "год" not in ns["decoys"]["ru"]                  # sounds like 'ход': excluded
    assert "и" in ns["decoys"]["ru"]
    g_on = ns["grammar_phrases"]("ru", True, True)
    g_off = ns["grammar_phrases"]("ru", True, False)
    assert set(g_off) < set(g_on)
    assert ns["grammar_phrases"]("es", False, True) == ["español"]
    L.begin(lang="ru")
    L.vocab_toggle.value = True
    html = L.vocab_html.value
    assert "твой ход" in html and "español" in html and "Decoy words" in html
    L.vocab_toggle.value = False
    assert L.vocab_html.value == ""
    L._render()
    assert len(L.plot_img.value) > 1000


def test_whisper_path_with_stub_model(L, ns):
    class Seg:
        def __init__(self, text, lp):
            self.text, self.avg_logprob = text, lp

    class StubWhisper:
        def __init__(self):
            self.reply = ""

        def transcribe(self, audio, **kw):
            assert audio.dtype == np.float32 and kw["language"] == "ru"
            return iter([Seg(self.reply, -0.2)]), None

    stub = StubWhisper()
    ns["_whisper"]["model"] = stub
    try:
        L.begin(lang="ru")
        L.verify_dd.options = [("off", "off"), ("vosk", "vosk"), ("Whisper", "whisper")]
        L.verify_dd.value = "whisper"
        for t, said, heard in [(2.0, "выйти", " Выйти!"), (5.0, "готово", " Что это?")]:
            utter("ru", t, said)
            run_until(L, t + 0.4)
            stub.reply = heard
            run_until(L, t + 2.5)
        assert [e[1] for e in L.events] == ["quit"], L.log_lines
        assert "whisper heard 'выйти' sim 1.00" in "\n".join(L.log_lines)
    finally:
        ns["_whisper"]["model"] = None


def test_command_set_switch_live(L, ns):
    L.begin()
    utter("en", 2.0, "start")                  # original set
    run_until(L, 4.0)
    L.worker = object()                        # pretend running so the switch is queued
    L.cmdset_dd.value = "distinct"
    L.worker = None
    utter("en", 5.0, "start")                  # no longer a command: ignored
    utter("en", 8.0, "your move")
    utter("en", 11.0, "goodbye")
    run_until(L, 13.5)
    assert L.events == [("cmd", "start", "en"), ("cmd", "your turn", "en"), ("cmd", "quit", "en")], L.log_lines
    assert "твоя очередь" in ns["grammar_phrases"]("ru", True, True)
    ns["use_command_set"]("original")
