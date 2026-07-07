import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
for k in ("ALL_PROXY","all_proxy","SOCKS_PROXY","socks_proxy"):
    os.environ.pop(k,None)
import yaml
from linkerbot.voice.classifier import GestureClassifier
from linkerbot.voice.gesture_library import GestureLibrary

glib = GestureLibrary(os.path.join(os.path.dirname(__file__), "..", "config", "gestures.yaml"))
glib.load()
vp = os.path.join(os.path.dirname(__file__), "..", "config", "voice.yaml")
with open(vp) as f:
    vc_cfg = yaml.safe_load(f).get("voice", {})
api_key = vc_cfg.get("api_key", "")
if api_key.startswith("${"):
    api_key = os.environ.get(api_key[2:-1], "")
env_path = os.path.join(os.path.dirname(__file__), "..", ".env_test")
if not api_key and os.path.exists(env_path):
    for line in open(env_path):
        if line.startswith("DEEPSEEK_API_KEY="):
            api_key = line.strip().split("=",1)[1]; break
if not api_key:
    print("No API key"); sys.exit(1)

clf = GestureClassifier(api_key=api_key)
tests = ["fist", "open hand", "thumbs up", "peace sign", "grab something", "pinch"]
for t in tests:
    r = clf.classify(t, glib.names)
    print(f'  "{t}" -> {r}')
print("Done!")
