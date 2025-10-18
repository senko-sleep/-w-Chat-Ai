# enhanced_ollama_chat.py
# Revised, lower-memory, faster version of your script.
# Usage: python enhanced_ollama_chat.py
# Set OLLAMA_MODEL env var to override model (default: tinyllama)

import os
import time
import json
import gzip
import socket
import hashlib
import asyncio
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import aiofiles

import ollama  # assumes ollama python client is installed

# ---------- small helpers ----------
def truncate(text: str, max_len: int) -> str:
    return text[:max_len] + ("..." if len(text) > max_len else "")

def compress_json(data: Dict) -> bytes:
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return gzip.compress(js.encode("utf-8"))

def decompress_json(b: bytes) -> Dict:
    return json.loads(gzip.decompress(b).decode("utf-8"))

# ---------- ensure ollama server ----------
def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def ensure_ollama_running(wait_seconds: float = 6.0):
    # Ollama service default listener is localhost:11434
    host, port = "127.0.0.1", 11434
    if is_port_open(host, port):
        return
    # If process exists, maybe starting — avoid duplicate starts
    for p in psutil.process_iter(attrs=["name"]):
        name = (p.info.get("name") or "").lower()
        if "ollama" in name:
            # wait a bit for it to bind
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                if is_port_open(host, port):
                    return
                time.sleep(0.25)
            break
    # start ollama serve (cross-platform attempt)
    print("🚀 starting ollama server...")
    try:
        if os.name == "nt":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[Warn] failed to start ollama process: {e}")
    # wait for socket
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_port_open(host, port):
            return
        time.sleep(0.25)
    print("[Warn] ollama did not open port within timeout. Ensure Ollama is installed and 'ollama serve' works.")

# ---------- lightweight embedding store & functions ----------
class SimpleEmbeddings:
    def __init__(self, use_embeddings: bool = False, dim: int = 768):
        self.use = use_embeddings
        self.dim = dim
        self.cache: Dict[str, np.ndarray] = {}
        self.max_cache = 50

    def get(self, text: str) -> np.ndarray:
        if not self.use:
            return np.zeros(self.dim, dtype=np.float32)
        key = hashlib.md5(truncate(text, 300).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        try:
            resp = ollama.embeddings(model="nomic-embed-text", prompt=truncate(text, 300))
            emb = np.array(resp.get("embedding", []), dtype=np.float32)
            if emb.size == 0:
                raise ValueError("empty embedding")
            self.cache[key] = emb
            if len(self.cache) > self.max_cache:
                self.cache.pop(next(iter(self.cache)))
            return emb
        except Exception as e:
            # disable embeddings on error
            print(f" [Emb fail: {e}] — disabling embeddings")
            self.use = False
            self.cache.clear()
            return np.zeros(self.dim, dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

# ---------- memory manager (lower footprint) ----------
class LowMemMemory:
    def __init__(self, file: str = "chat_memory.json.gz"):
        self.file = file
        self.live: List[Dict[str,str]] = []
        self.stored: List[Dict[str,str]] = []
        self.max_live = 40
        self.max_stored = 150
        self.emb_pairs: List[Tuple[np.ndarray, Dict[str,str]]] = []
        self.max_emb_pairs = 60
        self.embedding_disabled = False
        self.load()

    def add(self, user: str, ai: str, emb: np.ndarray = None):
        user = truncate(user, 200); ai = truncate(ai, 200)
        rec = {"timestamp": datetime.now().isoformat(), "user_message": user, "ai_response": ai}
        self.live.append(rec)
        if len(self.live) > self.max_live:
            self.live.pop(0)
        if emb is not None and emb.size:
            self.emb_pairs.append((emb, rec))
            if len(self.emb_pairs) > self.max_emb_pairs:
                self.emb_pairs.pop(0)

    def get_relevant(self, q_emb: np.ndarray, top: int = 1) -> List[Dict[str,str]]:
        if q_emb is None or q_emb.size == 0 or not self.emb_pairs:
            return []
        scored = sorted(((SimpleEmbeddings.cosine(q_emb, e), r) for e, r in self.emb_pairs), key=lambda x: -x[0])
        return [r for s, r in scored[:top]]

    async def save(self):
        try:
            self.stored.extend(self.live)
            if len(self.stored) > self.max_stored:
                self.stored = self.stored[-self.max_stored:]
            payload = {"stored": self.stored, "last_save": datetime.now().isoformat()}
            comp = compress_json(payload)
            async with aiofiles.open(self.file, "wb") as f:
                await f.write(comp)
            print(f"💾 saved memory ({len(comp)} bytes)")
        except Exception as e:
            print(f"[Error] save memory: {e}")

    def load(self):
        if not os.path.exists(self.file):
            return
        try:
            with open(self.file, "rb") as f:
                data = decompress_json(f.read())
            self.stored = data.get("stored", [])[-self.max_stored:]
            print(f"📂 loaded {len(self.stored)} saved interactions")
        except Exception as e:
            print(f"[Warn] failed to load memory: {e}")
            try:
                os.remove(self.file)
            except Exception:
                pass

# ---------- main chat system ----------
class EnhancedOllamaChat:
    def __init__(self, model: str = None, use_embeddings: bool = False):
        self.model = model or os.getenv("OLLAMA_MODEL", "tinyllama")
        self.client = ollama.Client()
        self.memory = LowMemMemory()
        self.emb = SimpleEmbeddings(use_embeddings=use_embeddings)
        self.system_prompt = ""
        if os.path.exists("prompt.txt"):
            try:
                with open("prompt.txt", "r", encoding="utf-8") as f:
                    self.system_prompt = f.read().strip()
            except Exception:
                pass
        print(f"🧠 OllamaChat initialized: model={self.model} embeddings={'on' if self.emb.use else 'off'}")

    def build_messages(self, query: str) -> List[Dict[str,str]]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role":"system","content":truncate(self.system_prompt, 800)})
        # small context summary
        if self.memory.stored:
            recent = self.memory.stored[-2:]
            summary = " / ".join(truncate(r.get("user_message","")+" | "+r.get("ai_response",""), 120) for r in recent)
            msgs.append({"role":"system","content":truncate("Recent: "+summary, 200)})
        # relevant from embeddings if available
        if self.emb.use:
            q_emb = self.emb.get(query)
            rel = self.memory.get_relevant(q_emb, top=1)
            for r in rel:
                msgs.append({"role":"system","content":truncate(f"Rel U:{r['user_message']} A:{r['ai_response']}", 200)})
        # attach small current context
        for r in self.memory.live[-4:]:
            msgs.append({"role":"user","content":truncate(r["user_message"],200)})
            msgs.append({"role":"assistant","content":truncate(r["ai_response"],200)})
        msgs.append({"role":"user","content":truncate(query, 1000)})
        return msgs

    async def chat(self, query: str) -> str:
        try:
            msgs = self.build_messages(query)
            # quick total length guard
            total_chars = sum(len(m.get("content","")) for m in msgs)
            if total_chars > 8000:
                msgs = msgs[-6:]  # aggressive trim
            print("🤖 AI:", end=" ", flush=True)
            response_parts = []
            # options tuned down for speed/memory
            stream = self.client.chat(model=self.model, messages=msgs, stream=True,
                                      options={"num_predict": 192, "temperature": 0.6})
            start = time.time()
            for chunk in stream:
                try:
                    if "message" in chunk and "content" in chunk["message"]:
                        text = chunk["message"]["content"]
                        print(text, end="", flush=True)
                        response_parts.append(text)
                except Exception:
                    # ignore malformed chunks
                    continue
            elapsed = time.time() - start
            print(f" [{elapsed:.2f}s]")
            full = "".join(response_parts).strip()
            # save to memory (attempt embedding if enabled)
            emb_vec = None
            if self.emb.use:
                emb_vec = self.emb.get(f"User: {truncate(query,200)} Assistant: {truncate(full,200)}")
            self.memory.add(query, full, emb=emb_vec)
            return full if full else "[No response]"
        except Exception as e:
            err = f"[Error] Ollama chat failed: {e}"
            print("\n" + err)
            # fallback: attempt a non-streamed call once
            try:
                resp = self.client.chat(model=self.model, messages=[{"role":"user","content":truncate(query,1000)}],
                                        stream=False, options={"num_predict":128})
                text = resp.get("choices")[0].get("message", {}).get("content", "")
                self.memory.add(query, text)
                return text
            except Exception as e2:
                return f"{err} | fallback failed: {e2}"

    def show_stats(self):
        s = {"stored": len(self.memory.stored), "live": len(self.memory.live), "embeddings": self.emb.use}
        print("\n📊", s)

    def show_context(self):
        print("\n💬 last messages:")
        for m in self.memory.live[-6:]:
            print("  You:", truncate(m["user_message"], 80))
            print("  AI :", truncate(m["ai_response"], 80))

    async def run(self):
        try:
            while True:
                u = input("👤 You: ").strip()
                if not u:
                    continue
                if u.lower() in ("exit","quit"):
                    break
                if u.lower() == "help":
                    print("help | stats | context | save | exit")
                    continue
                if u.lower() == "stats":
                    self.show_stats(); continue
                if u.lower() == "context":
                    self.show_context(); continue
                if u.lower() == "save":
                    await self.memory.save(); continue
                await self.chat(u)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            print("Saving memory...")
            await self.memory.save()
            print("Goodbye.")

# ---------- entry point ----------
async def main():
    ensure_ollama_running()
    model = os.getenv("OLLAMA_MODEL", "tinyllama")  # set env to switch models
    # default: embeddings off to reduce memory & latency. Set USE_EMB=1 env to enable.
    use_emb = os.getenv("USE_EMB", "0") == "1"
    chat = EnhancedOllamaChat(model=model, use_embeddings=use_emb)
    await chat.run()

if __name__ == "__main__":
    asyncio.run(main())
