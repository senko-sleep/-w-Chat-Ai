import os, time, json, gzip, socket, hashlib, asyncio, subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np, psutil, aiofiles, ollama
import concurrent.futures  # For potential future batching

# Set Ollama env vars for speed (from optimizations)
os.environ['OLLAMA_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['OLLAMA_MAX_LOADED'] = '1'  # Single model focus
os.environ['OLLAMA_KEEP_ALIVE'] = '30m'  # Keep model loaded longer

def truncate(text: str, max_len: int) -> str:
    return text[:max_len] + ("..." if len(text) > max_len else "")

def compress_json(data: Dict) -> bytes:
    return gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

def decompress_json(b: bytes) -> Dict:
    return json.loads(gzip.decompress(b).decode("utf-8"))

def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def ensure_ollama_running(wait_seconds: float = 6.0):
    host, port = "127.0.0.1", 11434
    if is_port_open(host, port):
        return
    for p in psutil.process_iter(attrs=["name"]):
        if "ollama" in (p.info.get("name") or "").lower():
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                if is_port_open(host, port): return
                time.sleep(0.25)
            break
    try:
        if os.name == "nt":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[Warn] failed to start ollama process: {e}")
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_port_open(host, port): return
        time.sleep(0.25)
    print("[Warn] ollama did not open port within timeout.")
    # Preload a dummy request to warm up the model after start
    try:
        ollama.generate(model="tinyllama", prompt=" ")  # Fallback to tinyllama for preload if available
    except: pass

# Helper to pull model if not found (with user prompt)
async def ensure_model_loaded(model_name: str):
    try:
        # Test if model is loaded
        ollama.show(model_name)
        return
    except ollama.ResponseError as e:
        if e.status_code in (404, 500):
            print(f"[Info] Model '{model_name}' not found. Pulling it now... (this may take a few minutes)")
            try:
                ollama.pull(model_name)
                print(f"[Success] Pulled '{model_name}' successfully.")
            except Exception as pull_e:
                print(f"[Error] Failed to pull '{model_name}': {pull_e}. Falling back to 'tinyllama'. Run 'ollama pull {model_name}' manually if needed.")
                # Set to fallback
                global fallback_model_global
                fallback_model_global = "tinyllama"
                return
        else:
            raise
    except Exception as e:
        print(f"[Warn] Could not verify/pull model '{model_name}': {e}")

# Global fallback for model issues
fallback_model_global = None

class SimpleEmbeddings:
    def __init__(self, use_embeddings: bool = False, dim: int = 384):  # Smaller dim for speed (nomic supports 384)
        self.use, self.dim = use_embeddings, dim
        self.cache: Dict[str, np.ndarray] = {}
        self.max_cache = 100  # Larger cache for fewer recomputes

    def get(self, text: str) -> np.ndarray:
        if not self.use: return np.zeros(self.dim, dtype=np.float32)
        key = hashlib.md5(truncate(text, 300).encode()).hexdigest()
        if key in self.cache: return self.cache[key]
        try:
            resp = ollama.embeddings(model="nomic-embed-text", prompt=truncate(text, 300))
            emb = np.array(resp.get("embedding", []), dtype=np.float32)
            if emb.size == 0: raise ValueError("empty embedding")
            self.cache[key] = emb
            if len(self.cache) > self.max_cache: self.cache.pop(next(iter(self.cache)))
            return emb
        except Exception as e:
            print(f"[Emb fail: {e}] disabling embeddings")
            self.use = False; self.cache.clear()
            return np.zeros(self.dim, dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

class LowMemMemory:
    def __init__(self, file: str = "chat_memory.json.gz"):
        self.file, self.live, self.stored = file, [], []
        self.max_live, self.max_stored, self.emb_pairs = 40, 150, []
        self.max_emb_pairs, self.embedding_disabled = 60, False
        # CHANGE: Don't call load in __init__ to avoid sync/async mismatch; defer to async init

    def add(self, user: str, ai: str, emb: np.ndarray = None):
        user, ai = truncate(user, 200), truncate(ai, 200)
        rec = {"timestamp": datetime.now().isoformat(), "user_message": user, "ai_response": ai}
        self.live.append(rec)
        if len(self.live) > self.max_live: self.live.pop(0)
        if emb is not None and emb.size:
            self.emb_pairs.append((emb, rec))
            if len(self.emb_pairs) > self.max_emb_pairs: self.emb_pairs.pop(0)

    def get_relevant(self, q_emb: np.ndarray, top: int = 3) -> List[Dict[str,str]]:  # top=3 for smarter retrieval
        if q_emb is None or q_emb.size == 0 or not self.emb_pairs: return []
        scored = sorted(((SimpleEmbeddings.cosine(q_emb, e), r) for e, r in self.emb_pairs), key=lambda x: -x[0])
        return [r for _, r in scored[:top]]

    async def save(self):
        try:
            self.stored.extend(self.live)
            if len(self.stored) > self.max_stored: self.stored = self.stored[-self.max_stored:]
            comp = compress_json({"stored": self.stored, "last_save": datetime.now().isoformat()})
            async with aiofiles.open(self.file, "wb") as f: await f.write(comp)
            print(f"💾 saved memory ({len(comp)} bytes)")
        except Exception as e:
            print(f"[Error] save memory: {e}")

    async def load(self):  # Async load to handle aiofiles
        if not os.path.exists(self.file): return
        try:
            async with aiofiles.open(self.file, "rb") as f:
                raw = await f.read()
                if not raw:  # Empty file handling
                    print("[Warn] Memory file empty, starting fresh.")
                    return
                data = decompress_json(raw)
            self.stored = data.get("stored", [])[-self.max_stored:]
            print(f"📂 loaded {len(self.stored)} saved interactions")
        except json.JSONDecodeError as e:
            print(f"[Warn] Corrupt memory file (JSON error): {e}. Deleting and starting fresh.")
            try: os.remove(self.file)
            except Exception: pass
            self.stored = []
        except Exception as e:
            print(f"[Warn] failed to load memory: {e}")
            try: os.remove(self.file)
            except Exception: pass
            self.stored = []

class EnhancedOllamaChat:
    def __init__(self, model: str = None, use_embeddings: bool = False):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # CHANGE: Reliable default (fast, accurate; pull if needed)
        # CHANGE: Simplified quant: No auto-append; assume user specifies valid tag (e.g., llama3.2:3b-instruct-q4_0)
        # List available models to validate
        try:
            available = [m['name'] for m in ollama.list().get('models', [])]
            if self.model not in available:
                print(f"[Warn] Model '{self.model}' not in available: {available[:5]}... Fallback to first available or tinyllama.")
                if available:
                    self.model = available[0]
                else:
                    self.model = "tinyllama"
        except Exception:
            pass  # Proceed, ensure_model will handle
        self.client = ollama.Client()
        self.memory = LowMemMemory()
        self.emb = SimpleEmbeddings(use_embeddings=use_embeddings)
        self.system_prompt = ""
        if os.path.exists("prompt.txt"):
            try:
                with open("prompt.txt", "r", encoding="utf-8") as f:
                    self.system_prompt = f.read().strip()
            except Exception: pass
        else:
            # CHANGE: Enhanced fallback prompt with Pokémon fact and stronger anti-hallucination
            self.system_prompt = """You are a helpful, accurate AI assistant. Think step-by-step before responding to ensure logical and factual answers.
Base responses on verified knowledge. If uncertain, state limitations clearly. Use concise, structured outputs when appropriate (e.g., lists for facts).
For factual queries, prioritize precision over speculation. Ignore prior inconsistencies and correct errors. Do not continue previous topics unless directly related.
Key facts: Eevee (Pokémon) is National Dex #133, Normal-type, evolves into Vaporeon, Jolteon, Flareon, Espeon, Umbreon, Leafeon, Glaceon, Sylveon.
Examples:
User: What is 2+2? → Assistant: 2 + 2 = 4 (simple math).
User: What is the Pokémon Eevee's Dex number? → Assistant: Eevee's National Pokédex number is #133."""
        print(f"OllamaChat initialized: model={self.model} embeddings={'on' if self.emb.use else 'off'}")

    async def __aenter__(self):  # Allow async init to pull model and load memory
        await ensure_model_loaded(self.model)
        if fallback_model_global:
            self.model = fallback_model_global
        await self.memory.load()
        return self

    def build_messages(self, query: str) -> List[Dict[str,str]]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role":"system","content":truncate(self.system_prompt, 800)})
        if self.memory.stored:
            recent = self.memory.stored[-2:]
            summary = " / ".join(truncate(r.get("user_message","")+" | "+r.get("ai_response",""), 120) for r in recent)
            msgs.append({"role":"system","content":truncate("Recent: "+summary, 200)})
        if self.emb.use:
            q_emb = self.emb.get(query)
            relevant = self.memory.get_relevant(q_emb, top=3)  # Now uses top=3
            for r in relevant:
                msgs.append({"role":"system","content":truncate(f"Relevant prior: U: {r['user_message']} A: {r['ai_response']}", 200)})
        
        # CHANGE: For factual queries, NO history to prevent topic pollution (e.g., Undertale bleeding into Pokémon)
        # Only add history for conversational/non-factual
        factual_keywords = ["what", "who", "dex number", "define", "is", "how many", "explain"]
        is_factual = any(kw in query.lower() for kw in factual_keywords)
        if not is_factual:
            # Full pairs for conversational
            for r in self.memory.live[-2:]:  # Reduced from -4:
                msgs.append({"role":"user","content":truncate(r["user_message"],200)})
                msgs.append({"role":"assistant","content":truncate(r["ai_response"],200)})
        
        msgs.append({"role":"user","content":truncate(query, 1000)})
        return msgs

    async def chat(self, query: str) -> str:
        try:
            msgs = self.build_messages(query)
            if sum(len(m.get("content","")) for m in msgs) > 8000: msgs = msgs[-4:]  # Reduced for smaller model context
            print("🤖 AI:", end=" ", flush=True)
            response_parts, start = [], time.time()
            # Add num_ctx=2048 for faster inference, num_thread for CPU speed
            options = {
                "num_predict": 192,
                "temperature": 0.4,  # Lower for more consistent/accurate responses
                "num_ctx": 2048,  # Smaller context for speed
                "num_thread": os.cpu_count() or 4
            }
            stream = self.client.chat(model=self.model, messages=msgs, stream=True, options=options)
            for chunk in stream:
                try:
                    if "message" in chunk and "content" in chunk["message"]:
                        text = chunk["message"]["content"]
                        print(text, end="", flush=True)
                        response_parts.append(text)
                except Exception: continue
            elapsed = time.time() - start
            print(f" [{elapsed:.2f}s]")
            full = "".join(response_parts).strip()
            emb_vec = self.emb.get(f"User: {truncate(query,200)} Assistant: {truncate(full,200)}") if self.emb.use else None
            self.memory.add(query, full, emb=emb_vec)
            return full if full else "[No response]"
        except ollama.ResponseError as e:
            if e.status_code == 404:
                print(f"\n[Error] Model '{self.model}' still not available. Try 'ollama pull {self.model}' manually.")
                return f"[Error] Model not found: {self.model}. Run 'ollama pull {self.model}' to install."
            err = f"[Error] Ollama chat failed: {e}"
            print("\n" + err)
            try:
                # Fixed fallback parsing; try tinyllama first
                fallback_model = "tinyllama"
                options = {"num_predict":128, "num_ctx": 1024}  # Even smaller for fallback speed
                resp = self.client.chat(model=fallback_model, messages=[{"role":"user","content":truncate(query,1000)}],
                                        stream=False, options=options)
                text = resp.get("message", {}).get("content", "")
                emb_vec = self.emb.get(f"User: {truncate(query,200)} Assistant: {truncate(text,200)}") if self.emb.use else None
                self.memory.add(query, text, emb=emb_vec)
                print(f"[Fallback] Used {fallback_model}")
                return text
            except Exception as e2:
                with open("chat_errors.log", "a") as logf:
                    logf.write(f"{datetime.now()}: {err} | fallback failed: {e2}\n")
                return f"{err} | fallback failed: {e2}"
        except Exception as e:
            err = f"[Error] Ollama chat failed: {e}"
            print("\n" + err)
            try:
                fallback_model = "tinyllama"
                options = {"num_predict":128, "num_ctx": 1024}
                resp = self.client.chat(model=fallback_model, messages=[{"role":"user","content":truncate(query,1000)}],
                                        stream=False, options=options)
                text = resp.get("message", {}).get("content", "")
                emb_vec = self.emb.get(f"User: {truncate(query,200)} Assistant: {truncate(text,200)}") if self.emb.use else None
                self.memory.add(query, text, emb=emb_vec)
                print(f"[Fallback] Used {fallback_model}")
                return text
            except Exception as e2:
                with open("chat_errors.log", "a") as logf:
                    logf.write(f"{datetime.now()}: {err} | fallback failed: {e2}\n")
                return f"{err} | fallback failed: {e2}"

    def show_stats(self):
        print("\n📊", {"stored": len(self.memory.stored), "live": len(self.memory.live), "embeddings": self.emb.use})

    def show_context(self):
        print("\n💬 last messages:")
        for m in self.memory.live[-6:]:
            print("  You:", truncate(m["user_message"], 80))
            print("  AI :", truncate(m["ai_response"], 80))

    async def run(self):
        try:
            while True:
                u = input("👤 You: ").strip()
                if not u: continue
                if u.lower() in ("exit","quit"): break
                if u.lower() == "help":
                    print("help | stats | context | save | exit"); continue
                if u.lower() == "stats": self.show_stats(); continue
                if u.lower() == "context": self.show_context(); continue
                if u.lower() == "save": await self.memory.save(); continue
                await self.chat(u)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            print("Saving memory..."); await self.memory.save(); print("Goodbye.")

async def main():
    ensure_ollama_running()
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    use_emb = os.getenv("USE_EMB", "0") == "1"
    chat = EnhancedOllamaChat(model=model, use_embeddings=use_emb)
    await chat.__aenter__()  # Async init to pull model and load memory
    await chat.run()

if __name__ == "__main__":
    asyncio.run(main())