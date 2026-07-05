import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom/client";
import { GoogleGenAI, LiveServerMessage, Modality, Behavior, Type, ActivityHandling, StartSensitivity, EndSensitivity } from "@google/genai";
import { createClient } from "@supabase/supabase-js";
import {
  Send, BrainCircuit, RefreshCw, GraduationCap, Stethoscope,
  ChevronRight, Mic, X, Headphones, Sparkles, BookOpen,
  AlertCircle, ChevronDown, Trophy, LogOut, CheckCircle, XCircle,
  Flame, Target, Eye, EyeOff, Loader2
} from "lucide-react";

// ─────────────────────────────────────────────
// SUPABASE CLIENT
// ─────────────────────────────────────────────

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL as string,
  import.meta.env.VITE_SUPABASE_ANON_KEY as string
);

// Call the chat edge function using the anon key directly to avoid 401s from
// expired user JWTs. The function is stateless and doesn't use user identity.
async function invokeChat(body: { message: string; history: { role: string; text: string }[]; level: number }) {
  const resp = await fetch(
    `${import.meta.env.VITE_SUPABASE_URL as string}/functions/v1/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY as string}`,
        "apikey": import.meta.env.VITE_SUPABASE_ANON_KEY as string,
      },
      body: JSON.stringify(body),
    }
  );
  if (!resp.ok) throw new Error(`chat function HTTP ${resp.status}`);
  return resp.json() as Promise<{ response: string }>;
}

const RAG_URL = `${import.meta.env.VITE_SUPABASE_URL as string}/functions/v1/rag-query`;
const RAG_HEADERS = {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY as string}`,
  "apikey": import.meta.env.VITE_SUPABASE_ANON_KEY as string,
};

// Fast RAG for quiz generation — Edge Function, 8 s timeout.
async function queryLightRAGForQuiz(query: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(RAG_URL, {
      method: "POST",
      headers: RAG_HEADERS,
      body: JSON.stringify({ query, top_k: 40 }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = await res.json();
    return data.response ? (data.response as string) : null;
  } catch {
    clearTimeout(timer);
    return null;
  }
}

// Compact system instruction for quiz generation — keeps the key formatting rules
// without the full teaching prose so it fits in a lower-latency direct Gemini call.
const QUIZ_SYSTEM_INSTRUCTION = `You are a Plastic Surgery quiz question generator. Calibrate every question to the learner level specified by the [ACTIVE LEVEL] tag.

OUTPUT RULES — apply to every response:
- Do NOT use markdown tables (no pipe characters | for tabular data)
- Use ## headers and **bold** key terms where helpful
- Do not editorialize importance (no "crucial", "key", "important", "essential", "remember", "note that")

FOR RECALL QUESTIONS: Ask one direct factual question — no vignette. After the user responds, give detailed feedback: what was correct, what was missing, the right answer with reasoning.

FOR BOARD QUESTIONS — output format exactly:
1. Start directly with the clinical vignette (2–4 sentences). No title or heading before it.
2. Question stem on its own line.
3. Blank line.
4. Exactly 5 choices, each on its own line, labeled A through E. One correct answer only.
5. Last line must be exactly: CORRECT: [letter]  — machine-readable, no extra text on that line.

Calibrate topic, terminology, and difficulty to the active level:
- Medical Student: Direct recall — anatomy, definitions, blood supply, indications, etc
- Junior Resident: Applied clinical scenarios with patient factors, classification systems, one follow-up probe rooted in facts
- Senior Resident: Direct recall — clinical vignettes rooted in facts

Ground questions in [RAG CONTEXT] when provided. Do not fabricate citations or statistics.`;

// Quiz generation via the chat Edge Function — sends quiz instruction as part of the message.
async function callGeminiForQuiz(
  prompt: string,
  history: { role: string; text: string }[],
  _thinkingBudget: number
): Promise<string> {
  const quizMessage = `[QUIZ MODE]\n${QUIZ_SYSTEM_INSTRUCTION}\n\n${prompt}`;
  const res = await invokeChat({ message: quizMessage, history, level: 3 });
  const data = await res.json();
  return data?.reply ?? "";
}

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────

interface Message { role: "user" | "model"; text: string; isBoardQuestion?: boolean; }
type Level = 1 | 2 | 3;
type Page = "login" | "chat" | "leaderboard";
type QuizType = "recall" | "board";

interface UserProfile {
  id: string;
  username: string;
  total_points: number;
  correct_answers: number;
  total_questions: number;
  current_streak: number;
  best_streak: number;
  baseline_level: Level | null;
  preferred_level: Level | null;
  last_active: string;
  login_streak: number;
  last_login_date: string | null;  // ISO date string YYYY-MM-DD
}

// ─────────────────────────────────────────────
// LEVEL CONFIGURATION
// ─────────────────────────────────────────────

const BASE_POINTS = 10; // equal for all levels
const LEVEL_BONUS: Record<number, number> = { 0: 0, 1: 2, 2: 4 }; // bonus for answering above baseline

const AUDIO_TOPICS: { label: string; queries: string[] }[] = [
  { label: "General Overview", queries: [
    "plastic surgery history scope anatomy skin soft tissue principles",
    "wound healing flaps grafts vascularity tissue transfer reconstruction techniques",
    "aesthetic principles facial anatomy proportions operative planning outcomes",
    "plastic surgery complications infection hematoma necrosis postoperative management",
  ]},
  { label: "Basic Principles & Physiology", queries: [
    "wound healing phases collagen scar formation contraction biology",
    "skin anatomy dermis epidermis blood supply innervation physiology",
    "flaps classification blood supply delay phenomenon vascularity perforator",
    "skin grafts split thickness full thickness take failure infection nutrition",
  ]},
  { label: "Congenital Anomalies — Head & Neck", queries: [
    "cleft lip repair Millard rotation advancement technique unilateral bilateral",
    "cleft palate repair Furlow Bardach speech Eustachian tube feeding",
    "craniosynostosis craniofacial syndromes surgical correction fronto-orbital advancement",
    "microtia ear reconstruction Nagata rib cartilage hemangioma vascular malformation treatment",
  ]},
  { label: "Skin Cancer", queries: [
    "melanoma staging Breslow Clark level sentinel node biopsy wide excision margins",
    "basal cell carcinoma squamous cell carcinoma Mohs surgery excision margins recurrence",
    "skin cancer reconstruction local flap skin graft after Mohs defect closure",
    "Merkel cell carcinoma dermatofibrosarcoma rare skin tumors prognosis treatment",
  ]},
  { label: "Head & Neck Tumors", queries: [
    "oral cavity oropharyngeal cancer resection margins neck dissection levels",
    "free flap reconstruction fibula ALT radial forearm jaw mandible defect",
    "parotid gland tumor facial nerve superficial parotidectomy",
    "larynx hypopharynx cancer total laryngectomy reconstruction pharynx esophagus",
  ]},
  { label: "Maxillofacial Trauma", queries: [
    "mandible fracture classification treatment ORIF IMF condyle symphysis",
    "zygoma zygomatic arch orbital floor blowout fracture repair approaches",
    "naso-orbito-ethmoid NOE fracture telecanthus medial canthus repair",
    "panfacial fracture Le Fort midface maxillary fracture sequencing repair",
  ]},
  { label: "Trunk, Lower Extremity & Perineum", queries: [
    "pressure ulcer staging debridement flap coverage sacral ischial trochanteric",
    "lower extremity leg wound coverage free flap local flap Gustilo classification",
    "abdominal wall reconstruction component separation hernia mesh technique",
    "perineal reconstruction gynecologic oncology gracilis flap pelvic exenteration",
  ]},
  { label: "Burns", queries: [
    "burn classification depth TBSA rule of nines pathophysiology systemic response",
    "burn fluid resuscitation Parkland formula Brooke inhalation injury airway management",
    "burn wound escharotomy fasciotomy debridement topical agents silver infection",
    "burn skin grafting donor site management reconstruction contracture scar rehabilitation",
  ]},
  { label: "Hand", queries: [
    "hand anatomy intrinsic extrinsic tendons zones flexor extensor repair rehabilitation",
    "nerve repair digital nerve median ulnar radial carpal tunnel cubital tunnel",
    "replantation indications technique revascularization amputation microsurgery",
    "hand fracture dislocation scaphoid metacarpal phalanx fixation Dupuytren contracture",
  ]},
  { label: "Breast", queries: [
    "breast augmentation implant types plane subglandular submuscular dual plane complications",
    "breast reduction mastopexy techniques pedicle nipple areola blood supply ptosis",
    "breast reconstruction implant expander DIEP TRAM latissimus flap nipple reconstruction",
    "oncoplastic breast surgery margins volume displacement replacement oncology",
  ]},
  { label: "Aesthetic", queries: [
    "facelift SMAS platysma technique deep plane composite neck lift anatomy",
    "rhinoplasty open closed technique tip projection dorsum osteotomy cartilage grafts",
    "blepharoplasty upper lower eyelid browlift ptosis canthopexy anatomy fat",
    "liposuction fat grafting body contouring abdominoplasty technique complications",
  ]},
  { label: "Gender Affirmation & Perineal Aesthetic", queries: [
    "phalloplasty radial forearm ALT technique urethroplasty complications outcomes",
    "vaginoplasty penile inversion peritoneal technique dilation complications",
    "facial feminization masculinization forehead jaw tracheal shave rhinoplasty",
    "labiaplasty perineal aesthetic clitoral hood reduction technique outcomes",
  ]},
  { label: "Patient Safety", queries: [
    "DVT pulmonary embolism VTE prophylaxis risk stratification plastic surgery",
    "anesthesia local regional general sedation complications airway management",
    "surgical site infection hematoma seroma prevention management antibiotics",
    "patient selection comorbidities BMI smoking diabetes risk assessment outcomes",
  ]},
];

const KNOWLEDGE_BASE_TOOL = {
  functionDeclarations: [{
    name: "search_knowledge_base",
    description: "Search the plastic surgery knowledge base (textbook chapters and cited journal articles) for specific clinical information. Use this when the pre-loaded context does not contain sufficient detail to answer a question — for example, specific dosing, classification criteria, operative steps, outcomes data, or evidence from a particular study.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        query: {
          type: Type.STRING,
          description: "Precise clinical search query using medical terminology, e.g. 'Parkland formula fluid resuscitation calculation burns' or 'TRAM flap zone perfusion necrosis risk'",
        },
      },
      required: ["query"],
    },
    behavior: Behavior.NON_BLOCKING,
  }],
};

const LEVELS = {
  1: {
    name: "Medical Student", short: "MS", icon: "🩺",
    description: "Core anatomy, foundational concepts, and essential principles — no jargon assumed.",
    welcomeMsg: "🩺 Welcome, Medical Student! I'll teach at a foundational level — clear explanations, defined terminology, and memorable mnemonics. Ask me anything or say \"quiz me\" to get started.",
  },
  2: {
    name: "Junior Resident", short: "PGY 1–3", icon: "🔬",
    description: "Clinical decision-making, operative planning, and complication management for early residents.",
    welcomeMsg: "🔬 Welcome, Junior Resident! I'll focus on clinical reasoning, surgical planning, and hands-on technique. Ask about cases, classifications, or say \"quiz me\" for a clinical vignette.",
  },
  3: {
    name: "Senior Resident", short: "PGY 4+", icon: "🏆",
    description: "Advanced techniques, outcomes data, surgical controversy, and oral board preparation.",
    welcomeMsg: "🏆 Welcome, Senior Resident. We'll engage at the level of a fellow or attending — evidence-based nuance, complex cases, technical pearls, and board-style cases. Challenge me.",
  },
} as const;

// System instruction lives in supabase/functions/chat/index.ts — not in this bundle.

// ─────────────────────────────────────────────
// AUDIO HELPERS
// ─────────────────────────────────────────────

function downsampleBuffer(buffer: Float32Array, inputRate: number, targetRate: number): Float32Array {
  if (inputRate <= targetRate) return buffer;
  const ratio = inputRate / targetRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const ni = i * ratio;
    const idx = Math.floor(ni);
    const frac = ni - idx;
    const next = idx + 1 < buffer.length ? buffer[idx + 1] : buffer[idx];
    result[i] = buffer[idx] * (1 - frac) + next * frac;
  }
  return result;
}

function pcmToGenAIBlob(data: Float32Array): { data: string; mimeType: string } {
  const int16 = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return { data: encodeB64(new Uint8Array(int16.buffer)), mimeType: "audio/pcm;rate=16000" };
}

function encodeB64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function decodeB64(base64: string): Uint8Array {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = buffer.getChannelData(ch);
    for (let i = 0; i < frameCount; i++) channelData[i] = dataInt16[i * numChannels + ch] / 32768.0;
  }
  return buffer;
}

// ─────────────────────────────────────────────
// RAG HELPERS (via Supabase Edge Functions)
// ─────────────────────────────────────────────

async function queryLightRAG(query: string): Promise<string | null> {
  console.group("%c[RAG] Query", "color: #6366f1; font-weight: bold;");
  console.log("Query:", query);
  const t0 = performance.now();
  const doFetch = () => fetch(RAG_URL, {
    method: "POST",
    headers: RAG_HEADERS,
    body: JSON.stringify({ query, top_k: 80, entity_k: 20, relation_k: 20, graph_k: 40 }),
  });
  try {
    let res = await doFetch();
    if (res.status >= 500) {
      console.warn(`%c[RAG] HTTP ${res.status} — retrying in 2 s`, "color: #f59e0b;");
      await new Promise(r => setTimeout(r, 2000));
      res = await doFetch();
    }
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    if (!res.ok) {
      console.warn(`%c[RAG] HTTP ${res.status}`, "color: #f59e0b;");
      console.groupEnd();
      return null;
    }
    const data = await res.json();
    const ctx: string | null = data?.response || null;
    if (!ctx) {
      console.warn("%c[RAG] No context found", "color: #f59e0b;");
      console.groupEnd();
      return "";
    }
    console.log(`%c[RAG] Context retrieved in ${elapsed}s`, "color: #10b981; font-weight: bold;");
    console.log(`Context length: ${ctx.length} chars`);
    console.log("Context preview:", ctx.slice(0, 300) + (ctx.length > 300 ? "…" : ""));
    console.groupEnd();
    return ctx;
  } catch (e) {
    console.error("%c[RAG] Edge Function unreachable", "color: #ef4444;", e);
    console.groupEnd();
    return null;
  }
}

// Fast query for audio tool calls — Edge Function, 8-second hard abort.
async function queryLightRAGFast(query: string): Promise<string | null> {
  const t0 = performance.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(RAG_URL, {
      method: "POST",
      headers: RAG_HEADERS,
      body: JSON.stringify({ query, top_k: 30 }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = await res.json();
    const ctx = data.response ? (data.response as string).slice(0, 20000) : null;
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    console.log(`%c[Audio Tool] RAG returned in ${elapsed}s (${ctx?.length ?? 0} chars)`, "color: #10b981;");
    return ctx;
  } catch (e: any) {
    clearTimeout(timer);
    if (e?.name === "AbortError") console.warn("[Audio Tool] RAG timed out after 8s");
    else console.error("[Audio Tool] RAG error", e);
    return null;
  }
}

// Get Gemini API key for audio mode — Edge Function checks auth, returns key.
// Uses anon key for Authorization (never expires) — same pattern as invokeChat().
async function getGeminiToken(): Promise<string> {
  const res = await fetch(
    `${import.meta.env.VITE_SUPABASE_URL as string}/functions/v1/gemini-token`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY as string}`,
        "apikey": import.meta.env.VITE_SUPABASE_ANON_KEY as string,
      },
    }
  );
  if (!res.ok) throw new Error(`Audio token HTTP ${res.status}`);
  const data = await res.json();
  if (!data?.token) throw new Error("Failed to get audio token");
  return data.token as string;
}

// ─────────────────────────────────────────────
// LEVEL SWITCH DETECTION
// ─────────────────────────────────────────────

function detectLevelSwitch(text: string): Level | null {
  const l = text.toLowerCase();
  if (/level\s*1\b|medical\s*stud|make\s*it\s*(easier|simpler)|simplif|too\s*(hard|advanced)|go\s*easier|basic(s)?\b/.test(l)) return 1;
  if (/level\s*2\b|junior\s*res|pgy\s*[1-3]\b|intermediate/.test(l)) return 2;
  if (/level\s*3\b|senior\s*res|pgy\s*[4-9]\b|make\s*it\s*harder|more\s*advanced|too\s*(easy|basic)|fellowship/.test(l)) return 3;
  return null;
}

// ─────────────────────────────────────────────
// APP
// ─────────────────────────────────────────────

const App = () => {
  // ── Auth & Page ──
  const [page, setPage] = useState<Page>("login");
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [authLoading, setAuthLoading] = useState(true); // true while checking existing session

  // ── Chat ──
  const [material, setMaterial] = useState(`# Introduction to Plastic Surgery

Plastic surgery is a surgical specialty involving the restoration, reconstruction, or alteration of the human body. It can be divided into two main categories:
1. Reconstructive Surgery: Corrects defects to restore function and normal appearance (e.g., burn repair, cleft lip).
2. Cosmetic (Aesthetic) Surgery: Focuses on enhancing appearance (e.g., rhinoplasty, liposuction).

Key Principles:
- Tissue handling and preservation of blood supply.
- Tension-free closure.
- Respect for anatomical planes.

Common Procedures:
- Skin Grafts: Transfer of skin without its blood supply.
- Flaps: Transfer of tissue with its own blood supply.
- Z-plasty: A technique to lengthen a contracted scar or change its direction.`);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [quizActive, setQuizActive] = useState(false);
  const [quizType, setQuizType] = useState<QuizType | null>(null);
  const [boardCorrectAnswer, setBoardCorrectAnswer] = useState<string | null>(null);
  const [quizAnswered, setQuizAnswered] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [showNextBoardBtn, setShowNextBoardBtn] = useState(false);
  const [recallResponded, setRecallResponded] = useState(false);
  const [showFollowUpButtons, setShowFollowUpButtons] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showBaselineMenu, setShowBaselineMenu] = useState(false);
  const [showMaterialMenu, setShowMaterialMenu] = useState(false);
  const [showAudioTopicMenu, setShowAudioTopicMenu] = useState(false);

  const speechRecognitionRef = useRef<any>(null);

  // ── Level ──
  const [level, setLevel] = useState<Level | null>(null);
  const levelRef = useRef<Level | null>(null);

  // ── Audio ──
  const [isAudioMode, setIsAudioMode] = useState(false);
  const [audioStatus, setAudioStatus] = useState<"disconnected" | "connecting" | "active" | "error">("disconnected");
  const [audioError, setAudioError] = useState("");
  const [audioStatusDetail, setAudioStatusDetail] = useState("");
  const [audioTopic, setAudioTopic] = useState(AUDIO_TOPICS[0]);
  const [micVolume, setMicVolume] = useState(0);
  const [isSearchingKB, setIsSearchingKB] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputCtxRef = useRef<AudioContext | null>(null);
  const outputCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nextStartTimeRef = useRef(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const audioGenRef = useRef(0);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioSessionRef = useRef<any>(null);

  useEffect(() => { levelRef.current = level; }, [level]);

  // ── Daily login streak helper ──
  const updateLoginStreak = async (profile: UserProfile, userId: string): Promise<UserProfile> => {
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const last = profile.last_login_date;
    if (last === today) return profile; // already logged in today, no change

    const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
    const newStreak = last === yesterday ? (profile.login_streak ?? 0) + 1 : 1;
    const updates = { login_streak: newStreak, last_login_date: today };
    await supabase.from("user_progress").update(updates).eq("id", userId);
    return { ...profile, ...updates };
  };

  // ── Restore session on mount ──
  useEffect(() => {
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      if (session) {
        // Verify the session is still valid by checking the user
        const { data: { user }, error: userErr } = await supabase.auth.getUser();
        if (userErr || !user) {
          // Stale/expired session — clear it so the login page works cleanly
          await supabase.auth.signOut();
          setAuthLoading(false);
          return;
        }
        const displayName = user.user_metadata?.display_name || user.email?.split("@")[0] || "User";
        let { data: profile } = await supabase
          .from("user_progress")
          .select("*")
          .eq("id", user.id)
          .maybeSingle();
        if (!profile) {
          await supabase.from("user_progress").insert({
            id: user.id,
            username: displayName,
            total_points: 0, correct_answers: 0, total_questions: 0,
            current_streak: 0, best_streak: 0,
            login_streak: 1, last_login_date: new Date().toISOString().slice(0, 10),
          });
          const { data: refetched } = await supabase
            .from("user_progress").select("*").eq("id", user.id).maybeSingle();
          profile = refetched;
        }
        if (profile) {
          const updated = await updateLoginStreak(profile as UserProfile, user.id);
          setCurrentUser(updated.username);
          setUserProfile(updated);
          if (updated.preferred_level) setLevel(updated.preferred_level);
        } else {
          setCurrentUser(displayName);
          setUserProfile({
            id: user.id, username: displayName,
            total_points: 0, correct_answers: 0, total_questions: 0,
            current_streak: 0, best_streak: 0, baseline_level: null, preferred_level: null,
            last_active: new Date().toISOString(), login_streak: 1, last_login_date: new Date().toISOString().slice(0, 10),
          });
        }
        setPage("chat");
      }
      setAuthLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) {
        setCurrentUser(null);
        setUserProfile(null);
        setPage("login");
      }
    });

    return () => {
      subscription.unsubscribe();
      stopAudioSession();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Close any open dropdown on outside click
  useEffect(() => {
    const anyOpen = showUserMenu || showBaselineMenu || showMaterialMenu || showAudioTopicMenu;
    if (!anyOpen) return;
    const handler = () => { setShowUserMenu(false); setShowBaselineMenu(false); setShowMaterialMenu(false); setShowAudioTopicMenu(false); };
    window.addEventListener("click", handler);
    return () => window.removeEventListener("click", handler);
  }, [showUserMenu, showBaselineMenu, showMaterialMenu, showAudioTopicMenu]);

  // ─────────────────────────────────────────────
  // AUTH
  // ─────────────────────────────────────────────

  // Accepts "drsmith" (username) OR "drsmith@gmail.com" (email).
  // If an email is provided, the local part (before @) becomes the display name.
  const parseInput = (input: string): { authEmail: string; displayName: string } => {
    const trimmed = input.trim();
    if (trimmed.includes("@")) {
      return { authEmail: trimmed.toLowerCase(), displayName: trimmed.split("@")[0] };
    }
    return { authEmail: `${trimmed.toLowerCase()}@pstutor.app`, displayName: trimmed };
  };

  const signIn = async (username: string, password: string): Promise<string | null> => {
    if (!username.trim() || !password.trim()) return "Username and password are required.";
    const { authEmail } = parseInput(username);
    const { data, error } = await supabase.auth.signInWithPassword({ email: authEmail, password });
    if (error) return "Incorrect username or password.";
    const displayName = data.user.user_metadata?.display_name || authEmail.split("@")[0];
    let { data: profile } = await supabase
      .from("user_progress").select("*").eq("id", data.user.id).maybeSingle();
    if (!profile) {
      // First login — admin created the account but no profile row exists yet
      await supabase.from("user_progress").insert({
        id: data.user.id,
        username: displayName,
        total_points: 0, correct_answers: 0, total_questions: 0,
        current_streak: 0, best_streak: 0,
        login_streak: 1, last_login_date: new Date().toISOString().slice(0, 10),
      });
      const { data: refetched } = await supabase
        .from("user_progress").select("*").eq("id", data.user.id).maybeSingle();
      profile = refetched;
    }
    if (profile) {
      const updated = await updateLoginStreak(profile as UserProfile, data.user.id);
      setCurrentUser(updated.username);
      setUserProfile(updated);
      if (updated.preferred_level) setLevel(updated.preferred_level);
      await supabase.from("user_progress").update({ last_active: new Date().toISOString() }).eq("id", data.user.id);
    } else {
      // Profile creation failed (e.g. username conflict) — proceed with defaults
      setCurrentUser(displayName);
      setUserProfile({
        id: data.user.id, username: displayName,
        total_points: 0, correct_answers: 0, total_questions: 0,
        current_streak: 0, best_streak: 0, baseline_level: null, preferred_level: null,
        last_active: new Date().toISOString(), login_streak: 1, last_login_date: new Date().toISOString().slice(0, 10),
      });
    }
    setPage("chat");
    return null;
  };

  const signUp = async (name: string, email: string): Promise<string | null> => {
    if (!name.trim()) return "Name is required.";
    if (name.trim().length < 3) return "Name must be at least 3 characters.";
    if (!email.trim()) return "Email is required.";
    if (!email.includes("@")) return "Please enter a valid email address.";

    const { error } = await supabase.from("account_requests").insert({
      display_name: name.trim(),
      email: email.trim().toLowerCase(),
    });
    if (error) return "Failed to submit request. Please try again.";

    return "SUCCESS";
  };

  const handleLogout = async () => {
    stopAudioSession();
    await supabase.auth.signOut();
    setCurrentUser(null);
    setUserProfile(null);
    setPage("login");
    setLevel(null);
    levelRef.current = null;
    setMessages([]);
    setQuizActive(false);
    setBoardCorrectAnswer(null);
    setQuizAnswered(false);
    setSelectedAnswer(null);
    setShowNextBoardBtn(false);
    setRecallResponded(false);
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.abort();
      speechRecognitionRef.current = null;
    }
    setIsRecording(false);
  };

  // ─────────────────────────────────────────────
  // SPEECH-TO-TEXT
  // ─────────────────────────────────────────────

  const toggleRecording = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;

    if (isRecording) {
      speechRecognitionRef.current?.stop();
      speechRecognitionRef.current = null;
      setIsRecording(false);
      return;
    }

    const recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (e: any) => {
      let transcript = "";
      for (let i = 0; i < e.results.length; i++) {
        transcript += e.results[i][0].transcript;
      }
      setInputText(transcript);
    };

    recognition.onerror = () => {
      speechRecognitionRef.current = null;
      setIsRecording(false);
    };

    recognition.onend = () => {
      speechRecognitionRef.current = null;
      setIsRecording(false);
    };

    speechRecognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  };

  // ─────────────────────────────────────────────
  // SCORE TRACKING
  // ─────────────────────────────────────────────

  const scoreAnswer = async (correct: boolean, deactivateQuiz = true) => {
    if (!userProfile) return;
    const activeLevel = levelRef.current ?? 1;
    const baselineLevel = userProfile.baseline_level ?? activeLevel;
    const levelDiff = Math.min(2, Math.max(0, activeLevel - baselineLevel));
    const basePoints = correct ? BASE_POINTS : 0;
    const bonusPoints = correct ? (LEVEL_BONUS[levelDiff] ?? 0) : 0;
    const newStreak = correct ? userProfile.current_streak + 1 : 0;
    const streakBonus = correct ? Math.floor(newStreak / 3) * 5 : 0;
    const totalPts = basePoints + bonusPoints + streakBonus;

    const updates = {
      total_points: userProfile.total_points + totalPts,
      correct_answers: userProfile.correct_answers + (correct ? 1 : 0),
      total_questions: userProfile.total_questions + 1,
      current_streak: newStreak,
      best_streak: Math.max(userProfile.best_streak, newStreak),
      preferred_level: activeLevel,
      last_active: new Date().toISOString(),
    };

    await supabase.from("user_progress").update(updates).eq("id", userProfile.id);
    setUserProfile(prev => prev ? { ...prev, ...updates } : null);
    if (deactivateQuiz) {
      setQuizActive(false);
      setQuizAnswered(false);
      setSelectedAnswer(null);
      setRecallResponded(false);
    }
  };

  // Auto-score when user clicks a board answer choice
  const handleAnswerClick = async (letter: string) => {
    if (!level || quizAnswered) return;
    const isCorrect = boardCorrectAnswer !== null && letter === boardCorrectAnswer;
    setSelectedAnswer(letter);
    setQuizAnswered(true);
    await scoreAnswer(isCorrect, false); // score now, deactivate after explanation

    const activeLevel = levelRef.current ?? 1;
    // Capture history before adding the answer message
    const historySnapshot = messages.slice(1).map(m => ({ role: m.role, text: m.text }));
    setMessages(prev => [...prev, { role: "user", text: `My answer: ${letter}` }]);
    setIsLoading(true);
    try {
      const prompt =
        `[ACTIVE LEVEL: ${activeLevel} — ${LEVELS[activeLevel].name}]\n\n` +
        `[USER ANSWER]\nThe user selected answer ${letter}. ` +
        (boardCorrectAnswer ? `The correct answer is ${boardCorrectAnswer}. ` : "") +
        `Please now provide a complete explanation:\n` +
        `1. State clearly whether ${letter} is correct or incorrect.\n` +
        `2. Explain in depth why the correct answer is right, with clinical/anatomical reasoning.\n` +
        `3. Briefly explain why each of the other options is incorrect.\n` +
        `Do NOT include summary tables, mnemonics, LaTeX, or "boards memory tip" sections at the end.\n` +
        `Do NOT suggest asking another question or say anything like "ready for the next one" — end after the explanation.`;
      let data: { response: string } | null = null;
      try { data = await invokeChat({ message: prompt, history: historySnapshot, level: activeLevel }); } catch { data = null; }
      if (!data?.response) {
        setMessages(prev => [...prev, { role: "model", text: "Error getting explanation." }]);
      } else {
        setMessages(prev => [...prev, { role: "model", text: data.response as string }]);
        setShowNextBoardBtn(true);
      }
    } catch {
      setMessages(prev => [...prev, { role: "model", text: "Error getting explanation." }]);
    } finally {
      setIsLoading(false);
      setQuizActive(false);
    }
  };

  // Parse a board question message into pre-text, A-E choices, and post-text
  const splitBoardMessage = (text: string) => {
    const lines = text.split("\n");
    const pre: string[] = [], post: string[] = [];
    const choices: { letter: string; text: string }[] = [];
    let inChoices = false, pastChoices = false;
    for (const line of lines) {
      const m = line.trim().match(/^([A-E])\.\s+(.+)$/);
      if (m && !pastChoices) {
        inChoices = true;
        choices.push({ letter: m[1], text: m[2] });
      } else if (inChoices && !pastChoices) {
        pastChoices = true;
        post.push(line);
      } else if (!inChoices) {
        pre.push(line);
      } else {
        post.push(line);
      }
    }
    return { pre: pre.join("\n"), choices, post: post.join("\n").trim() };
  };

  // ─────────────────────────────────────────────
  // CHAT
  // ─────────────────────────────────────────────

  const handleSelectLevel = async (selectedLevel: Level) => {
    setLevel(selectedLevel);
    levelRef.current = selectedLevel;
    setMessages([{ role: "model", text: LEVELS[selectedLevel].welcomeMsg }]);
    setQuizActive(false);
    setBoardCorrectAnswer(null);
    setQuizAnswered(false);
    setSelectedAnswer(null);
    setShowNextBoardBtn(false);
    setRecallResponded(false);
    if (speechRecognitionRef.current) {
      speechRecognitionRef.current.abort();
      speechRecognitionRef.current = null;
    }
    setIsRecording(false);
    if (userProfile) {
      // First level selection ever → also set as baseline
      const isFirstSelection = userProfile.baseline_level === null;
      const updates: Partial<UserProfile> = { preferred_level: selectedLevel };
      if (isFirstSelection) updates.baseline_level = selectedLevel;
      await supabase.from("user_progress").update(updates).eq("id", userProfile.id);
      setUserProfile(prev => prev ? { ...prev, ...updates } : null);
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || !level) return;
    const userMsg = inputText;
    // Capture history before adding the new user message (messages is the old state in this closure)
    const historySnapshot = messages.slice(1).map(m => ({ role: m.role, text: m.text }));
    setInputText("");
    setShowFollowUpButtons(false);
    setMessages(prev => [...prev, { role: "user", text: userMsg }]);
    setIsLoading(true);
    const wasQuizActive = quizActive;

    try {
      const switchedLevel = detectLevelSwitch(userMsg);
      let activeLevel = levelRef.current ?? 1;
      if (switchedLevel && switchedLevel !== activeLevel) {
        setLevel(switchedLevel);
        levelRef.current = switchedLevel;
        activeLevel = switchedLevel;
      }

      const ragContext = await queryLightRAG(userMsg);
      if (ragContext === null) {
        setMessages(prev => [...prev, { role: "model", text: "⚠️ The knowledge base is currently unavailable. Please try again." }]);
        setQuizActive(false);
        return;
      }
      if (ragContext === "") {
        setMessages(prev => [...prev, { role: "model", text: "This answer was not found in my knowledge base. Please refer to other verified resources." }]);
        setQuizActive(false);
        return;
      }

      const cappedContext = ragContext.length > 120_000 ? ragContext.slice(0, 120_000) : ragContext;
      const augmented = wasQuizActive && quizType === "recall"
        ? `[ACTIVE LEVEL: ${activeLevel} — ${LEVELS[activeLevel].name}]\n\n[RAG CONTEXT]\n${cappedContext}\n\n[TASK]\nThe student was asked a recall question. Evaluate their answer below against the knowledge base. Tell them what they got right, what they missed or got wrong, and provide the complete correct answer.\n\n[STUDENT ANSWER]\n${userMsg}`
        : `[ACTIVE LEVEL: ${activeLevel} — ${LEVELS[activeLevel].name}]\n\n[RAG CONTEXT]\n${cappedContext}\n\n[USER QUESTION]\n${userMsg}`;
      console.log("%c[Chat] Sending to Edge Function", "color: #6366f1; font-weight: bold;", { level: activeLevel, contextChars: cappedContext.length, question: userMsg });
      let data: { response: string } | null = null;
      try { data = await invokeChat({ message: augmented, history: historySnapshot, level: activeLevel }); } catch { data = null; }
      if (!data?.response) {
        setMessages(prev => [...prev, { role: "model", text: "I encountered an error. Please try again." }]);
      } else {
        const responseText = data.response as string;
        console.log("%c[Chat] Response received", "color: #10b981; font-weight: bold;", { responseChars: responseText.length });
        setMessages(prev => [...prev, { role: "model", text: responseText }]);
        if (wasQuizActive) {
          setQuizActive(true);
          if (quizType === "recall") setRecallResponded(true);
        } else {
          setShowFollowUpButtons(true);
        }
      }
    } catch {
      setMessages(prev => [...prev, { role: "model", text: "I encountered an error. Please try again." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }
  };

  const startQuiz = async (type: QuizType) => {
    if (!level) return;
    const activeLevel = levelRef.current ?? level;

    // Reset board/recall state
    setShowFollowUpButtons(false);
    setBoardCorrectAnswer(null);
    setQuizAnswered(false);
    setSelectedAnswer(null);
    setShowNextBoardBtn(false);
    setRecallResponded(false);

    const recallPrompt =
      `[ACTIVE LEVEL: ${activeLevel} — ${LEVELS[activeLevel].name}]\n\n` +
      `[USER QUESTION]\nAsk me one concise recall question appropriate for my level — ` +
      `no clinical vignette, just a direct factual question (e.g. "What is the blood supply to the TRAM flap?"). ` +
      `After I respond, give detailed feedback.`;

    const boardPrompt =
      `[ACTIVE LEVEL: ${activeLevel} — ${LEVELS[activeLevel].name}]\n\n` +
      `[USER QUESTION]\nAsk me one board-style multiple choice question. Requirements:\n` +
      `1. Do NOT include any title, heading, or label at the top — start directly with the clinical vignette.\n` +
      `2. A clinical vignette (2-4 sentences): patient age/sex, presenting scenario, relevant history and findings.\n` +
      `3. A clear question stem on its own line.\n` +
      `4. Exactly 5 answer choices, each on its own line, labeled A through E — only one is correct.\n` +
      `5. Leave a blank line between the question stem and the choices.\n` +
      `6. On the very last line, write exactly: CORRECT: [letter] — where [letter] is only A, B, C, D, or E. This line is machine-readable.\n` +
      `Calibrate topic and difficulty to my active level.`;

    const userLabel = type === "board" ? "Give me a clinical vignette MCQ question." : "Quiz me! (simple recall)";

    // Capture history before adding the quiz label message
    const historySnapshot = messages.slice(1).map(m => ({ role: m.role, text: m.text }));
    setMessages(prev => [...prev, { role: "user", text: userLabel }]);
    setIsLoading(true);
    setQuizActive(false);
    setQuizType(null);
    try {
      // Query LightRAG using the last user message as topic context
      const lastUserMsg = [...messages].reverse().find(m => m.role === "user")?.text ?? "plastic surgery anatomy techniques";
      const ragContext = await queryLightRAG(lastUserMsg);
      if (ragContext === null) {
        setMessages(prev => [...prev, { role: "model", text: "⚠️ The knowledge base is currently unavailable. Please try again." }]);
        setIsLoading(false);
        return;
      }
      const cappedRag = ragContext.length > 120_000 ? ragContext.slice(0, 120_000) : ragContext;
      const ragPrefix = cappedRag ? `[RAG CONTEXT — base your question on this material]\n${cappedRag}\n\n` : "";
      const prompt = ragPrefix + (type === "board" ? boardPrompt : recallPrompt);
      let chatData: { response: string } | null = null;
      try { chatData = await invokeChat({ message: prompt, history: historySnapshot, level: activeLevel }); } catch { chatData = null; }
      if (!chatData?.response) {
        setMessages(prev => [...prev, { role: "model", text: "Error starting quiz." }]);
      } else {
        const responseText = chatData.response as string;
        if (type === "board") {
          // Parse and strip the CORRECT: X marker from the displayed text
          const correctMatch = responseText.match(/^CORRECT:\s*([A-E])\s*$/m);
          const cleanText = responseText.replace(/^CORRECT:\s*[A-E]\s*$/m, "").trimEnd();
          setBoardCorrectAnswer(correctMatch ? correctMatch[1] : null);
          setMessages(prev => [...prev, { role: "model", text: cleanText, isBoardQuestion: true }]);
        } else {
          setMessages(prev => [...prev, { role: "model", text: responseText }]);
        }
        setQuizActive(true);
        setQuizType(type);
      }
    } catch {
      setMessages(prev => [...prev, { role: "model", text: "Error starting quiz." }]);
    } finally {
      setIsLoading(false);
    }
  };

  // ─────────────────────────────────────────────
  // AUDIO
  // ─────────────────────────────────────────────

  const startAudioSession = async () => {
    // Ensure any previous session is fully torn down before starting a new one
    stopAudioSession(false);
    const activeLevel = levelRef.current;
    setIsAudioMode(true); setAudioStatus("connecting"); setAudioError(""); setAudioStatusDetail("Fetching knowledge base...");
    try {
      // Fetch RAG context first, then get token with session config baked in
      setAudioStatusDetail("Fetching knowledge base...");
      const ragResults = await Promise.all(audioTopic.queries.map(q => queryLightRAG(q)));
      const ragCtxParts = ragResults.filter((r): r is string => !!r);
      const ragCtx = ragCtxParts.length > 0 ? ragCtxParts.join("\n\n---\n\n") : null;
      const knowledgeBase = ragCtx
        ? ragCtx.slice(0, 120000)
        : material;
      const ragSource = ragCtx ? "RAG knowledge base (Grabb & Smith)" : "built-in overview (RAG unavailable)";
      console.log(`%c[Audio] Knowledge source: ${ragSource}`, "color: #10b981; font-weight: bold;", { chars: knowledgeBase.length });

      setAudioStatusDetail("Authorizing audio session...");
      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputCtxRef.current = inputCtx; outputCtxRef.current = outputCtx; nextStartTimeRef.current = 0; audioGenRef.current = 0;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      streamRef.current = stream;
      const levelInstructions: Record<number, string> = {
        1: `LEARNER LEVEL: Medical Student

Teaching focus:
- Define every technical term the first time it is used
- Explain the underlying anatomy or physiology before the clinical point
- Focus on anatomy, definitions, blood supply, indications
- Teach through pattern recognition: what does this look like, why does it happen
- Use simple analogies and memorable mnemonics
- Limit each answer to 3–5 key points — do not overload
- End with recall questions to reinforce learning

Quiz/testing focus (when testing this learner):
- Ask direct recall questions: names, definitions, anatomy, blood supply, indications
- One concept per question — keep it simple and unambiguous
- After a correct answer, reinforce briefly with one related fact
- After an incorrect answer, correct gently, give the right answer, explain why in one sentence
- Example style: "What is the name of the muscle that depresses the lower lip?", "Name three zones of the hand", "What is the gold standard for fingertip reconstruction in a child?"

Do NOT assume: operative exposure, clinical autonomy, or familiarity with classifications`,

        2: `LEARNER LEVEL: Junior Resident (PGY 1–3)

Teaching focus:
- Skip basics — the resident knows anatomy and definitions
- Lead with clinical facts and the decision that matters
- Walk through preoperative planning: patient factors, risk stratification, workup needed
- Present applied scenarios with patient factors (BMI, smoking, prior surgery, radiation)
- Cover classification systems concisely — name, tiers, and what changes with each tier
- Give stepwise operative overview when asked about a procedure
- Mention early complication recognition and first-line management
- Include one or two clinical pearls that separate good from average operative planning
- Probe one level deeper — follow up with a question that tests reasoning

Quiz/testing focus (when testing this learner):
- Present applied clinical scenarios requiring recall
- Include patient factors that change the answer (BMI, smoking, prior surgery, radiation)
- Ask about classification systems: name the system, classify this injury, what changes management
- After an answer, probe one level deeper

Do NOT: define common terms, explain basic science, hedge with excessive caveats`,

        3: `LEARNER LEVEL: Senior Resident (PGY 4+)

Teaching focus:
- Peer-to-peer: this is an attending-to-fellow discussion, not a teaching session
- Go directly to nuance of topics — what separates good from great
- Assume complete command of anatomy, physiology, flap principles, and standard technique
- Present clinical vignettes rooted in facts
- Discuss edge cases: the obese patient, the irradiated field, the revision, the failed flap
- Address failure modes and salvage strategy — what goes wrong and how you recover
- Highlight genuine controversy or evolving evidence; state your view and why
- If a complication is mentioned, go straight to management algorithm

Quiz/testing focus (when testing this learner):
- Clinical vignettes rooted in facts
- After an answer, probe two levels deeper
- Example style: "You are 4 days post–free fibula jaw reconstruction. The patient spikes a fever and the neck looks tense. Walk me through your next 30 minutes."

Do NOT: define terms, re-explain standard technique, give background context, use preamble`,
      };
      const levelNote = activeLevel ? levelInstructions[activeLevel] : "";

      // Full system instruction with knowledge base — sent via ai.live.connect() config.
      // v1beta with real API key properly respects systemInstruction (unlike ephemeral tokens on v1alpha).
      const audioSystemInstruction = `You are a Plastic Surgery attending giving a audio tutorial. This is spoken audio — be direct and conversational. No markdown, no lists, no headers. Answer immediately without restating the question or adding preamble. Stop when you have made the point.

WORD BAN: Never use these words or their variants: crucial, critical, meticulous, important, essential, vital, key, significant, noteworthy, remember, note that, it is worth, bear in mind, always, never (as emphasis), comprehensive, thorough, careful, proper, appropriate, ensure, make sure. State the fact plainly — do not editorialize its importance.

PRONUNCIATION GUIDE — say these exactly as written phonetically:
- platysmaplasty → "plat-IZ-mah-plas-tee"
- platysma → "plat-IZ-mah"
- rhytidectomy → "rit-ih-DEK-toh-mee"
- blepharoplasty → "BLEF-ar-oh-plas-tee"
- mentoplasty → "MEN-toh-plas-tee"
- genioplasty → "JEE-nee-oh-plas-tee"
- mastopexy → "MAS-toh-pek-see"
- mammaplasty → "MAM-ah-plas-tee"
- abdominoplasty → "ab-DOM-ih-noh-plas-tee"
- fasciocutaneous → "fash-ee-oh-kyoo-TAY-nee-us"
- SMAS → say each letter "S-M-A-S"
- ptosis → "TOH-sis"
- canthopexy → "KAN-thoh-pek-see"
- canthoplasty → "KAN-thoh-plas-tee"
- ectropion → "ek-TROH-pee-on"
- entropion → "en-TROH-pee-on"
- lagophthalmos → "lag-off-THAL-mos"
- osteotomy → "os-tee-OT-oh-mee"
- septorhinoplasty → "sep-toh-RY-noh-plas-tee"
- cheiloplasty → "KY-loh-plas-tee"
- palatoplasty → "PAL-at-oh-plas-tee"
- otoplasty → "OH-toh-plas-tee"
- philtrum → "FIL-trum"
- columella → "kol-yoo-MEL-ah"
- alar → "AY-lar"
- nasolabial → "nay-zoh-LAY-bee-al"
- malar → "MAY-lar"
- zygoma → "zy-GOH-mah"
- zygomatic → "zy-goh-MAT-ik"
- pogonion → "poh-GOH-nee-on"
- gnathion → "NAY-thee-on"
- tragus → "TRAY-gus"
- lobule → "LOB-yool"
- helix → "HEE-liks"
- antihelix → "an-tee-HEE-liks"
- concha → "KONG-kah"
- gynecomastia → "GY-neh-koh-MAS-tee-ah"
- liposuction → "LIP-oh-suk-shun"
- tumescent → "too-MES-ent"
- seroma → "seh-ROH-mah"
- hematoma → "hee-mah-TOH-mah"
- dehiscence → "deh-HIS-ents"
- escharotomy → "es-kar-OT-oh-mee"
- fasciotomy → "fash-ee-OT-oh-mee"
- perforator → "PER-for-ay-tor"
- anastomosis → "ah-nas-toh-MOH-sis"
- lymphedema → "lim-feh-DEE-mah"

KNOWLEDGE BASE TOOL: The pre-loaded context above contains the primary knowledge — answer from it directly whenever possible. Only call search_knowledge_base if the pre-loaded context is genuinely missing a specific fact the question requires — exact numeric values, named classification tiers, or a specific cited study. Do not call it for general clinical questions you can answer from the context or your own knowledge. If called and it returns nothing, continue from your own knowledge without mentioning the tool.

TESTING AND QUIZZING:
When the user asks to be tested, quizzed, or questioned — on the material just reviewed, on a specific topic they name, or on anything from the knowledge base — enter question mode immediately.

Question mode rules:
- Ask ONE question at a time. Wait for the answer before continuing.
- Tailor the question type to the learner level (see below).
- After the user answers: briefly assess it (correct / partially correct / missed something), then fill in what was missing or wrong. Keep this tight — one to three sentences max. Then ask if they want another question or move on.
- If the user gets it wrong or incomplete, give the right answer concisely and explain the key reasoning. Do not lecture — just correct and move on.
- If the user asks for a hint, give one short hint then wait again.
- You may draw questions from the pre-loaded context, the current topic, or any plastic surgery concept the user specifies.
- For multi-part oral board vignettes, reveal one part at a time and wait for each response before advancing.

Level-appropriate question types:
- Medical Student: Direct recall — "What are the zones of the hand?", "Name the layers of the scalp", "What is the blood supply to the deltopectoral flap?"
- Junior Resident: Applied clinical scenario with patient factors and one follow-up probe — "A 45-year-old woman with a 4 cm breast tumor needs mastectomy. What reconstruction options do you offer and what drives your choice?", "You are planning a TRAM flap and the patient has a prior Pfannenstiel scar. What do you do?" After the user answers, ask one classification or decision follow-up.
- Senior Resident: Direct recall — "What is the blood supply to the latissimus dorsi?", "Name the branches of the facial nerve", "What are the indications for a free fibula flap?"

Never reveal the correct answer before the user responds. Never ask compound double-barreled questions. Keep the question itself brief and spoken-word natural.

${levelNote}

[PRE-LOADED CONTEXT]
${knowledgeBase}`;

      const audioModel = "gemini-2.5-flash-native-audio-preview-12-2025";

      // Get API key from Edge Function (verifies user JWT server-side).
      // Using real key with v1beta — ephemeral tokens on v1alpha have a known Google bug
      // where system instructions are ignored, causing audio quality degradation + timeouts.
      const audioKey = await getGeminiToken();

      setAudioStatusDetail("Connecting to audio AI...");
      const ai = new GoogleGenAI({ apiKey: audioKey, httpOptions: { apiVersion: "v1beta" } });

      const sessionPromise = ai.live.connect({
        model: audioModel,
        config: {
          responseModalities: [Modality.AUDIO],
          tools: [KNOWLEDGE_BASE_TOOL],
          systemInstruction: audioSystemInstruction,
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Kore" } } },
          realtimeInputConfig: {
            activityHandling: ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            automaticActivityDetection: {
              startOfSpeechSensitivity: StartSensitivity.START_SENSITIVITY_HIGH,
              endOfSpeechSensitivity: EndSensitivity.END_SENSITIVITY_HIGH,
              prefixPaddingMs: 200,
              silenceDurationMs: 400,
            },
          },
        },
        callbacks: {
          onopen: () => {
            setAudioStatus("active"); setAudioStatusDetail("");
            try {
              const source = inputCtx.createMediaStreamSource(stream);
              const processor = inputCtx.createScriptProcessor(4096, 1, 1);
              processorRef.current = processor;
              processor.onaudioprocess = (e) => {
                const session = audioSessionRef.current;
                if (!session) return;
                const inputData = e.inputBuffer.getChannelData(0);
                let sum = 0;
                for (let i = 0; i < inputData.length; i += 50) sum += Math.abs(inputData[i]);
                setMicVolume(Math.min(100, (sum / (inputData.length / 50)) * 500));
                const downsampled = downsampleBuffer(inputData, inputCtx.sampleRate, 16000);
                try { session.sendRealtimeInput({ media: pcmToGenAIBlob(downsampled) }); } catch (_) {}
              };
              // Route processor through a silent gain node (gain=0) to keep
              // ScriptProcessorNode alive without feeding mic audio to speakers (avoids echo)
              const silentGain = inputCtx.createGain();
              silentGain.gain.value = 0;
              source.connect(processor);
              processor.connect(silentGain);
              silentGain.connect(inputCtx.destination);
            } catch (err) { console.error("Audio pipeline error", err); stopAudioSession(); }
          },
          onmessage: async (msg: LiveServerMessage) => {
            const b64 = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (b64) {
              const ctx = outputCtxRef.current; if (!ctx) return;
              const gen = audioGenRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              try {
                const buf = await decodeAudioData(decodeB64(b64), ctx, 24000, 1);
                // Drop stale chunk if an interrupt fired while we were decoding
                if (audioGenRef.current !== gen) return;
                const src = ctx.createBufferSource();
                src.buffer = buf; src.connect(ctx.destination); src.start(nextStartTimeRef.current);
                nextStartTimeRef.current += buf.duration;
                activeSourcesRef.current.add(src);
                src.onended = () => activeSourcesRef.current.delete(src);
              } catch (e) { console.error("Audio decode error", e); }
            }
            if (msg.serverContent?.interrupted) {
              audioGenRef.current++;
              activeSourcesRef.current.forEach(s => { try { s.stop(); } catch (_) {} });
              activeSourcesRef.current.clear(); nextStartTimeRef.current = 0;
            }
            if (msg.serverContent?.turnComplete) {
              // Model finished its turn — resume both audio contexts if browser suspended them
              if (outputCtxRef.current?.state === "suspended") outputCtxRef.current.resume().catch(() => {});
              if (inputCtxRef.current?.state === "suspended") inputCtxRef.current.resume().catch(() => {});
            }
            // Tool call: model wants to search the knowledge base.
            // Use .then()/.catch() (not await) so the response is ALWAYS sent
            // even if an exception occurs, and never blocks the onmessage handler.
            if (msg.toolCall?.functionCalls?.length) {
              for (const call of msg.toolCall.functionCalls) {
                if (call.name !== "search_knowledge_base") continue;
                const query = ((call.args as any)?.query as string) || "";
                console.log("[Audio Tool] search_knowledge_base:", query);
                setIsSearchingKB(true);
                const sendResponse = (output: string) => {
                  setIsSearchingKB(false);
                  try {
                    audioSessionRef.current?.sendToolResponse({
                      functionResponses: [{ id: call.id, name: call.name, response: { output } }],
                    });
                  } catch (e) { console.error("[Audio Tool] sendToolResponse threw:", e); }
                };
                queryLightRAGFast(query)
                  .then(ctx => sendResponse(ctx || "No specific information found for this query."))
                  .catch(e => { console.error("[Audio Tool] query error:", e); sendResponse("Knowledge base query failed."); });
              }
            }
            if (msg.toolCallCancellation?.ids?.length) {
              console.warn("[Audio Tool] Cancellation:", msg.toolCallCancellation.ids);
            }
          },
          onclose: (e: any) => { console.warn("[Audio] Session closed — code:", e?.code, "reason:", e?.reason); stopAudioSession(false); setIsAudioMode(false); setAudioStatus("disconnected"); setAudioStatusDetail(""); },
          onerror: (err: any) => { console.error("[Audio] Session error:", err); setAudioError(err.message || "Network Error"); setAudioStatus("error"); },
        },
      });
      sessionPromise.then(s => {
        audioSessionRef.current = s;
      }).catch(err => {
        setAudioError("Connection failed. Check network connection."); setAudioStatus("error");
      });
    } catch (error: any) { setAudioError(error.message || "Could not access microphone."); setAudioStatus("error"); }
  };

  const stopAudioSession = (closeOverlay = true) => {
    try {
      // Close the Gemini Live session first to stop incoming audio
      if (audioSessionRef.current) {
        try { audioSessionRef.current.close(); } catch (_) {}
        audioSessionRef.current = null;
      }
      // Stop all playing output audio sources
      activeSourcesRef.current.forEach(s => { try { s.stop(); } catch (_) {} });
      activeSourcesRef.current.clear();
      // Disconnect and release mic processor node
      if (processorRef.current) { try { processorRef.current.disconnect(); } catch (_) {} processorRef.current = null; }
      // Stop mic stream tracks
      streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null;
      // Close audio contexts
      if (inputCtxRef.current?.state !== "closed") inputCtxRef.current?.close();
      if (outputCtxRef.current?.state !== "closed") outputCtxRef.current?.close();
    } catch (e) { console.error(e); }
    inputCtxRef.current = null; outputCtxRef.current = null;
    setIsSearchingKB(false);
    if (closeOverlay) { setIsAudioMode(false); setAudioStatus("disconnected"); setAudioStatusDetail(""); }
  };

  // ─────────────────────────────────────────────
  // SMALL COMPONENTS
  // ─────────────────────────────────────────────

  const LevelBadge = ({ l, compact = false }: { l: Level; compact?: boolean }) => {
    const colors: Record<Level, string> = {
      1: "bg-emerald-100 text-emerald-800 border-emerald-300",
      2: "bg-blue-100 text-blue-800 border-blue-300",
      3: "bg-purple-100 text-purple-800 border-purple-300",
    };
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${colors[l]}`}>
        <span>{LEVELS[l].icon}</span>
        {compact ? `L${l}: ${LEVELS[l].short}` : LEVELS[l].name}
      </span>
    );
  };

  // ─────────────────────────────────────────────
  // LOGIN PAGE
  // ─────────────────────────────────────────────

  const LoginPage = () => {
    const [isRegister, setIsRegister] = useState(false);
    const [name, setName] = useState("");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [submitting, setSubmitting] = useState(false);
    const [showPw, setShowPw] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      setSuccess(null);
      setSubmitting(true);
      if (isRegister) {
        const result = await signUp(name, username);
        if (result === "SUCCESS") {
          setSuccess("Your access request has been submitted. Once approved, you'll receive an email invitation to set your password and sign in.");
          setName(""); setUsername("");
        } else if (result) {
          setError(result);
        }
      } else {
        const err = await signIn(username, password);
        if (err) setError(err);
      }
      setSubmitting(false);
    };

    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center px-4">
        <div className="flex flex-col items-center mb-8">
          <div className="w-16 h-16 bg-teal-500/20 rounded-2xl flex items-center justify-center border border-teal-500/40 mb-4">
            <Stethoscope className="text-teal-400" size={32} />
          </div>
          <h1 className="text-3xl font-bold text-white">Plastic Surgery Tutor</h1>
          <p className="text-slate-400 text-sm mt-1">AI-powered surgical education platform</p>
        </div>

        <div className="w-full max-w-sm bg-slate-800 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden">
          <div className="flex border-b border-slate-700">
            <button className={`flex-1 py-3.5 text-sm font-semibold transition-colors ${!isRegister ? "bg-teal-600 text-white" : "text-slate-400 hover:text-slate-200"}`} onClick={() => { setIsRegister(false); setError(null); setSuccess(null); }}>Sign In</button>
            <button className={`flex-1 py-3.5 text-sm font-semibold transition-colors ${isRegister ? "bg-teal-600 text-white" : "text-slate-400 hover:text-slate-200"}`} onClick={() => { setIsRegister(true); setError(null); setSuccess(null); }}>Register</button>
          </div>
          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            {isRegister && (
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Full name</label>
                <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Dr. Smith"
                  className="w-full bg-slate-700 border border-slate-600 rounded-xl px-4 py-3 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500" autoComplete="name" />
              </div>
            )}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Email</label>
              <input type="email" value={username} onChange={e => setUsername(e.target.value)} placeholder={isRegister ? "you@example.com" : "Your email"}
                className="w-full bg-slate-700 border border-slate-600 rounded-xl px-4 py-3 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500" autoComplete="email" />
            </div>
            {!isRegister && (
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Password</label>
                <div className="relative">
                  <input type={showPw ? "text" : "password"} value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter your password"
                    className="w-full bg-slate-700 border border-slate-600 rounded-xl px-4 py-3 pr-11 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500" autoComplete="current-password" />
                  <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200">
                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
            )}
            {isRegister && (
              <p className="text-slate-500 text-xs leading-relaxed">Once your request is approved, you'll receive an email invitation to set your own password.</p>
            )}
            {error && (
              <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-red-300 text-sm">
                <AlertCircle size={16} className="shrink-0 mt-0.5" />{error}
              </div>
            )}
            {success && (
              <div className="flex items-start gap-2 bg-teal-500/10 border border-teal-500/30 rounded-xl px-4 py-3 text-teal-300 text-sm">
                <CheckCircle size={16} className="shrink-0 mt-0.5" />{success}
              </div>
            )}
            <button type="submit" disabled={submitting} className="w-full bg-teal-600 hover:bg-teal-500 disabled:bg-teal-800 text-white font-semibold py-3 rounded-xl transition-colors text-sm flex items-center justify-center gap-2">
              {submitting ? <><Loader2 size={16} className="animate-spin" /> {isRegister ? "Submitting request..." : "Signing in..."}</> : isRegister ? "Request Account" : "Sign In"}
            </button>
          </form>
          <div className="px-6 pb-6">
            <div className="bg-slate-700/50 rounded-xl p-3 flex items-center gap-3">
              <Trophy className="text-amber-400 shrink-0" size={18} />
              <p className="text-slate-400 text-xs leading-relaxed">Earn points by answering quiz questions correctly. Compete with your study group on the leaderboard!</p>
            </div>
          </div>
        </div>
        <p className="text-slate-600 text-xs mt-6">Powered by Supabase — accounts work across any browser.</p>
      </div>
    );
  };

  // ─────────────────────────────────────────────
  // LEADERBOARD PAGE
  // ─────────────────────────────────────────────

  const LeaderboardPage = () => {
    const [users, setUsers] = useState<UserProfile[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
      supabase.from("user_progress").select("*").order("total_points", { ascending: false })
        .then(({ data }) => {
          const rows = (data as UserProfile[]) ?? [];
          const best = new Map<string, UserProfile>();
          rows.forEach(r => { if (!best.has(r.username)) best.set(r.username, r); });
          setUsers(Array.from(best.values()));
          setLoading(false);
        });
    }, []);

    const accuracy = (u: UserProfile) =>
      u.total_questions > 0 ? Math.round((u.correct_answers / u.total_questions) * 100) : 0;

    const podiumOrder = [1, 0, 2];
    const podiumColors = [
      { bg: "bg-slate-400", medal: "🥈" },
      { bg: "bg-amber-400", medal: "🥇" },
      { bg: "bg-orange-500", medal: "🥉" },
    ];
    const podiumHeights = ["h-24", "h-36", "h-20"];
    const podium = users.slice(0, 3);

    return (
      <div className="flex-1 overflow-y-auto bg-slate-50">
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 px-6 pt-8 pb-16">
          <div className="max-w-2xl mx-auto text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Trophy className="text-amber-400" size={28} />
              <h2 className="text-2xl font-bold text-white">Leaderboard</h2>
            </div>
          </div>
        </div>

        <div className="max-w-2xl mx-auto px-4 -mt-10">
          {loading ? (
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
              <Loader2 className="text-teal-500 animate-spin mx-auto mb-3" size={32} />
              <p className="text-slate-500 text-sm">Loading leaderboard...</p>
            </div>
          ) : users.length === 0 ? (
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
              <Trophy className="text-slate-300 mx-auto mb-4" size={48} />
              <h3 className="text-slate-600 font-semibold text-lg mb-1">No scores yet!</h3>
              <p className="text-slate-400 text-sm">Go to Chat → click <strong>Quiz</strong> → answer → mark yourself correct to earn points.</p>
            </div>
          ) : (
            <>
              {/* Podium */}
              {podium.length > 0 && (
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-4">
                  <div className="flex items-end justify-center gap-6">
                    {podiumOrder.map((rankIdx, col) => {
                      const user = podium[rankIdx];
                      if (!user) return <div key={col} className="w-20" />;
                      const style = podiumColors[col];
                      const isMe = user.username === currentUser;
                      return (
                        <div key={col} className="flex flex-col items-center gap-2">
                          <span className="text-2xl">{style.medal}</span>
                          <div className={`w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold text-white bg-slate-700 ${isMe ? "ring-4 ring-teal-400" : ""}`}>
                            {user.username[0].toUpperCase()}
                          </div>
                          <div className="text-center">
                            <p className={`text-xs font-semibold truncate max-w-[72px] ${isMe ? "text-teal-600" : "text-slate-700"}`}>{isMe ? "You" : user.username}</p>
                            <p className="text-xs text-slate-500 font-medium">{user.total_points} pts</p>
                          </div>
                          <div className={`w-16 rounded-t-lg ${style.bg} ${podiumHeights[col]} flex items-end justify-center pb-1`}>
                            <span className="text-white text-xs font-bold">#{rankIdx + 1}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Rankings table */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-4">
                <div className="grid grid-cols-[2rem_1fr_5rem_5rem_4rem] text-xs font-semibold text-slate-500 uppercase tracking-wide px-4 py-3 border-b border-slate-100 bg-slate-50">
                  <span>#</span><span>User</span><span className="text-right">Points</span><span className="text-right">Accuracy</span><span className="text-right">Streak</span>
                </div>
                {users.map((user, idx) => {
                  const isMe = user.username === currentUser;
                  return (
                    <div key={user.username} className={`grid grid-cols-[2rem_1fr_5rem_5rem_4rem] items-center px-4 py-3.5 border-b border-slate-50 last:border-0 text-sm ${isMe ? "bg-teal-50 font-semibold" : "hover:bg-slate-50"}`}>
                      <span className="text-slate-400 text-xs">{idx === 0 ? "🥇" : idx === 1 ? "🥈" : idx === 2 ? "🥉" : `${idx + 1}`}</span>
                      <div className="flex items-center gap-2 min-w-0">
                        <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0 ${isMe ? "bg-teal-600" : "bg-slate-500"}`}>
                          {user.username[0].toUpperCase()}
                        </div>
                        <div className="min-w-0">
                          <p className={`truncate text-sm ${isMe ? "text-teal-700" : "text-slate-700"}`}>{user.username}{isMe && " (you)"}</p>
                          {user.preferred_level && <p className="text-[10px] text-slate-400">{LEVELS[user.preferred_level].icon} {LEVELS[user.preferred_level].short}</p>}
                        </div>
                      </div>
                      <div className="text-right"><span className="font-bold text-amber-600">{user.total_points}</span></div>
                      <div className="text-right"><span className={`text-xs font-medium ${accuracy(user) >= 70 ? "text-emerald-600" : accuracy(user) >= 40 ? "text-amber-600" : "text-red-500"}`}>{accuracy(user)}%</span></div>
                      <div className="text-right flex items-center justify-end gap-1">{user.best_streak > 0 && <Flame size={11} className="text-orange-400" />}<span className="text-xs text-slate-500">{user.best_streak}</span></div>
                    </div>
                  );
                })}
              </div>

              {/* Points guide */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 mb-8">
                <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">How points are earned</h4>
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div className="flex flex-col items-center bg-slate-50 rounded-xl p-3 text-center">
                    <span className="text-xl mb-1">✅</span>
                    <p className="text-xs font-semibold text-slate-600">At baseline</p>
                    <p className="text-xs text-slate-400">any level</p>
                    <p className="text-sm font-bold text-amber-600 mt-1">+{BASE_POINTS} pts</p>
                  </div>
                  <div className="flex flex-col items-center bg-indigo-50 rounded-xl p-3 text-center border border-indigo-100">
                    <span className="text-xl mb-1">⬆️</span>
                    <p className="text-xs font-semibold text-slate-600">1 level above</p>
                    <p className="text-xs text-slate-400">baseline bonus</p>
                    <p className="text-sm font-bold text-amber-600 mt-1">+{BASE_POINTS + LEVEL_BONUS[1]} pts</p>
                  </div>
                  <div className="flex flex-col items-center bg-purple-50 rounded-xl p-3 text-center border border-purple-100">
                    <span className="text-xl mb-1">🚀</span>
                    <p className="text-xs font-semibold text-slate-600">2 levels above</p>
                    <p className="text-xs text-slate-400">baseline bonus</p>
                    <p className="text-sm font-bold text-amber-600 mt-1">+{BASE_POINTS + LEVEL_BONUS[2]} pts</p>
                  </div>
                </div>
                <p className="text-[10px] text-slate-400 text-center">🔥 Every 3 consecutive correct answers = +5 streak bonus</p>
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  // ─────────────────────────────────────────────
  // ROOT RENDER
  // ─────────────────────────────────────────────

  // Loading splash while checking existing session
  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center gap-4">
        <div className="w-14 h-14 bg-teal-500/20 rounded-2xl flex items-center justify-center border border-teal-500/40">
          <Stethoscope className="text-teal-400" size={28} />
        </div>
        <Loader2 className="text-teal-500 animate-spin" size={28} />
        <p className="text-slate-500 text-sm">Connecting to Supabase...</p>
      </div>
    );
  }

  if (!currentUser) return <LoginPage />;

  return (
    <div className="flex h-screen w-full overflow-hidden text-slate-800 relative font-sans">

      {/* LEVEL SELECTION OVERLAY */}
      {level === null && (
        <div className="absolute inset-0 z-50 bg-slate-950 flex flex-col items-center justify-center px-4">
          <div className="max-w-2xl w-full">
            <div className="text-center mb-10">
              <div className="w-14 h-14 bg-teal-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg shadow-teal-500/30">
                <Stethoscope className="text-white" size={28} />
              </div>
              <h1 className="text-3xl font-bold text-white mb-2">Welcome, {currentUser}!</h1>
              <p className="text-slate-300 text-sm">Choose your <strong className="text-white">baseline level</strong> — this is locked to your account and used to calculate bonus points.</p>
              <div className="flex items-center justify-center gap-4 mt-3 text-xs text-slate-400">
                <span className="flex items-center gap-1"><span className="text-amber-400 font-bold">+{BASE_POINTS} pts</span> correct answer</span>
                <span className="text-slate-600">|</span>
                <span className="flex items-center gap-1"><span className="text-amber-400 font-bold">+{BASE_POINTS + LEVEL_BONUS[1]} pts</span> 1 level above baseline</span>
                <span className="text-slate-600">|</span>
                <span className="flex items-center gap-1"><span className="text-amber-400 font-bold">+{BASE_POINTS + LEVEL_BONUS[2]} pts</span> 2 levels above</span>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {([1, 2, 3] as Level[]).map(l => {
                const cardColors: Record<Level, string> = {
                  1: "border-emerald-500 bg-emerald-950 hover:bg-emerald-900",
                  2: "border-blue-500 bg-blue-950 hover:bg-blue-900",
                  3: "border-purple-500 bg-purple-950 hover:bg-purple-900"
                };
                const btnColors: Record<Level, string> = { 1: "bg-emerald-500 hover:bg-emerald-400", 2: "bg-blue-500 hover:bg-blue-400", 3: "bg-purple-500 hover:bg-purple-400" };
                const badgeColors: Record<Level, string> = { 1: "bg-emerald-500 text-white", 2: "bg-blue-500 text-white", 3: "bg-purple-500 text-white" };
                const iconColors: Record<Level, string> = { 1: "text-emerald-400", 2: "text-blue-400", 3: "text-purple-400" };
                return (
                  <button key={l} onClick={() => handleSelectLevel(l)} className={`flex flex-col items-start p-6 rounded-2xl border-2 transition-all duration-200 text-left shadow-lg ${cardColors[l]}`}>
                    <div className="flex items-center justify-between w-full mb-4">
                      <span className={`text-3xl ${iconColors[l]}`}>{LEVELS[l].icon}</span>
                      <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${badgeColors[l]}`}>{LEVELS[l].short}</span>
                    </div>
                    <h3 className="text-white font-bold text-lg mb-2">Level {l}: {LEVELS[l].name}</h3>
                    <p className="text-slate-300 text-sm leading-relaxed mb-4 flex-1">{LEVELS[l].description}</p>
                    <div className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-white text-sm font-bold transition-colors shadow-sm ${btnColors[l]}`}>
                      Set as my baseline <ChevronRight size={16} />
                    </div>
                  </button>
                );
              })}
            </div>
            <p className="text-center text-slate-500 text-xs mt-6">You can switch levels during study to earn bonus points — your baseline stays fixed to your account.</p>
          </div>
        </div>
      )}

      {/* AUDIO OVERLAY */}
      {isAudioMode && (
        <div className="absolute inset-0 z-50 bg-slate-900/95 backdrop-blur-sm flex flex-col items-center justify-center text-white">
          <button onClick={() => stopAudioSession(true)} className="absolute top-6 right-6 p-2 bg-white/10 hover:bg-white/20 rounded-full"><X size={24} /></button>
          <div className="flex flex-col items-center gap-8 max-w-md px-4 text-center">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Audio Session</h2>
              {level && <div className="flex justify-center"><LevelBadge l={level} /></div>}
              <span className="text-xs bg-teal-500/20 text-teal-300 border border-teal-500/30 px-2.5 py-1 rounded-full font-medium">{audioTopic.label}</span>
              <p className="text-slate-400">{audioStatus === "connecting" ? (audioStatusDetail || "Connecting...") : audioStatus === "error" ? "Connection Error" : "Listening... Speak clearly"}</p>
              {audioStatus === "error" && <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-3 text-red-200 flex items-center gap-2 justify-center"><AlertCircle size={18} /><span className="text-sm">{audioError}</span></div>}
              {isSearchingKB && (
                <div className="flex items-center gap-2 bg-indigo-500/15 border border-indigo-500/30 text-indigo-300 text-sm px-3 py-2 rounded-full">
                  <Loader2 size={14} className="animate-spin shrink-0" />
                  <span>Searching knowledge base...</span>
                </div>
              )}
            </div>
            <div className="relative flex items-center justify-center py-8">
              {audioStatus === "active" && (<><div className="absolute w-32 h-32 bg-teal-500/20 rounded-full animate-ping opacity-75"></div><div className="absolute bg-teal-500/10 rounded-full transition-all duration-75" style={{ width: `${100 + micVolume * 2}px`, height: `${100 + micVolume * 2}px` }}></div></>)}
              <div className={`w-24 h-24 rounded-full flex items-center justify-center shadow-2xl transition-all duration-500 ${audioStatus === "active" ? "bg-teal-500 scale-110 shadow-teal-500/50" : audioStatus === "error" ? "bg-red-500" : "bg-slate-700"}`}>
                {audioStatus === "connecting" ? <RefreshCw className="animate-spin text-white/50" size={32} /> : audioStatus === "error" ? <AlertCircle className="text-white" size={40} /> : <Mic className={`text-white ${micVolume > 10 ? "scale-110" : "scale-100"}`} size={40} />}
              </div>
            </div>
            <div className="flex gap-4">
              {audioStatus === "error" ? (
                <button onClick={startAudioSession} className="px-6 py-3 bg-white text-slate-900 hover:bg-slate-100 font-medium rounded-full flex items-center gap-2"><RefreshCw size={20} /> Retry</button>
              ) : (
                <button onClick={() => stopAudioSession(true)} className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white font-medium rounded-full flex items-center gap-2"><Headphones size={20} /> End Session</button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* MAIN CONTENT */}
      <div className="flex-1 flex flex-col h-full relative overflow-hidden">

        {/* HEADER */}
        <header className="border-b border-slate-200 bg-white shadow-sm z-10 shrink-0">
          <div className="h-12 md:h-16 flex items-center justify-between px-3 md:px-4">
          <div className="flex items-center gap-2 md:gap-3">
            <div className="w-9 h-9 bg-teal-100 rounded-full flex items-center justify-center text-teal-600 shrink-0"><Stethoscope size={20} /></div>
            <div className="hidden sm:block">
              <h1 className="font-bold text-slate-800 text-base leading-tight">Plastic Surgery Tutor</h1>
              {level && page === "chat" && <LevelBadge l={level} compact />}
            </div>
          </div>

          {/* Nav tabs */}
          <div className="flex items-center gap-0.5 md:gap-1 bg-slate-100 rounded-xl p-0.5 md:p-1">
            <button onClick={() => setPage("chat")} className={`flex items-center gap-1 md:gap-1.5 px-2 md:px-3 py-1 md:py-1.5 rounded-lg text-xs md:text-sm font-medium transition-all ${page === "chat" ? "bg-white text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
              <BrainCircuit size={15} /> Chat
            </button>
            <button onClick={() => setPage("leaderboard")} className={`flex items-center gap-1 md:gap-1.5 px-2 md:px-3 py-1 md:py-1.5 rounded-lg text-xs md:text-sm font-medium transition-all ${page === "leaderboard" ? "bg-white text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
              <Trophy size={15} /> Board
            </button>
          </div>

          <div className="flex items-center gap-1.5 md:gap-2">
            {/* Baseline level indicator */}
            {page === "chat" && userProfile?.baseline_level && (
              <div className="relative hidden md:block">
                <button onClick={e => { e.stopPropagation(); setShowBaselineMenu(!showBaselineMenu); setShowMaterialMenu(false); setShowAudioTopicMenu(false); setShowUserMenu(false); }} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-50 hover:bg-slate-100 text-slate-500 rounded-lg text-xs font-medium border border-slate-200">
                  <span className="text-[10px] uppercase tracking-wide text-slate-400 font-semibold">Baseline</span>
                  <span>{LEVELS[userProfile.baseline_level].icon} L{userProfile.baseline_level}</span>
                  <ChevronDown size={11} />
                </button>
                {showBaselineMenu && <div className="absolute right-0 top-full mt-1 w-56 bg-white border border-slate-200 rounded-xl shadow-xl z-30 overflow-hidden">
                  <div className="p-2">
                    <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide px-2 pb-1">Set baseline level</p>
                    <p className="text-[10px] text-slate-400 px-2 pb-2 leading-4">Bonus points are earned for answering above your baseline.</p>
                    {([1, 2, 3] as Level[]).map(l => {
                      const h: Record<Level, string> = { 1: "hover:bg-emerald-50", 2: "hover:bg-blue-50", 3: "hover:bg-purple-50" };
                      const isBaseline = userProfile.baseline_level === l;
                      return (
                        <button key={l} onClick={async () => {
                          await supabase.from("user_progress").update({ baseline_level: l }).eq("id", userProfile.id);
                          setUserProfile(prev => prev ? { ...prev, baseline_level: l } : null);
                        }} className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-colors ${isBaseline ? "bg-slate-100 font-semibold" : h[l]}`}>
                          <span>{LEVELS[l].icon}</span><span>Level {l}: {LEVELS[l].name}</span>
                          {isBaseline && <span className="ml-auto text-[10px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded-full">current</span>}
                        </button>
                      );
                    })}
                  </div>
                </div>}
              </div>
            )}

            {/* Level switcher */}
            {page === "chat" && level && (
              <div className="relative">
                <button onClick={e => { e.stopPropagation(); setShowMaterialMenu(!showMaterialMenu); setShowBaselineMenu(false); setShowAudioTopicMenu(false); setShowUserMenu(false); }} className="flex items-center gap-1 md:gap-1.5 px-2 md:px-2.5 py-1.5 bg-slate-50 hover:bg-slate-100 text-slate-500 rounded-lg text-xs font-medium border border-slate-200">
                  <span className="hidden md:inline text-[10px] uppercase tracking-wide text-slate-400 font-semibold">Material</span>
                  <span>{LEVELS[level].icon} L{level}</span>
                  <ChevronDown size={11} />
                </button>
                {showMaterialMenu && <div className="absolute right-0 top-full mt-1 w-52 bg-white border border-slate-200 rounded-xl shadow-xl z-30 overflow-hidden">
                  <div className="p-2">
                    <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide px-2 pb-1">Switch level</p>
                    {([1, 2, 3] as Level[]).map(l => {
                      const h: Record<Level, string> = { 1: "hover:bg-emerald-50", 2: "hover:bg-blue-50", 3: "hover:bg-purple-50" };
                      return (
                        <button key={l} onClick={() => { handleSelectLevel(l); setShowMaterialMenu(false); }} className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-colors ${level === l ? "bg-slate-100 font-semibold" : h[l]}`}>
                          <span>{LEVELS[l].icon}</span><span>Level {l}: {LEVELS[l].name}</span>
                          {level === l && <span className="ml-auto text-[10px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded-full">active</span>}
                        </button>
                      );
                    })}
                  </div>
                </div>}
              </div>
            )}

            {page === "chat" && (
              <div className="hidden md:flex items-center rounded-lg border border-teal-200 overflow-visible">
                <button onClick={startAudioSession} disabled={!level} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-teal-50 text-teal-700 hover:bg-teal-100 disabled:opacity-40 text-sm font-medium">
                  <Mic size={15} /><span className="hidden md:inline">Audio Mode</span>
                </button>
                <div className="relative">
                  <button onClick={e => { e.stopPropagation(); setShowAudioTopicMenu(!showAudioTopicMenu); setShowMaterialMenu(false); setShowBaselineMenu(false); setShowUserMenu(false); }} disabled={!level} className="flex items-center px-1.5 py-1.5 bg-teal-50 text-teal-600 hover:bg-teal-100 disabled:opacity-40 border-l border-teal-200 text-sm">
                    <ChevronDown size={13} />
                  </button>
                  {showAudioTopicMenu && <div className="absolute right-0 top-full mt-1 w-64 bg-white border border-slate-200 rounded-xl shadow-xl z-30 overflow-hidden">
                    <div className="p-2">
                      <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide px-2 pb-1">Select topic for audio session</p>
                      {AUDIO_TOPICS.map(t => (
                        <button key={t.label} onClick={() => { setAudioTopic(t); setShowAudioTopicMenu(false); }}
                          className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${audioTopic.label === t.label ? "bg-teal-50 text-teal-700 font-semibold" : "hover:bg-slate-50 text-slate-700"}`}>
                          {t.label}
                          {audioTopic.label === t.label && <span className="ml-2 text-[10px] bg-teal-100 text-teal-600 px-1.5 py-0.5 rounded-full">selected</span>}
                        </button>
                      ))}
                    </div>
                  </div>}
                </div>
              </div>
            )}
            {page === "chat" && (
              <div className="hidden md:flex items-center gap-1.5">
                <button onClick={() => startQuiz("recall")} disabled={isLoading || !level} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-teal-50 text-teal-700 hover:bg-teal-100 disabled:opacity-40 rounded-lg text-sm font-medium border border-teal-200">
                  <GraduationCap size={15} /> Recall Question
                </button>
                <button onClick={() => startQuiz("board")} disabled={isLoading || !level} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 disabled:opacity-40 rounded-lg text-sm font-medium border border-indigo-200">
                  <BookOpen size={15} /> MCQ Question
                </button>
              </div>
            )}

            {/* User menu */}
            <div className="relative">
              <button onClick={e => { e.stopPropagation(); setShowUserMenu(!showUserMenu); setShowBaselineMenu(false); setShowMaterialMenu(false); setShowAudioTopicMenu(false); }} className="flex items-center gap-2 px-2.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium">
                <div className="w-5 h-5 bg-teal-600 rounded-full flex items-center justify-center text-white text-[10px] font-bold">{currentUser[0].toUpperCase()}</div>
                <span className="hidden sm:inline max-w-[80px] truncate">{currentUser}</span>
                <ChevronDown size={12} />
              </button>
              {showUserMenu && (
                <div className="absolute right-0 top-full mt-1 w-52 bg-white border border-slate-200 rounded-xl shadow-xl z-30 overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-100">
                    <p className="text-xs text-slate-500">Signed in as</p>
                    <p className="text-sm font-semibold text-slate-800 truncate">{currentUser}</p>
                    {userProfile && (
                      <div className="mt-2 space-y-1">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-1 text-xs text-amber-600 font-semibold"><Trophy size={11} /> {userProfile.total_points} pts</div>
                          {userProfile.best_streak > 0 && <div className="flex items-center gap-1 text-xs text-orange-500 font-medium"><Flame size={11} /> {userProfile.best_streak} best</div>}
                        </div>
                        {(userProfile.login_streak ?? 0) > 0 && (
                          <div className="flex items-center gap-1 text-xs text-orange-500 font-medium">
                            <Flame size={11} /> {userProfile.login_streak} day login streak
                          </div>
                        )}
                        {userProfile.baseline_level && (
                          <p className="text-[10px] text-slate-400">Baseline: {LEVELS[userProfile.baseline_level].icon} {LEVELS[userProfile.baseline_level].name}</p>
                        )}
                      </div>
                    )}
                  </div>
                  <button onClick={handleLogout} className="w-full flex items-center gap-2 px-4 py-3 text-sm text-red-600 hover:bg-red-50 transition-colors">
                    <LogOut size={15} /> Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
          </div>
          {/* Row 2: mobile-only action bar */}
          {page === "chat" && (
            <div className="md:hidden relative flex items-center justify-center gap-2 px-3 py-1.5 border-t border-slate-100">
              <div className="flex items-center rounded-lg border border-teal-200 overflow-visible">
                <button onClick={startAudioSession} disabled={!level} className="flex items-center gap-1.5 px-2.5 py-1 bg-teal-50 text-teal-700 hover:bg-teal-100 disabled:opacity-40 text-sm font-medium">
                  <Mic size={14} /> Audio
                </button>
                <button onClick={e => { e.stopPropagation(); setShowAudioTopicMenu(!showAudioTopicMenu); setShowMaterialMenu(false); setShowBaselineMenu(false); setShowUserMenu(false); }} disabled={!level} className="flex items-center px-1.5 py-1 bg-teal-50 text-teal-600 hover:bg-teal-100 disabled:opacity-40 border-l border-teal-200 text-sm">
                  <ChevronDown size={12} />
                </button>
              </div>
              <button onClick={() => startQuiz("recall")} disabled={isLoading || !level} className="flex items-center gap-1 px-2.5 py-1 bg-teal-50 text-teal-700 hover:bg-teal-100 disabled:opacity-40 rounded-lg text-sm font-medium border border-teal-200">
                <GraduationCap size={14} /> Recall
              </button>
              <button onClick={() => startQuiz("board")} disabled={isLoading || !level} className="flex items-center gap-1 px-2.5 py-1 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 disabled:opacity-40 rounded-lg text-sm font-medium border border-indigo-200">
                <BookOpen size={14} /> MCQ
              </button>
              {showAudioTopicMenu && (
                <div className="absolute left-3 right-3 top-full mt-1 bg-white border border-slate-200 rounded-xl shadow-xl z-30 overflow-hidden">
                  <div className="p-2">
                    <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wide px-2 pb-1">Select topic for audio session</p>
                    {AUDIO_TOPICS.map(t => (
                      <button key={t.label} onClick={() => { setAudioTopic(t); setShowAudioTopicMenu(false); }}
                        className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${audioTopic.label === t.label ? "bg-teal-50 text-teal-700 font-semibold" : "hover:bg-slate-50 text-slate-700"}`}>
                        {t.label}
                        {audioTopic.label === t.label && <span className="ml-2 text-[10px] bg-teal-100 text-teal-600 px-1.5 py-0.5 rounded-full">selected</span>}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </header>

        {/* CHAT PAGE */}
        {page === "chat" && (
          <>
            <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50/50">
              {/* Daily login streak banner */}
              {userProfile && (userProfile.login_streak ?? 0) > 0 && (
                <div className="max-w-3xl mx-auto">
                  <div className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium border ${
                    (userProfile.login_streak ?? 0) >= 7
                      ? "bg-orange-50 border-orange-200 text-orange-700"
                      : "bg-amber-50 border-amber-200 text-amber-700"
                  }`}>
                    <Flame size={16} className={(userProfile.login_streak ?? 0) >= 7 ? "text-orange-500" : "text-amber-500"} />
                    <span>
                      <strong>{userProfile.login_streak} day{(userProfile.login_streak ?? 0) !== 1 ? "s" : ""} in a row</strong>
                      {" — "}
                      {(userProfile.login_streak ?? 0) >= 30 ? "Legendary dedication! 🏆" :
                       (userProfile.login_streak ?? 0) >= 14 ? "Two weeks strong! Keep it up." :
                       (userProfile.login_streak ?? 0) >= 7  ? "One week streak! You're on fire." :
                       (userProfile.login_streak ?? 0) >= 3  ? "Building momentum — don't break it!" :
                       "Good start — come back tomorrow to keep your streak!"}
                    </span>
                  </div>
                </div>
              )}

              {messages.map((msg, idx) => {
                // Strip emojis and render **bold** — all other * stripped
                const inlineRender = (text: string) => {
                  const noEmoji = text.replace(/[\u{1F000}-\u{1FFFF}\u{2600}-\u{27BF}]/gu, "");
                  return noEmoji.split(/(\*\*[^*\n]+\*\*)/).map((part, j) => {
                    if (part.startsWith("**") && part.endsWith("**") && part.length > 4)
                      return <strong key={j}>{part.slice(2, -2).replace(/\*/g, "")}</strong>;
                    return <span key={j}>{part.replace(/\*/g, "")}</span>;
                  });
                };

                // Render a single line, converting markdown to clean elements
                const renderLine = (line: string, i: number) => {
                  const trimmed = line.trim().replace(/[\u{1F000}-\u{1FFFF}\u{2600}-\u{27BF}]/gu, "");

                  // Horizontal rules → thin divider
                  if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
                    return <div key={i} className="border-t border-slate-200 my-2" />;
                  }

                  // Markdown table rows (| ... |) and separator rows (|:---|) → skip entirely
                  if (/^\|/.test(trimmed)) {
                    return null;
                  }

                  // Headings (# / ## / ### etc.) → strip # and render slightly bold
                  const headingMatch = trimmed.match(/^#{1,6}\s+(.+)$/);
                  if (headingMatch) {
                    return (
                      <div key={i} className="font-semibold text-slate-800 mt-2 mb-0.5 min-h-[1rem]">
                        {inlineRender(headingMatch[1])}
                      </div>
                    );
                  }

                  // Bullet points (* text or - text)
                  const bulletMatch = trimmed.match(/^[\*\-]\s+(.+)$/);
                  if (bulletMatch) {
                    return (
                      <div key={i} className="flex gap-2 min-h-[1rem] pl-1">
                        <span className="text-slate-400 shrink-0">•</span>
                        <span>{inlineRender(bulletMatch[1])}</span>
                      </div>
                    );
                  }

                  // Normal line (strip any stray leading * or # not caught above)
                  const cleaned = trimmed.replace(/^\*+\s*/, "");
                  return (
                    <div key={i} className="min-h-[1rem]">
                      {inlineRender(cleaned || line)}
                    </div>
                  );
                };

                // Board question with interactive A–E buttons
                const showInteractive = msg.isBoardQuestion && quizActive;
                if (showInteractive) {
                  const { pre, choices, post } = splitBoardMessage(msg.text);
                  return (
                    <div key={idx} className="flex gap-3 max-w-3xl mr-auto">
                      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-teal-600 text-white">
                        <BrainCircuit size={16} />
                      </div>
                      <div className="rounded-2xl px-5 py-4 shadow-sm max-w-[85%] md:max-w-[75%] bg-white border border-slate-200 text-slate-700 rounded-tl-none">
                        {/* Vignette + question stem */}
                        <div className="text-sm leading-6 mb-4">
                          {pre.split("\n").map((line, i) => renderLine(line, i))}
                        </div>
                        {/* Answer choice buttons */}
                        <div className="space-y-2">
                          {choices.map(choice => {
                            const isSelected = selectedAnswer === choice.letter;
                            const isCorrect = boardCorrectAnswer === choice.letter;
                            let cls = "w-full text-left px-3 py-2.5 rounded-xl border text-sm transition-all flex items-center gap-3 ";
                            if (!quizAnswered) {
                              cls += "border-slate-200 hover:border-indigo-400 hover:bg-indigo-50 text-slate-700 cursor-pointer";
                            } else if (isCorrect) {
                              cls += "border-emerald-400 bg-emerald-50 text-emerald-800 font-medium cursor-default";
                            } else if (isSelected) {
                              cls += "border-red-400 bg-red-50 text-red-800 cursor-default";
                            } else {
                              cls += "border-slate-100 bg-slate-50 text-slate-400 cursor-default";
                            }
                            return (
                              <button
                                key={choice.letter}
                                className={cls}
                                onClick={() => !quizAnswered && handleAnswerClick(choice.letter)}
                                disabled={quizAnswered}
                              >
                                <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 border ${
                                  !quizAnswered ? "border-slate-300 text-slate-600 bg-white" :
                                  isCorrect ? "border-emerald-500 text-emerald-700 bg-emerald-100" :
                                  isSelected ? "border-red-400 text-red-700 bg-red-100" :
                                  "border-slate-200 text-slate-400 bg-white"
                                }`}>{choice.letter}</span>
                                <span className="flex-1">{choice.text}</span>
                                {quizAnswered && isCorrect && <CheckCircle size={15} className="text-emerald-500 shrink-0" />}
                                {quizAnswered && isSelected && !isCorrect && <XCircle size={15} className="text-red-400 shrink-0" />}
                              </button>
                            );
                          })}
                        </div>
                        {/* Immediate feedback badge */}
                        {quizAnswered && (
                          <div className={`mt-3 flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold border ${
                            selectedAnswer === boardCorrectAnswer
                              ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                              : "bg-red-50 border-red-200 text-red-700"
                          }`}>
                            {selectedAnswer === boardCorrectAnswer
                              ? <><CheckCircle size={15} /> Correct! Explanation loading…</>
                              : <><XCircle size={15} /> Incorrect — correct answer is {boardCorrectAnswer}. Explanation loading…</>
                            }
                          </div>
                        )}
                      </div>
                    </div>
                  );
                }

                // Normal message rendering
                return (
                  <div key={idx} className={`flex gap-3 max-w-3xl ${msg.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"}`}>
                    <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${msg.role === "user" ? "bg-indigo-600 text-white" : "bg-teal-600 text-white"}`}>
                      {msg.role === "user" ? <div className="text-xs font-bold">You</div> : <BrainCircuit size={16} />}
                    </div>
                    <div className={`rounded-2xl px-5 py-3.5 shadow-sm max-w-[85%] md:max-w-[75%] text-sm leading-6 ${msg.role === "user" ? "bg-indigo-600 text-white rounded-tr-none" : "bg-white border border-slate-200 text-slate-700 rounded-tl-none"}`}>
                      {msg.text.split("\n").map((line, i) => renderLine(line, i))}
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex gap-3 mr-auto max-w-3xl animate-pulse">
                  <div className="w-8 h-8 rounded-full bg-teal-600/20 flex items-center justify-center"><BrainCircuit size={16} className="text-teal-600" /></div>
                  <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-none px-5 py-4 shadow-sm">
                    <div className="flex items-center gap-2 text-sm text-slate-500"><Sparkles size={14} className="text-amber-500" /><span>Thinking thoroughly...</span></div>
                  </div>
                </div>
              )}

              {/* Follow-up action buttons — shown after any regular chat answer */}
              {showFollowUpButtons && !isLoading && !quizActive && (
                <div className="mr-auto">
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={async () => {
                        setShowFollowUpButtons(false);
                        const historySnapshot = messages.slice(1).map(m => ({ role: m.role, text: m.text }));
                        const lastUserMsg = [...messages].reverse().find(m => m.role === "user")?.text ?? "this topic";
                        const ragContext = await queryLightRAG(lastUserMsg);
                        if (!ragContext) return;
                        const activeLevel = levelRef.current ?? 1;
                        const cappedCtx = ragContext.length > 120_000 ? ragContext.slice(0, 120_000) : ragContext;
                        const prompt = `[ACTIVE LEVEL: ${activeLevel} — ${LEVELS[activeLevel].name}]\n\n[RAG CONTEXT]\n${cappedCtx}\n\n[USER QUESTION]\nTell me more about this topic. Go deeper — cover additional clinical nuances, edge cases, or advanced considerations not yet mentioned.`;
                        setIsLoading(true);
                        try {
                          let data: { response: string } | null = null;
                          try { data = await invokeChat({ message: prompt, history: historySnapshot, level: activeLevel }); } catch { data = null; }
                          if (!data?.response) {
                            setMessages(prev => [...prev, { role: "model", text: "I encountered an error. Please try again." }]);
                          } else {
                            setMessages(prev => [...prev, { role: "model", text: data.response as string }]);
                            setShowFollowUpButtons(true);
                          }
                        } catch {
                          setMessages(prev => [...prev, { role: "model", text: "I encountered an error. Please try again." }]);
                        } finally {
                          setIsLoading(false);
                        }
                      }}
                      className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 hover:border-teal-400 hover:bg-teal-50 text-slate-700 rounded-xl text-sm font-medium transition-colors shadow-sm"
                    >
                      <Sparkles size={14} className="text-teal-500" /> Learn More
                    </button>
                    <button
                      onClick={() => startQuiz("recall")}
                      className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 hover:border-teal-400 hover:bg-teal-50 text-slate-700 rounded-xl text-sm font-medium transition-colors shadow-sm"
                    >
                      <GraduationCap size={14} className="text-teal-500" /> Recall Question
                    </button>
                    <button
                      onClick={() => startQuiz("board")}
                      className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50 text-slate-700 rounded-xl text-sm font-medium transition-colors shadow-sm"
                    >
                      <BookOpen size={14} className="text-indigo-500" /> Board Style Q
                    </button>
                  </div>
                </div>
              )}

              {/* Next board question button — shown after explanation loads */}
              {showNextBoardBtn && !isLoading && (
                <div className="mr-auto">
                  <button
                    onClick={() => startQuiz("board")}
                    className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold transition-colors shadow-sm"
                  >
                    <BookOpen size={15} /> Next Question
                  </button>
                </div>
              )}

              {/* Quiz scoring panel — recall only, after user has answered */}
              {quizActive && !isLoading && quizType !== "board" && recallResponded && messages.length > 0 && messages[messages.length - 1].role === "model" && (() => {
                const activeLevel = level ?? 1;
                const baselineLevel = userProfile?.baseline_level ?? activeLevel;
                const levelDiff = Math.min(2, Math.max(0, activeLevel - baselineLevel));
                const bonusPts = LEVEL_BONUS[levelDiff] ?? 0;
                return (
                  <div className="mr-auto max-w-sm">
                    <div className="bg-white border-2 border-indigo-200 rounded-2xl p-4 shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <Target size={16} className="text-indigo-500" />
                        <div>
                          <p className="text-sm font-semibold text-slate-700">How did you do?</p>
                          {quizType && (
                            <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${quizType === "board" ? "bg-indigo-100 text-indigo-600" : "bg-teal-100 text-teal-600"}`}>
                              {quizType === "board" ? "📋 Board style" : "📝 Simple recall"}
                            </span>
                          )}
                        </div>
                        <div className="ml-auto text-right">
                          <span className="text-xs font-bold text-amber-600">+{BASE_POINTS + bonusPts} pts</span>
                          {bonusPts > 0 && <p className="text-[10px] text-indigo-500 font-medium">incl. +{bonusPts} above-baseline bonus</p>}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={() => scoreAnswer(true)} className="flex-1 flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white py-2.5 rounded-xl text-sm font-semibold transition-colors">
                          <CheckCircle size={16} /> Got it right
                        </button>
                        <button onClick={() => scoreAnswer(false)} className="flex-1 flex items-center justify-center gap-2 bg-slate-200 hover:bg-slate-300 text-slate-700 py-2.5 rounded-xl text-sm font-semibold transition-colors">
                          <XCircle size={16} /> Need review
                        </button>
                      </div>
                      {userProfile && userProfile.current_streak >= 2 && (
                        <p className="text-center text-[11px] text-orange-500 mt-2 font-medium flex items-center justify-center gap-1">
                          <Flame size={11} /> {userProfile.current_streak} streak — get it right for a bonus!
                        </p>
                      )}
                    </div>
                  </div>
                );
              })()}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 bg-white border-t border-slate-200 shrink-0">
              <div className="max-w-3xl mx-auto flex gap-2">
                <button onClick={toggleRecording} disabled={!level}
                  title={isRecording ? "Stop recording" : "Speak to fill input"}
                  className={`flex-shrink-0 w-12 h-12 flex items-center justify-center rounded-xl transition-colors ${isRecording ? "bg-red-500 hover:bg-red-600 text-white animate-pulse" : "bg-slate-100 hover:bg-slate-200 text-slate-500 disabled:opacity-40 disabled:cursor-not-allowed"}`}>
                  <Mic size={20} />
                </button>
                <textarea value={inputText} onChange={e => setInputText(e.target.value)} onKeyDown={handleKeyDown}
                  placeholder={level ? `Ask a question... (Level ${level}: ${LEVELS[level].name})` : "Select your level above to begin..."}
                  disabled={!level} rows={1}
                  className="flex-1 bg-slate-100 border-0 rounded-xl px-4 py-3 focus:ring-2 focus:ring-teal-500 focus:bg-white transition-all resize-none max-h-32 min-h-[50px] outline-none disabled:opacity-50 disabled:cursor-not-allowed" />
                <button onClick={handleSendMessage} disabled={isLoading || !inputText.trim() || !level} className="bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white rounded-xl w-12 h-12 flex items-center justify-center transition-colors shadow-sm flex-shrink-0">
                  <Send size={20} />
                </button>
              </div>
              <div className="text-center mt-2">
                <p className="text-[10px] text-slate-400">AI answers are grounded in your knowledge base. Verify all medical information independently.</p>
              </div>
            </div>
          </>
        )}

        {/* LEADERBOARD PAGE */}
        {page === "leaderboard" && <LeaderboardPage />}
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);
root.render(<App />);
