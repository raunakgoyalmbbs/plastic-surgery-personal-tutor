# Plastic Surgery Personal Tutor

Plastic Surgery Personal Tutor is an open-source, **agentic AI** tutoring application for plastic surgery education. It is built on **Graph RAG** (graph-based retrieval-augmented generation): a curated corpus of plastic-surgery literature — textbooks and their references — is indexed into a **knowledge graph** that maps clinical entities (anatomical structures, procedures, concepts) and the relationships between them, so that every response is generated from content retrieved from that verified knowledge base rather than the language model's pretrained knowledge. This grounding guards against *hallucination* — the plausible-sounding but fabricated information that makes generic AI chatbots unreliable for medical education.

Beyond answering questions, the tutor behaves as an educational *agent*: it reasons through multi-step tasks, retrieves from the knowledge base during a session, adapts its instruction to the learner's training level (medical student, junior resident, senior resident), autonomously generates clinical vignettes and multiple-choice quizzes on recently studied topics, and supports study through text and multilingual audio. The system is intended for surgical trainees as learners, and for surgical educators who wish to reproduce the pipeline with their own document corpus to build an institution-specific tutor.

## How it works

Before the technical details, here is the overall idea explained without jargon. The system is made of a few parts that work together:

- **The frontend** is what you see and use: the website that opens in your browser, where you type questions, take quizzes, or speak with the Tutor out loud. It runs on your screen and holds no secrets.
- **The backend** is the behind the scenes helper running on the internet. When you ask something, the backend receives your question, looks up the relevant material, and talks to the artificial-intelligence (AI) services on your behalf. You never see it, but it does the heavy lifting and safely keeps the secret keys that the AI services require.
- **Indexing (building the knowledge base)** is a one-time preparation step you do before the Tutor can help. The system reads through your own PDF documents, breaks them into small passages, and organises them so they can later be searched *by meaning*. A helpful way to picture it: a librarian patiently reads every book once and builds a detailed index, so that afterwards any topic can be found in an instant. You only repeat this when you add new documents.
- **Retrieval (searching by meaning)** is what happens each time you ask a question. Instead of matching exact words, the system finds the passages from your documents whose *meaning* is closest to your question — so a question phrased differently from the text can still find the meaningful right answer.
- **Retrieval-augmented generation (RAG), and here *Graph RAG*,** is the central idea that keeps the tutor trustworthy. When you ask a question, the system *first* retrieves the most relevant passages from your own documents — and, because it uses a knowledge *graph*, also the connected concepts around them — and *then* asks the AI to write an answer using that material. Because the answer is built from your teaching content rather than the AI's general memory, the tutor stays grounded in your sources and is far less likely to invent facts or fabricate information (sometimes called "hallucinating").
- **The knowledge graph** is an extra map, also built during indexing, that records how the key concepts in your documents relate to one another — for example, linking a nerve to the muscle it supplies and the procedures that may place it at risk. It lets the tutor connect related ideas and give richer, better-grounded answers.
- **An "agentic" tutor** means the application does more than reply to a single question — it can carry out multi-step tasks on its own initiative: deciding when to search the knowledge base mid-conversation, tailoring its explanations to your training level, generating quiz questions from what you have just studied, and holding a spoken back-and-forth in your chosen language. In short, it behaves more like an attentive study partner or a knowledgable attending than a static chatbot.

In short: you prepare your documents once (indexing), and from then on every question is answered by finding the most relevant parts of *your* material and asking the AI to explain them (Graph RAG), with the tutor adapting to you and generating practice as an *agent* rather than a passive chatbot. The rest of this document explains how to set all of this up for yourself.


## System overview

The system consists of three layers:

1. **Frontend** — a single-page React application (Vite, TypeScript, Tailwind CSS). It contains no API keys; the production bundle holds only the public Supabase project URL and anonymous key.
2. **Backend** — three [Supabase Edge Functions](https://supabase.com/docs/guides/functions) (Deno):
   - `chat` — sends the conversation, together with retrieved context, to Google Gemini for answer synthesis;
   - `rag-query` — embeds the user's query with OpenAI `text-embedding-3-small`, performs vector similarity search over document chunks, entities, and relationships stored in Supabase Postgres (`pgvector`), performs a one-hop knowledge-graph traversal, and returns the combined context;
   - `gemini-token` — serves the Gemini API key to authenticated users for the real-time audio tutoring mode.
3. **Indexing pipeline (offline, laptop-only)** — a vendored copy of [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS, MIT license) under `LightRAG/`. It chunks PDF documents, extracts entities and relationships with OpenAI `gpt-4o-mini`, and embeds all text with OpenAI `text-embedding-3-small`. The resulting vectors are then migrated into Supabase Postgres with `scripts/migrate_to_supabase.py`. End users of the deployed application never interact with this pipeline; it is required only when the corpus is built or updated.

At query time, the flow is: user message → `rag-query` (embedding + pgvector search + graph traversal) → retrieved context + message → `chat` (Gemini synthesis) → answer. Audio mode additionally pre-loads topic-specific context into the Gemini Live API session and exposes a `search_knowledge_base` tool that the model can call mid-conversation.

A detailed technical description is provided in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Key terms

Welcome — this guide is written to be friendly to complete beginners, and no prior technical experience is needed. A few technical words come up throughout the walkthrough, so we have gathered them here with plain-language explanations. There is no need to memorise anything; please feel free to read this section once and then refer back to it whenever a term is unclear.

- **Terminal (or command line)** — a simple text window where you type instructions one line at a time, instead of clicking buttons. On a Mac it is the **Terminal** app (press `Cmd+Space`, type "Terminal", and press Enter); on Windows it is **PowerShell** (click Start, type "PowerShell", and press Enter). To run an instruction, you type or paste it and press Enter — that is all there is to it.
- **Command** — a single line of text you give the terminal to carry out a task, for example `node --version`. Throughout this guide, anything shown in a grey box is a command you can copy and paste exactly as written.
- **API (Application Programming Interface)** — simply a way for one program to ask another program for help, often over the internet. Here, the app uses APIs to send text to Google's and OpenAI's artificial-intelligence services and receive answers back. You do not need to set any of this up by hand; the app handles it for you.
- **API key** — think of this as a private password that identifies your account with an AI provider and lets your usage be billed to you. It should be kept secret, much like a credit-card number. This project always keeps keys safely on the server, never in the public code.
- **Token** — the unit AI providers use to measure and price usage. As a rough guide, one token is about three-quarters of a word, and you are charged per token of text sent and received (see [API usage and cost](#api-usage-and-cost)).
- **Large language model (LLM)** — the artificial-intelligence system (here, Google Gemini and OpenAI models) that reads text and writes helpful, human-like responses.
- **Repository (often shortened to "repo")** — simply the folder that holds all of this project's files: the code, the documents, and the instructions. "Downloading the repository" just means copying that folder onto your own computer.
- **Node.js** — free software that lets your computer run the part of the app you see in the browser. You install it just once, in Phase 1.
- **Python** — a free, widely used programming language needed only for the offline indexing step in Phase 4. You also install this just once.
- **`npm` / `npx`** — helper tools that come bundled with Node.js. `npm` downloads the building blocks the app relies on, and `npx` runs a tool without installing it permanently. You will only ever copy and paste these, so there is no need to understand how they work.
- **Database** — an organised store of information kept on a server. This project uses one (provided by Supabase) to hold user accounts, chat history, and the searchable knowledge base.
- **Supabase** — a friendly online service used here to provide the database, user log-in, and small server-side programs, so that you do not have to run your own server.
- **Frontend and backend** — the *frontend* is the part you see and interact with in your browser; the *backend* is the behind-the-scenes part that runs on servers, managing data and talking to the AI services.
- **Deploy (or deployment)** — publishing the finished app to the internet so that others can open it in their browser. This is covered in Phase 5, using a free service called Vercel.
- **Corpus** — the collection of source documents (here, plastic-surgery PDFs) that the tutor learns from.
- **Embedding (or vector)** — a way of turning a piece of text into a list of numbers that captures its meaning, so the computer can find passages that mean something *similar* to a question, rather than only matching exact words. This is why Phase 4 asks you to use the same AI model for indexing and for searching: the number-lists must come from the same model to be comparable.
- **Knowledge graph** — a map of the key concepts in your documents and how they connect to one another, built automatically during indexing to help the tutor give richer, better-grounded answers.
- **Retrieval-augmented generation (RAG)** — the overall approach this system uses: it first *finds* the most relevant passages in your own documents, then asks the AI to answer *using those passages*, so the replies stay grounded in your material rather than the model's general memory.

## Reproducing the system

The following walkthrough is written so that it can be comfortably followed without any prior programming experience. **If any term is unfamiliar, you are warmly encouraged to check [Key terms](#key-terms) above.** Every command can simply be copied and pasted exactly as shown into a terminal (on a Mac: open the **Terminal** application; on Windows: use **PowerShell**), and run by pressing Enter.

The reproduction has six phases:

| Phase | What it does | Where it runs |
|---|---|---|
| 1 | Install prerequisites and obtain API keys | Your computer + provider websites |
| 2 | Create and configure the Supabase project (database + auth) | supabase.com |
| 3 | Store API keys as server secrets and deploy the backend functions | Your computer (terminal) |
| 4 | Index your own PDF corpus into the knowledge base | Your computer (laptop required) |
| 5 | Run the app locally and deploy it to the web | Your computer + vercel.com |
| 6 | Manage user access | Supabase dashboard |

### Phase 1 — Prerequisites

1. **Install Node.js** (the JavaScript runtime used to build and run the frontend). Download the "LTS" installer from https://nodejs.org and run it, accepting the defaults. To confirm it worked, open a terminal and run:

   ```bash
   node --version
   ```

   This prints the installed version (e.g. `v22.x.x`). Any version 20 or newer is sufficient.

2. **Install Python 3** (needed only for Phase 4, the indexing pipeline). macOS usually ships with it; otherwise download it from https://www.python.org/downloads (version 3.10 or newer). Confirm with:

   ```bash
   python3 --version
   ```

3. **Create a free Supabase account** at https://supabase.com (click "Start your project" and sign up). Supabase provides the database, user authentication, and the serverless functions used by the backend.

4. **Obtain a Google Gemini API key** at https://aistudio.google.com/apikey (sign in with a Google account, click "Create API key", and copy the key somewhere safe). Gemini powers the chat, quiz, and audio tutoring.

5. **Obtain an OpenAI API key** at https://platform.openai.com/api-keys (sign in, click "Create new secret key", and copy it). OpenAI is used for text embeddings (every query) and for entity extraction during indexing (Phase 4). Note that indexing requires a paid OpenAI account with billing enabled.

6. **Download this repository.** If you have `git` installed:

   ```bash
   git clone https://github.com/raunakgoyalmbbs/plastic-surgery-personal-tutor.git
   cd plastic-surgery-personal-tutor
   ```

   Alternatively, click "Code → Download ZIP" on the repository page, unzip it, and open a terminal in the unzipped folder (`cd` followed by the folder path).

### Phase 2 — Set up the Supabase project

1. In the Supabase dashboard (https://supabase.com/dashboard), click **New project**. Choose any name (e.g. "personal-tutor"), set a strong database password (save it — Phase 4 needs it), pick the region closest to you, and click **Create new project**. Provisioning takes 1–2 minutes.

2. **Record your project's URL and anonymous key.** In the dashboard, go to **Project Settings → API** (gear icon in the left sidebar). Copy:
   - the **Project URL** (looks like `https://your-project-ref.supabase.co`), and
   - the **anon / public** API key (a long string starting with `eyJ...`).

   These two values are public by design and will go into the frontend configuration in Phase 5.

3. **Run the database migrations.** The files in `supabase/migrations/` create the tables (user progress, chat history, access requests, knowledge-base vector tables), enable the `pgvector` extension, define the search functions, and apply the row-level-security policies. The recommended way is via the Supabase command-line tool, which is bundled with this repository (`npx` runs it without a separate installation):

   ```bash
   npx supabase login
   ```

   This opens a browser window asking you to authorize the command-line tool with your Supabase account. Then link the repository to your project (replace `your-project-ref` with the short identifier visible in your Project URL):

   ```bash
   npx supabase link --project-ref your-project-ref
   ```

   You will be asked for the database password chosen in step 1. Finally, apply all migrations in order:

   ```bash
   npx supabase db push
   ```

   This executes each SQL file in `supabase/migrations/` against your database, oldest first.

   *Alternative without the command line:* in the dashboard, open **SQL Editor → New query**, then open each file in `supabase/migrations/` in a text editor, paste its contents, and click **Run** — strictly in filename (date) order.

4. **Disable public self-registration.** In the dashboard, go to **Authentication → Sign In / Providers → Email** and turn **off** "Allow new users to sign up". The application uses an invitation-based flow instead (Phase 6).

### Phase 3 — Configure secrets and deploy the backend functions

The two AI provider keys are stored as **Edge Function secrets** on Supabase servers. They are never placed in the frontend code or in any file that is committed to the repository.

1. Set the secrets (replace the placeholders with your actual keys from Phase 1):

   ```bash
   npx supabase secrets set GEMINI_API_KEY=your-gemini-key
   npx supabase secrets set OPENAI_API_KEY=your-openai-key
   ```

   Each command uploads one key to Supabase's encrypted secret store, where only your Edge Functions can read it.

2. Deploy the three backend functions:

   ```bash
   npx supabase functions deploy chat rag-query gemini-token
   ```

   This uploads the code in `supabase/functions/` to Supabase's servers, where it runs on demand. Re-run this command whenever the function code changes.

### Phase 4 — Build your knowledge base (laptop-only indexing)

This phase converts your own PDF corpus into the searchable knowledge base. It runs entirely on your own computer and only needs to be repeated when documents are added. It incurs paid OpenAI usage (entity extraction and embedding are billed per token; the cost scales with corpus size — see [API usage and cost](#api-usage-and-cost)). End users of the deployed application do **not** need this phase or a running laptop; all their queries are served from Supabase.

1. **Set up the Python environment** for the vendored LightRAG server:

   ```bash
   cd LightRAG
   python3 -m venv .venv
   .venv/bin/pip install -e ".[api]"
   ```

   The first command enters the `LightRAG/` folder; the second creates an isolated Python environment in `.venv/`; the third installs LightRAG and its API server into it.

2. **Configure LightRAG.** Copy the template configuration and open it in a text editor:

   ```bash
   cp .env.example .env
   ```

   In `LightRAG/.env`, fill in:
   - your OpenAI API key(s) — extraction uses OpenAI `gpt-4o-mini` and embedding uses OpenAI `text-embedding-3-small` (1536 dimensions). **The embedding model must never change after the first document is indexed**, because the query-time embeddings must match the stored vectors exactly;
   - your Supabase Postgres connection details (`POSTGRES_HOST=db.your-project-ref.supabase.co`, the database password from Phase 2, etc.), used later by the migration script.

   No local LLM (e.g. Ollama) is required at any point; all model calls during indexing go to OpenAI in the reference configuration. If you would prefer to avoid commercial APIs entirely, LightRAG can instead be pointed at a local model — see [Using free, local models instead](#using-free-local-models-instead) for how, and for the important limitations.

   > **Use the same models for indexing and retrieval.** Vector search only works when the query vector and the stored vectors come from the *identical* embedding model. The model configured here to embed your documents (OpenAI `text-embedding-3-small`, 1536 dimensions) must therefore be the same one the `rag-query` Edge Function uses to embed queries at retrieval time — in the reference configuration they already match, so do not change one without changing the other. If you switch the embedding model or its dimensions, the query embeddings will no longer align with the stored knowledge-graph vectors, retrieval will silently return wrong or empty results, and **the entire corpus must be re-indexed from scratch**. The same discipline applies to the `gpt-4o-mini` extraction model: keep it consistent across indexing runs so entities and relationships are extracted uniformly into the knowledge graph.

3. *(Optional, for large corpora)* **Start the key-rotation proxy.** `scripts/openai_proxy.py` round-robins extraction requests across up to three OpenAI keys to raise the effective rate limit. If you use a single key and a modest corpus, you may skip this and point `LLM_BINDING_HOST` in `LightRAG/.env` directly at `https://api.openai.com/v1`.

   ```bash
   python3 ../scripts/openai_proxy.py &
   ```

4. **Start the LightRAG server:**

   ```bash
   .venv/bin/lightrag-server
   ```

   Leave this terminal window open while indexing.

5. **Upload your PDFs.** Open http://localhost:9621 in a browser — this is LightRAG's administration interface. Go to the **Documents** tab, click **Upload**, and select your PDF files. For each document, LightRAG splits the text into chunks, calls OpenAI `gpt-4o-mini` to extract entities and relationships (building a knowledge graph), and calls OpenAI `text-embedding-3-small` to embed the chunks, entities, and relationships. Indexing a full textbook can take from tens of minutes to several hours. Note that scanned PDFs without a text layer must be OCR-processed externally before upload.

6. **Migrate the vectors to Supabase.** Once indexing has completed, stop the server (press `Ctrl+C` in its terminal) and run, from the repository root:

   ```bash
   cd ..
   pip3 install psycopg2-binary python-dotenv numpy
   python3 scripts/migrate_to_supabase.py
   ```

   The script reads the locally stored vectors (`LightRAG/rag_storage/`) and bulk-loads them into the Supabase Postgres tables, then builds the HNSW similarity-search indexes. A `--dry-run` flag is available to count rows without writing. The script is idempotent and can be re-run after adding documents.

### Phase 5 — Run locally and deploy

1. **Configure the frontend.** From the repository root, copy the environment template:

   ```bash
   cp .env.local.example .env.local
   ```

   Open `.env.local` in a text editor and paste in the Project URL and anon key recorded in Phase 2:

   ```
   VITE_SUPABASE_URL=https://your-project-ref.supabase.co
   VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
   ```

2. **Install dependencies and start the development server:**

   ```bash
   npm install
   npm run dev
   ```

   `npm install` downloads the JavaScript libraries the app depends on (one-time); `npm run dev` starts a local web server. Open http://localhost:3000 in a browser to use the app.

3. **Deploy to the web (Vercel).** Create a free account at https://vercel.com and click **Add New → Project**, importing this repository from your GitHub account (push the repository to your own GitHub first). Vercel auto-detects the Vite framework; the build command is `npm run build` and the output directory is `dist/`. Under **Settings → Environment Variables**, add the same two variables as in step 1 (`VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`), then trigger a deployment. Subsequent pushes to the configured production branch redeploy automatically.

### Phase 6 — Managing access

The application uses a single in-app login screen and an invitation-based registration flow. The person who deploys the app owns the Supabase project and acts as the administrator.

1. A prospective user clicks **Request access** on the login screen and submits their **name and email address only** — the application stores no passwords for requests.
2. The request is written to the `account_requests` table, which is readable only by the administrator (service role); it is not visible to other users.
3. The administrator reviews requests in the Supabase dashboard under **Table Editor → account_requests**.
4. To approve a request, the administrator goes to **Authentication → Users → Invite user** (or "Add user → Send invitation") and enters the requester's email. Supabase emails the person a link through which they set their own password.
5. The user can then sign in through the app's login screen.

## API usage and cost

The system depends on two external AI providers at runtime and during indexing.

> **These are paid, metered services — using them costs money beyond the free tier.** Google Gemini and OpenAI bill your account per token once you exceed their free allowances. Google AI Studio offers a rate-limited free tier and OpenAI sometimes grants introductory credit, but these cover only light, single-user evaluation. In particular, **indexing a corpus (Phase 4) is billed OpenAI usage with no free tier** — a paid OpenAI account with billing enabled is required, and the cost scales with corpus size. Before indexing or opening the app to users, set a hard spending limit in each provider's billing dashboard (OpenAI → *Settings → Limits*; Google Cloud → *Billing → Budgets & alerts*) so runaway usage cannot generate an unexpected bill.

> **Disclaimer:** Pricing and free-tier limits change frequently; the figures below reflect publicly listed rates as of 2026-07-05 and should be re-verified against the providers' official pricing pages before use.
> Sources: Google AI pricing — https://ai.google.dev/pricing ; OpenAI pricing — https://openai.com/api/pricing

| Service | Model | Role in this system | Pricing structure |
|---|---|---|---|
| Google Gemini | `gemini-2.5-flash` | Chat answer synthesis; quiz generation | Billed per input/output token on the paid tier; Google AI Studio also offers a free tier with request-rate limits that is sufficient for light single-user evaluation. Flash-class models are among the lowest-priced in the Gemini family. |
| Google Gemini Live | `gemini-2.5-flash-native-audio-preview-12-2025` | Real-time audio tutoring (bidirectional audio) | Billed per token on the paid tier, with audio input/output tokens priced higher than text tokens — real-time audio is the most expensive interaction mode in the system. However, Google's free tier currently allows roughly **50 Live API (audio) sessions per day at no cost**, which is enough to evaluate the audio tutoring mode with few users without any charges. |
| OpenAI Embeddings | `text-embedding-3-small` | Query embedding (every retrieval); chunk/entity/relationship embedding (indexing) | Billed per input token, at a very low rate (on the order of cents per million tokens). Query-time cost is negligible; indexing cost scales linearly with corpus size. |
| OpenAI | `gpt-4o-mini` | Entity/relationship extraction during PDF indexing only | Billed per input/output token at small-model rates. This is typically the dominant cost of building the knowledge base; it is incurred once per document, and repeated re-processing of the same document is served from LightRAG's local LLM cache at no additional cost. |

Practical notes:

- End users of a deployed instance generate only Gemini (chat/audio) and OpenAI embedding (query) charges; no OpenAI extraction calls occur at query time.
- The audio tutoring mode has a useful free allowance: Google's free tier currently permits about **50 Gemini Live (audio) sessions per day at no cost**, so the most expensive interaction mode can still be trialled for free by learners. This allowance is set by Google and subject to change (see the disclaimer above).
- Query synthesis is deliberately performed by Gemini in the `chat` function, not by LightRAG, so no LLM is invoked inside the retrieval layer at query time (`only_need_context` mode).
- Supabase's free tier accommodates a corpus of the scale described here (roughly 5,000 chunks and 175,000 entity/relationship vectors in the example deployment), though building the HNSW indexes benefits from a paid compute tier.

### Indexing cost and time (rough estimate)

The figures below are **planning estimates, not measured results**, intended to give a sense of scale before you commit to indexing. Actual cost and time vary substantially with page density, chunk size, the number of extraction "gleaning" passes, your API rate-limit tier, and — for local models — your hardware. Entity/relationship extraction, not embedding, is the dominant cost, because LightRAG sends each chunk through a large extraction prompt one or more times; embedding the same text with OpenAI `text-embedding-3-small` adds a negligible amount (well under $1 even for a whole textbook).

**At the small end — a single research article.** A 6–10 page article is roughly 4,000–8,000 tokens, which at the configured `CHUNK_SIZE=1200` is only about **5–7 chunks**. With OpenAI `gpt-4o-mini` this costs on the order of **$0.01–0.05 (one to five cents) and takes 1–3 minutes** — effectively free. This is what makes the knowledge base cheap to extend: adding new guidelines or papers one at a time costs pennies, so the corpus can grow continuously alongside the field.

**At the large end — a full textbook.** A ~4,000-page textbook holds roughly 3–4 million tokens, producing on the order of **3,500–4,500 chunks** (for comparison, the example deployment's full corpus of textbooks plus references was ~5,150 chunks). This is the case worth planning a budget and an overnight run for:

| Extraction model (via LightRAG) | Approx. API cost (4,000-page textbook) | Approx. wall-clock time on a laptop |
|---|---|---|
| **OpenAI `gpt-4o-mini`** *(what this project used)* | **~$10–25** | ~4–10 h with a single key; ~2–5 h with the 3-key `openai_proxy.py` — a comfortable overnight run |
| Google Gemini 2.5 Flash | ~$20–45 | similar (API-side, rate-limit bound) |
| Frontier models (GPT-4o, Claude Sonnet, Gemini 2.5 Pro) | ~$150–500 | similar or longer; rarely worth it for extraction |
| Local via [Ollama](https://ollama.com) (e.g. Llama 3.1 8B + `nomic-embed-text`) | **$0 API** | ~1–4 days on a typical Apple-Silicon/CPU laptop; an overnight run is realistic only with a strong discrete GPU |

The commercial small-model path (`gpt-4o-mini`) is good value for most users: a few tens of dollars for a whole textbook, finished by morning, and few cents per added article. The local path removes API cost entirely and keeps data on your machine, but on laptop-class hardware the extraction step is slow enough that a full textbook can take **several days rather than one night** — the trade-off discussed in [Using free, local models instead](#using-free-local-models-instead) below. Commercial prices change frequently; re-verify against the pricing pages linked above, and if you have measured the cost and time of your own run, please substitute those real numbers here.

### Using free, local models instead

If you would rather not use commercial APIs — to avoid cost, or to keep sensitive documents on your own hardware — it is possible to substitute free, locally run models such as those served by [Ollama](https://ollama.com) (which can run open-weight models like Llama, Qwen, Mistral, and local embedding models such as `nomic-embed-text`). **We did not use local models for this project**; all development and the published evaluation were carried out with the commercial models described above (Google Gemini and OpenAI). The notes below are therefore guidance, not a supported-and-tested path.

**What is straightforward.** The offline indexing pipeline (Phase 4) is the natural place for a local model. LightRAG already ships with adapters for Ollama and other OpenAI-compatible local endpoints, so both the extraction model (`gpt-4o-mini` in the reference setup) and the embedding model (`text-embedding-3-small`) can be repointed at a local server by changing the binding settings in `LightRAG/.env`. This lets you build the knowledge base with **no OpenAI charges and no data leaving your computer**.

**What is not straightforward — the deployed web app.** The live application's retrieval (`rag-query`) and chat (`chat`) run as Supabase Edge Functions in the cloud, and the audio mode uses the Google Gemini Live API. A model running locally on your laptop (Ollama listens on `localhost`) is **not reachable from those cloud functions**. Using local models for the *running app*, rather than just for indexing, would require re-architecting the backend — for example self-hosting it somewhere that can reach your model, or exposing your local model over the internet — which is beyond this reference implementation. There is also no drop-in local replacement for the real-time Gemini Live audio mode.

**Constraints to respect.**

- **Embedding consistency still applies.** If you index with a local embedding model, the query side must use the *same* model, and the pgvector column dimension must match that model's output size (the reference schema is built for 1536 dimensions; a different local embedding model with a different dimension requires adjusting the migrations and re-indexing). See the embedding-consistency note in Phase 4.
- **Quality is generally lower.** Open-weight models that run on consumer hardware typically produce weaker entity/relationship extraction and less accurate answers than frontier commercial models. The accuracy, completeness, and concept-coverage results reported for this project **do not carry over** to a local-model deployment and would need to be re-established.
- **Hardware and speed.** Capable local models need a reasonably powerful machine (a modern GPU or ample RAM); on modest laptops, indexing and responses can be slow.

In short: local models are a reasonable, cost-free, privacy-preserving option **for building the knowledge base**, at some cost to answer quality; using them for the full live application is possible but requires additional engineering that this repository does not provide out of the box.

## Responsible use, privacy, and security

**Educational use only.** This application is an educational aid for trainees. It is **not** a validated clinical decision-support tool. The underlying AI models can produce inaccurate or fabricated information, so all output should be verified against authoritative sources before being relied upon. **Do not enter patient-identifiable information or other protected health information (PHI) into the application.**

**API tiers and data privacy.** The app requires your own API key. Free API tiers carry per-minute and per-day request limits that will throttle use across multiple learners, and these limits can change without notice. More importantly, on most free tiers the provider may use submitted prompts and responses to improve their models — so the free tier should be treated as evaluation-only, and no sensitive content should be entered while using it. For any real deployment serving a training program, consider using a commercial API key, which provides both reliable capacity and a data-use policy that excludes your content from model training. Do not attempt to bypass rate limits by creating multiple keys or projects; this violates provider terms of service.

**Security and warranty.** The application includes standard authentication and input-validation measures, but no system is fully secure. Responsibility for securing a deployed instance — including protecting API keys, rotating credentials, and applying updates — rests with the deploying program. This software is provided "as is," without warranty of any kind, as set out in the [`LICENSE`](LICENSE) file. The maintainers accept no liability for any loss or harm arising from its use.

### Technical safeguards in this implementation

The reference implementation includes the following protections; anyone deploying an instance is responsible for keeping them in place.

- **API keys never reach the browser.** `GEMINI_API_KEY` and `OPENAI_API_KEY` exist only in Supabase Edge Function secrets. The frontend bundle contains only the Supabase project URL and anonymous key, both of which are public by design. The build configuration injects no other secrets.
- **Row-Level Security (RLS) on all user data.** `chat_history` is readable, writable, and deletable only by the owning user (supporting data-erasure requests). `user_progress` is readable by authenticated users (to support a leaderboard) but writable only by the owning user. `account_requests` accepts inserts from anyone (the request form) but is readable by no client role. The public anonymous key is safe to expose *because* these RLS policies enforce access — do not disable RLS.
- **Knowledge-base tables are not directly accessible.** All `LIGHTRAG_*` tables have RLS enabled with no policies, which blocks every direct client query. Retrieval is possible only through `SECURITY DEFINER` SQL functions invoked by the `rag-query` Edge Function using the service role.
- **Access is invitation-gated.** Public self-registration is disabled; accounts exist only after an administrator invitation (Phase 6). The access-request form collects name and email only — the application never stores passwords; password creation is handled entirely by Supabase's invitation email flow.
- **Audio sessions are authentication-gated.** The `gemini-token` function returns the Gemini key only to requests carrying a valid bearer token, so unauthenticated visitors cannot obtain it.
- Edge Function CORS is permissive (`*`) in the reference configuration and should be restricted to the deployed frontend origin in production.

## Citing this work

If you use this software in academic work, please cite it via its DOI.

<!-- DOI badge added after first Zenodo release -->

A machine-readable citation file, [`CITATION.cff`](CITATION.cff), is included in the repository root; GitHub's "Cite this repository" button and reference managers can consume it directly.

## License and acknowledgements

This project is released under the **MIT License** (see [`LICENSE`](LICENSE)).

It builds on the following third-party components, whose authors are gratefully acknowledged:

- **[LightRAG](https://github.com/HKUDS/LightRAG)** (HKUDS, MIT License) — graph-based retrieval-augmented generation framework; a trimmed copy is vendored under `LightRAG/` for the offline indexing pipeline.
- **Google Gemini** — chat synthesis, quiz generation, and the Live API for real-time audio tutoring.
- **OpenAI** — `text-embedding-3-small` embeddings and `gpt-4o-mini` entity extraction.
- **[Supabase](https://supabase.com)** — Postgres (with `pgvector`), authentication, and Edge Functions.
- **[React](https://react.dev)**, **[Vite](https://vite.dev)**, and **[Tailwind CSS](https://tailwindcss.com)** — frontend framework, build tooling, and styling.
- **[lucide-react](https://lucide.dev)** — icon set.
