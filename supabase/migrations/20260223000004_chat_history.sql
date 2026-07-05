-- chat_history: persists per-user conversation sessions.
-- Each row is one chat session containing an ordered array of messages.
-- All access is user-scoped via RLS — users can only see and modify their own history.

CREATE TABLE IF NOT EXISTS public.chat_history (
  id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  messages   JSONB       NOT NULL DEFAULT '[]'::jsonb,  -- [{role: "user"|"model", text: string}, ...]
  topic      TEXT,        -- optional label (e.g. "Burns", "Skin Cancer", null for general chat)
  level      INTEGER      CHECK (level BETWEEN 1 AND 3)
);

-- Indexes for fast per-user queries
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
  ON public.chat_history(user_id);

CREATE INDEX IF NOT EXISTS idx_chat_history_user_updated
  ON public.chat_history(user_id, updated_at DESC);

-- Enable Row Level Security
ALTER TABLE public.chat_history ENABLE ROW LEVEL SECURITY;

-- SELECT: users can read only their own sessions
CREATE POLICY "read_own_history" ON public.chat_history
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

-- INSERT: users can only create rows where user_id = their own uid
CREATE POLICY "insert_own_history" ON public.chat_history
  FOR INSERT
  TO authenticated
  WITH CHECK (user_id = auth.uid());

-- UPDATE: users can append messages to their own sessions (e.g. updated_at, messages)
CREATE POLICY "update_own_history" ON public.chat_history
  FOR UPDATE
  TO authenticated
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

-- DELETE: users can delete their own sessions (right to erasure / GDPR compliance)
CREATE POLICY "delete_own_history" ON public.chat_history
  FOR DELETE
  TO authenticated
  USING (user_id = auth.uid());

-- Anon users: blocked from all operations (no anon policies defined)
-- Service role: bypasses RLS (for admin operations and future analytics)
