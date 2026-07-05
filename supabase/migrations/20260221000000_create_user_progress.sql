CREATE TABLE IF NOT EXISTS public.user_progress (
  id UUID PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  total_points INTEGER DEFAULT 0,
  correct_answers INTEGER DEFAULT 0,
  total_questions INTEGER DEFAULT 0,
  current_streak INTEGER DEFAULT 0,
  best_streak INTEGER DEFAULT 0,
  preferred_level INTEGER CHECK (preferred_level IN (1, 2, 3)),
  last_active TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.user_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY "View leaderboard" ON public.user_progress
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Insert own row" ON public.user_progress
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = id);

CREATE POLICY "Update own row" ON public.user_progress
  FOR UPDATE TO authenticated USING (auth.uid() = id);
