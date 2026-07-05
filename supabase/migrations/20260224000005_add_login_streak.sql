ALTER TABLE public.user_progress
  ADD COLUMN IF NOT EXISTS login_streak INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_login_date TEXT;  -- YYYY-MM-DD
