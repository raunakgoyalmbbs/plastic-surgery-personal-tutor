ALTER TABLE public.user_progress
  ADD COLUMN IF NOT EXISTS baseline_level INTEGER CHECK (baseline_level IN (1, 2, 3));
