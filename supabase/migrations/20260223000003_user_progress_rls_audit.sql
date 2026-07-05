-- Idempotent audit/enforcement of user_progress RLS policies.
-- The table and initial policies were created in 20260221000000_create_user_progress.sql.
-- This migration re-states them explicitly for clarity and drops/recreates to ensure correctness.

ALTER TABLE public.user_progress ENABLE ROW LEVEL SECURITY;

-- Any authenticated user can read all rows (required for leaderboard feature)
DROP POLICY IF EXISTS "read_all_progress" ON public.user_progress;
CREATE POLICY "read_all_progress" ON public.user_progress
  FOR SELECT
  TO authenticated
  USING (true);

-- Users can only insert their own row (id must match their auth.uid())
DROP POLICY IF EXISTS "insert_own_progress" ON public.user_progress;
CREATE POLICY "insert_own_progress" ON public.user_progress
  FOR INSERT
  TO authenticated
  WITH CHECK (id = auth.uid());

-- Users can only update their own row
DROP POLICY IF EXISTS "update_own_progress" ON public.user_progress;
CREATE POLICY "update_own_progress" ON public.user_progress
  FOR UPDATE
  TO authenticated
  USING (id = auth.uid())
  WITH CHECK (id = auth.uid());

-- No DELETE policy: users cannot delete progress rows (admin-only via service role if needed)
-- Anon users: blocked from all operations (no anon policies defined)
