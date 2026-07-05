-- Drop UNIQUE constraint on username — display names don't need to be unique.
-- The primary key (id = auth.uid()) is the real identity. This prevents 409 conflicts
-- when an admin deletes and recreates a user (new UID, same display name).
ALTER TABLE public.user_progress DROP CONSTRAINT IF EXISTS user_progress_username_key;
