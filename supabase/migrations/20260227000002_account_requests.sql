-- Account request table: users submit registration requests, admin approves via Dashboard
CREATE TABLE account_requests (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  display_name text NOT NULL,
  email text NOT NULL,
  status text DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  created_at timestamptz DEFAULT now()
);

-- Allow anonymous inserts (no auth needed to request an account)
ALTER TABLE account_requests ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can request an account"
  ON account_requests FOR INSERT
  TO anon WITH CHECK (true);
-- No SELECT/UPDATE/DELETE policies — only viewable via Supabase Dashboard (service role)
