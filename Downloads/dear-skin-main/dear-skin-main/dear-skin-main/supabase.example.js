/**
 * Copy this file to supabase.js and paste your Supabase project URL + anon key.
 * https://supabase.com/dashboard → Project Settings → API
 */
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co";
const SUPABASE_ANON_KEY = "YOUR_ANON_KEY";

export const supabaseConfigured =
  !SUPABASE_URL.includes("YOUR_PROJECT") && !SUPABASE_ANON_KEY.includes("YOUR_ANON");

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
