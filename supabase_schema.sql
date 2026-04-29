create extension if not exists pgcrypto;

create table if not exists public.student_predictions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  student_name text not null,
  roll_number text not null,
  will_pass boolean not null,
  success_probability numeric(5, 2) not null,
  risk_score integer not null,
  risk_level text not null,
  academic_index integer not null,
  input_data jsonb not null,
  encoded_features jsonb not null,
  prediction jsonb not null,
  risk_factors jsonb not null,
  recommendations jsonb not null,
  feature_importance jsonb not null,
  model jsonb not null
);

create index if not exists student_predictions_created_at_idx
  on public.student_predictions (created_at desc);

create index if not exists student_predictions_roll_number_idx
  on public.student_predictions (roll_number);

alter table public.student_predictions enable row level security;

-- The Flask backend should use SUPABASE_SERVICE_ROLE_KEY, which bypasses RLS.
-- No browser/client access is granted here.
