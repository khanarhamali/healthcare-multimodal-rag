import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "skincare-images")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(" Missing Supabase configuration! Please check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
