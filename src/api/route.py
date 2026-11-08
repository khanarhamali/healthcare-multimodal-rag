import os
import sys
from fastapi import APIRouter

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..")) if "src" in current_dir else current_dir
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.api.chat_routes import chat_router
from src.api.thread_routes import thread_router

# Parent router without duplicate tag so endpoints appear only once under their specific sections
router = APIRouter(prefix="/api")
router.include_router(chat_router)
router.include_router(thread_router)
