 import os
  from pathlib import Path
  from dotenv import load_dotenv

  _root = Path(__file__).parent.parent.parent
  load_dotenv(_root / ".env")

  LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
  LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
  LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

  LOCAL_ZEP_DB_PATH = os.environ.get(
      "LOCAL_ZEP_DB_PATH",
      str(_root / "data" / "local_memory.db")
  )

  MAX_CONTENT_LENGTH = 50 * 1024 * 1024
  ALLOWED_EXTENSIONS = {"pdf", "md", "txt"}
  CHUNK_SIZE = 500
  CHUNK_OVERLAP = 50

  OASIS_MAX_ROUNDS = int(os.environ.get("OASIS_MAX_ROUNDS", "10"))
  TWITTER_ACTIONS = [
      "like_post", "retweet_post", "create_post",
      "follow_user", "unfollow_user", "mute_user",
      "unmute_user", "create_comment",
  ]
  REDDIT_ACTIONS = [
      "like_post", "dislike_post", "create_post",
      "follow_user", "unfollow_user", "create_comment",
  ]

  REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get("REPORT_AGENT_MAX_TOOL_CALLS", "30"))
  REPORT_AGENT_REFLECTION_ROUNDS = int(os.environ.get("REPORT_AGENT_REFLECTION_ROUNDS", "2"))


  def validate():
      errors = []
      if not LLM_API_KEY:
          errors.append("LLM_API_KEY is missing.")
      return errors


  class Config:
      LLM_API_KEY = LLM_API_KEY
      LLM_BASE_URL = LLM_BASE_URL
      LLM_MODEL_NAME = LLM_MODEL_NAME
      LOCAL_ZEP_DB_PATH = LOCAL_ZEP_DB_PATH
      MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
      ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
      CHUNK_SIZE = CHUNK_SIZE
      CHUNK_OVERLAP = CHUNK_OVERLAP
      OASIS_MAX_ROUNDS = OASIS_MAX_ROUNDS
      TWITTER_ACTIONS = TWITTER_ACTIONS
      REDDIT_ACTIONS = REDDIT_ACTIONS
      REPORT_AGENT_MAX_TOOL_CALLS = REPORT_AGENT_MAX_TOOL_CALLS
      REPORT_AGENT_REFLECTION_ROUNDS = REPORT_AGENT_REFLECTION_ROUNDS
