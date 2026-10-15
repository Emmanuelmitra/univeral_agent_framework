"""Application package exposing `root_agent`."""

from dotenv import load_dotenv
load_dotenv()

from app.core.engine import build_agent_from_cartridge

root_agent = build_agent_from_cartridge()
