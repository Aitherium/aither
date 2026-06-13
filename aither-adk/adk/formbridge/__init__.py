"""FormBridge — local form automation: capture → deterministic mapping → PDF AcroForm fill.

Everything in this package operates on the local machine only. Captured field
values NEVER leave the box: the HTTP routes reject non-loopback clients, the
MCP tools return names/counts/job-ids but never values, and there is no code
path that forwards captured content to any remote endpoint.

See .PRODUCTS/.FORMBRIDGE/ in the AitherOS repo for the product corpus
(04-ARCHITECTURE / 05-API-DESIGN / 06-SECURITY-MODEL are the design authority).
"""

from adk.formbridge.store import CaptureStore, get_store
from adk.formbridge.mapper import MappingPack, load_pack

__all__ = ["CaptureStore", "get_store", "MappingPack", "load_pack"]
