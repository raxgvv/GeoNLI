"""Request identifiers shared by Flask routes and model timing logs."""
from contextvars import ContextVar

REQUEST_ID = ContextVar("geonli_request_id", default="-")


def current_request_id() -> str:
    return REQUEST_ID.get()
