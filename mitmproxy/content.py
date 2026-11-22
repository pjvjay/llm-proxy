"""Extract scannable text units from an OpenAI chat payload, and write redacted
text back into it.

`content` may be a string, null, or a list of parts (multimodal / agent SDKs):
    [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
The old addon assumed a string and raised TypeError on lists -- inside the
mitmproxy hook that meant the request was forwarded *unscreened*. Here each text
part is its own unit, so redaction maps back 1:1 without re-parsing."""
from dataclasses import dataclass, field

# Content-part types whose "text" field carries model-visible text.
_TEXT_PART_TYPES = {"text", "input_text", "output_text"}


@dataclass
class Unit:
    """One piece of text to screen: a whole string message or one text part."""
    msg_idx: int
    part_idx: int | None          # None => message.content is a plain string
    role: str
    text: str
    tool: str | None = None       # for tool results: the tool that produced them (decision context)
    key: str = ""                 # content hash, filled in by session.plan()
    cached: dict | None = None    # previous raw signals for this key, if any
    signals: dict = field(default_factory=dict)


def message_text(msg: dict) -> str:
    """Model-visible text of a message regardless of content shape."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") in _TEXT_PART_TYPES)
    return ""


def _tool_names(messages: list) -> dict[str, str]:
    """tool_call_id -> function name, from the assistant messages that issued them."""
    names = {}
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    names[tc["id"]] = (tc.get("function") or {}).get("name")
    return names


def units(messages: list, roles: set[str]) -> list[Unit]:
    out = []
    names = _tool_names(messages) if roles & {"tool", "function"} else {}
    for i, m in enumerate(messages or []):
        if not isinstance(m, dict) or m.get("role") not in roles:
            continue
        role = "tool" if m["role"] == "function" else m["role"]
        tool = names.get(m.get("tool_call_id")) or m.get("name") if role == "tool" else None
        content = m.get("content")
        if isinstance(content, str):
            if content:
                out.append(Unit(i, None, role, content, tool))
        elif isinstance(content, list):
            for j, p in enumerate(content):
                if isinstance(p, dict) and p.get("type") in _TEXT_PART_TYPES and p.get("text"):
                    out.append(Unit(i, j, role, p["text"], tool))
    return out


def set_unit_text(messages: list, u: Unit, text: str) -> None:
    if u.part_idx is None:
        messages[u.msg_idx]["content"] = text
    else:
        messages[u.msg_idx]["content"][u.part_idx]["text"] = text


def system_text(messages: list) -> str:
    return "\n".join(message_text(m) for m in messages or []
                     if isinstance(m, dict) and m.get("role") in ("system", "developer"))


def first_user_text(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            return message_text(m)
    return ""
