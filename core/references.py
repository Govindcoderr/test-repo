"""Resolve ``$ref`` placeholders so one tool can consume an earlier tool's output.

Without this, a dependent pair cannot be expressed. ``gmail.send_draft`` needs the
``draftId`` that ``gmail.create_draft`` produced, and a static payload has no way to
carry it:

    {"toolName": "gmail.send_draft", "input": {"draftId": "???"}}

With it:

    {
      "executionMode": "SEQUENTIAL",
      "tools": [
        {"toolCallId": "t1", "toolName": "gmail.create_draft",
         "input": {"to": "x@y.com", "subject": "Hi", "body": "..."}},

        {"toolCallId": "t2", "toolName": "gmail.send_draft",
         "input": {"draftId": {"$ref": "t1.output.draftId"}}}
      ]
    }

Four rules
----------
1. **SEQUENTIAL only.** In PARALLEL every call starts at once, so an earlier result
   does not exist yet. A ``$ref`` in a parallel batch is rejected rather than
   silently resolving to nothing.
2. **Backwards only.** A call may reference calls that already ran, never a later one.
3. **A failed dependency skips the dependent call**, with ``DEPENDENCY_FAILED``. The
   value genuinely does not exist, so running the tool with a hole in its input would
   be worse than not running it.
4. **A bad path fails loudly** with ``INVALID_REFERENCE``, never as a silent ``None``.

Kept deliberately small: substitution only. No conditionals, no loops, no
expressions. If a workflow needs those, that is the Planning Engine's job, and this
staying small is what keeps the boundary honest.
"""

from typing import Any

REF_KEY = "$ref"


class ReferenceResolutionError(Exception):
    """Base for reference problems.

    Not named ``ReferenceError`` because that is a Python builtin, and shadowing it
    inside a module that also handles ordinary exceptions would be a trap.
    """


class InvalidReferenceError(ReferenceResolutionError):
    """The reference is malformed, points at an unknown call, or the path is missing."""


class DependencyFailedError(ReferenceResolutionError):
    """The referenced call exists but did not succeed, so its output is unusable."""


def contains_reference(value: Any) -> bool:
    """True if ``value`` holds a ``$ref`` anywhere inside it."""
    if isinstance(value, dict):
        if REF_KEY in value:
            return True
        return any(contains_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_reference(item) for item in value)
    return False


def find_references(value: Any) -> list[str]:
    """Every reference string inside ``value``, for error messages and validation."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if REF_KEY in node:
                found.append(str(node[REF_KEY]))
                return
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return found


def resolve(value: Any, outputs: dict[str, dict[str, Any] | None]) -> Any:
    """Return a copy of ``value`` with every ``$ref`` replaced.

    ``outputs`` maps a completed ``toolCallId`` to that call's output, or to None if
    the call did not succeed.
    """
    if isinstance(value, dict):
        if REF_KEY in value:
            return _lookup(str(value[REF_KEY]), outputs)
        return {key: resolve(item, outputs) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve(item, outputs) for item in value]

    return value


def _lookup(reference: str, outputs: dict[str, dict[str, Any] | None]) -> Any:
    parts = reference.split(".")
    if len(parts) < 3 or parts[1] != "output":
        raise InvalidReferenceError(
            f"reference {reference!r} must look like '<toolCallId>.output.<field>'"
        )

    call_id, path = parts[0], parts[2:]

    if call_id not in outputs:
        raise InvalidReferenceError(
            f"reference {reference!r} points at {call_id!r}, which has not run. "
            "References can only look backwards, and only in SEQUENTIAL mode."
        )

    output = outputs[call_id]
    if output is None:
        raise DependencyFailedError(
            f"reference {reference!r} depends on {call_id!r}, which did not succeed"
        )

    current: Any = output
    for index, key in enumerate(path):
        walked = ".".join(path[:index]) or "output"
        if isinstance(current, list):
            current = _index_into(current, key, reference, walked)
            continue
        if not isinstance(current, dict) or key not in current:
            available = (
                sorted(current.keys()) if isinstance(current, dict) else type(current).__name__
            )
            raise InvalidReferenceError(
                f"reference {reference!r} cannot resolve {key!r} at {walked!r}; "
                f"available: {available}"
            )
        current = current[key]

    return current


def _index_into(items: list[Any], key: str, reference: str, walked: str) -> Any:
    try:
        position = int(key)
    except ValueError:
        raise InvalidReferenceError(
            f"reference {reference!r} needs a list index at {walked!r}, got {key!r}"
        ) from None

    if not -len(items) <= position < len(items):
        raise InvalidReferenceError(
            f"reference {reference!r} index {position} is out of range at {walked!r} "
            f"(length {len(items)})"
        )
    return items[position]
