"""Status and mode vocabulary for the Tool Engine.

Values are UPPERCASE to match the AI OS platform convention (``ResponseStatus``,
``EngineStatus``, ``RequestType`` in the kernel repo are all uppercase).

These are the engine's *internal* vocabulary. They travel inside the free-form
``result`` dict of a ``KernelResponse``, not as its top-level ``status``. The
kernel's ``ResponseStatus`` has only SUCCESS / FAILURE / PARTIAL / TIMEOUT /
CANCELLED and no DENIED, which is why per-tool detail has to live in ``result``.
"""

from enum import Enum


class ToolSource(str, Enum):
    """Where a tool is implemented.

    NATIVE — built-in Python function registered in the native runner.
    MCP    — served by an MCP server subprocess.
    API    — external REST/HTTP endpoint; the executor builds the call
             based on the tool's specific API contract.
    """

    NATIVE = "NATIVE"
    MCP = "MCP"
    API = "API"


class ToolStatus(str, Enum):
    """Outcome of a single tool call.

    DENIED and FAILED are kept distinct on purpose: DENIED means the tool was
    never attempted because the caller lacked permission, FAILED means it ran and
    errored. Collapsing them loses information that matters when debugging
    permissions.
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    # Gateway suspended the call pending a human approval decision.
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    # Caller has exceeded their quota for this tool/tenant.
    RATE_LIMITED = "RATE_LIMITED"


class AggregateStatus(str, Enum):
    """Overall outcome of a batch of tool calls.

    Deliberately mirrors the kernel's ``ResponseStatus`` values one-for-one so the
    adapter can map without a lookup table.
    """

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILURE = "FAILURE"


class ExecutionMode(str, Enum):
    """How to run multiple tools: one after another, or all at once."""

    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"


class OnFailure(str, Enum):
    """What to do when a tool call does not succeed.

    Orthogonal to ExecutionMode. All four combinations are meaningful, including
    PARALLEL + ABORT, which cancels calls that have not finished yet.
    """

    CONTINUE = "CONTINUE"
    ABORT = "ABORT"
