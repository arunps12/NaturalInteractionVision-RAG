"""Custom exception with traceback context."""

import sys


class PipelineException(Exception):
    """Rich exception that captures file and line info from the traceback."""

    def __init__(self, message: str, error_detail: Exception | None = None):
        self.error_message = self._format(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def _format(message: str, error_detail: Exception | None) -> str:
        _, _, exc_tb = sys.exc_info()
        fname = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
        lineno = exc_tb.tb_lineno if exc_tb else "?"
        detail = f" | Cause: {error_detail}" if error_detail else ""
        return f"{message}{detail} [file={fname}, line={lineno}]"

    def __str__(self) -> str:
        return self.error_message
