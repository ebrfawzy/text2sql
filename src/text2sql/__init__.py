"""text2sql-toolkit: Open-source Text-to-SQL with automatic database profiling."""

__version__ = "0.1.0"

from text2sql.core import Text2SQL, Text2SQLResult
from text2sql.pipeline.events import PipelineEvent, TokenDelta

__all__ = ["Text2SQL", "Text2SQLResult", "PipelineEvent", "TokenDelta", "__version__"]
