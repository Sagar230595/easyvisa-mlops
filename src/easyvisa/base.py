"""Abstract base class for all pipeline stages."""
from abc import ABC, abstractmethod

from .config import Config
from .infra import SparkProvider, get_logger


class PipelineStage(ABC):
    """A single step of the ML pipeline.

    Subclasses set ``name`` and implement ``run``. Spark is lazily provided so
    that non-Spark logic can be unit tested without a cluster.
    """

    name: str = "stage"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = get_logger(f"easyvisa.{self.name}")

    @property
    def spark(self):
        return SparkProvider.get()

    @abstractmethod
    def run(self):
        raise NotImplementedError
