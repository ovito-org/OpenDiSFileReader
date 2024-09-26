#### Python File Reader Name ####
# Description of your Python-based file reader for OVITO.

from ovito.io import FileReaderInterface
from ovito.data import DataCollection
from traits.api import *


class FileReaderName(FileReaderInterface):
    @staticmethod
    def detect(filename: str):
        pass

    def scan(self, filename: str, register_frame: Callable[..., None]):
        pass

    def parse(
        self, data: DataCollection, filename: str, parser_data: int, **kwargs: Any
    ):
        pass
