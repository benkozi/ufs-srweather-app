import logging
import shutil
import sys
import tempfile
import unittest

logger = logging.getLogger('test_generate_fire_emissions')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter('[%(levelname)s][%(name)s][%(created)f] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG) #tdk:set to info

class TestGenerateFireEmissions(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir)

    def test(self) -> None:
        logger.debug('hello world')