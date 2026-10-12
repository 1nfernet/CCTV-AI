import unittest

from src.utils.time_utils import format_timestamp


class TestFormatTimestampInvalidInput(unittest.TestCase):
    def test_rejects_empty_and_unicode_lookalike_timestamps(self):
        """Invalid text must not be interpreted as an ISO-8601 timestamp."""
        for timestamp in ("", "２０２６-０３-１５T１９:１２:００"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    format_timestamp(timestamp)


if __name__ == "__main__":
    unittest.main()
