import unittest
from vrsbench_finetune.preprocess import is_binary_answer, parse_numeric_answer

class TestPreprocess(unittest.TestCase):
    def test_binary(self):
        self.assertEqual(is_binary_answer("Yes"), (True, 1))
        self.assertEqual(is_binary_answer("no."), (True, 0))
        self.assertEqual(is_binary_answer("maybe"), (False, None))

    def test_numeric(self):
        self.assertEqual(parse_numeric_answer("123.45"), (True, 123.45))
        self.assertEqual(parse_numeric_answer("1.23e-5"), (True, 1.23e-5))
        self.assertEqual(parse_numeric_answer("five"), (True, 5.0))
        self.assertEqual(parse_numeric_answer("cat"), (False, None))

if __name__ == '__main__':
    unittest.main()
