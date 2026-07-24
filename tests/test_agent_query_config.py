import unittest

from pydantic import ValidationError

from web.models import AgentQuery


class AgentQueryConfigTest(unittest.TestCase):
    def test_query_accepts_optional_top_k(self) -> None:
        self.assertIsNone(AgentQuery(question="测试").top_k)
        self.assertEqual(AgentQuery(question="测试", top_k=8).top_k, 8)

    def test_query_rejects_out_of_range_top_k(self) -> None:
        with self.assertRaises(ValidationError):
            AgentQuery(question="测试", top_k=0)
        with self.assertRaises(ValidationError):
            AgentQuery(question="测试", top_k=21)


if __name__ == "__main__":
    unittest.main()
