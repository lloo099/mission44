import copy
import json
import unittest
from pathlib import Path

from compare_schema import validate_compare

ROOT = Path(__file__).resolve().parents[1]


class CompareSchemaTest(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "data/compare.json").read_text())

    def test_repository_data(self):
        self.assertEqual(validate_compare(self.data), [])
        self.assertTrue((ROOT / self.data["archive"]["url"]).is_file())

    def test_required_provenance(self):
        for key in ("version", "evaluator", "budget", "capturedAt", "evaluationDate", "sourceUrl"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                del data["metrics"][0][key]
                self.assertTrue(validate_compare(data))

    def test_no_cross_protocol_ranking(self):
        for change in ({"evidence": "vendor"}, {"harness": "mixed"}):
            data = copy.deepcopy(self.data)
            data["metrics"][0].update(change)
            self.assertTrue(validate_compare(data))
        self.data["metrics"][0]["items"][0]["version"] = "4.3"
        self.assertTrue(validate_compare(self.data))

    def test_bad_scores(self):
        for value in (True, "65", -1, 101, float("nan"), float("inf"), None):
            with self.subTest(value=value):
                self.data["metrics"][0]["items"][0]["score"] = value
                self.assertTrue(validate_compare(self.data))

    def test_explicit_missing_not_zero(self):
        row = self.data["metrics"][0]["items"][0]
        row.update(score=None, missingReason="Not evaluated")
        self.assertEqual(validate_compare(self.data), [])

    def test_duplicate_configuration(self):
        metric = self.data["metrics"][0]
        metric["items"].append(copy.deepcopy(metric["items"][0]))
        self.assertTrue(validate_compare(self.data))

    def test_mixed_harness_is_visible(self):
        metric = next(m for m in self.data["metrics"] if m["harness"] == "mixed")
        del metric["items"][0]["harness"]
        self.assertTrue(validate_compare(self.data))

    def test_dates_and_sources(self):
        self.data["metrics"][0]["evaluationDate"] = "2027-01-01"
        self.assertTrue(validate_compare(self.data))
        self.data["metrics"][0]["evaluationDate"] = None
        self.data["metrics"][0]["items"][0]["sourceUrl"] = "javascript:alert(1)"
        self.assertTrue(validate_compare(self.data))


if __name__ == "__main__":
    unittest.main()
