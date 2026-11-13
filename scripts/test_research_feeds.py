import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch_agentic
import fetch_arxiv


XML = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2611.02455v2</id>
<title>Agentic reinforcement learning</title>
<summary>A language model learns tool use.</summary>
<published>2026-11-02T03:17:51Z</published>
<author><name>Author</name></author>
</entry></feed>'''


class ResearchFeedTests(unittest.TestCase):
    def test_arxiv_identity(self):
        for url in (
            "http://arxiv.org/abs/2611.02455v2",
            "https://arxiv.org/pdf/2611.02455v1.pdf",
            "https://export.arxiv.org/abs/2611.02455?context=cs",
        ):
            self.assertEqual(fetch_arxiv.canonical_url(url), "https://arxiv.org/abs/2611.02455")
        self.assertEqual(fetch_arxiv.canonical_url("https://arxiv.org/abs/cs/9901001v3"),
                         "https://arxiv.org/abs/cs/9901001")
        url = "https://example.org/releases/v2?tag=1"
        self.assertEqual(fetch_arxiv.canonical_url(url), url)

    def test_parsers_keep_day_and_author_claim_status(self):
        for parse in (fetch_agentic.parse, fetch_arxiv.parse):
            item = parse(XML, "agentic-rl")[0]
            self.assertEqual(item["year"], "2026-11-02")
            self.assertEqual(item["url"], "https://arxiv.org/abs/2611.02455")
            self.assertEqual(item["confidence"], "self-reported")
            self.assertNotIn("verified", item)

    def test_agentic_refresh_preserves_manual_and_deduplicates_versions(self):
        manual = {"title": "Manual note", "url": "https://arxiv.org/abs/2611.02455",
                  "summary": "Human review", "verified": "2026-11-13"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "agentic.json"
            out.write_text(json.dumps({"items": [manual], "trends": [{"title": "Keep"}]}))
            with patch.object(fetch_agentic, "OUT", str(out)), \
                 patch.object(fetch_agentic, "fetch", return_value=XML), \
                 patch.object(fetch_agentic, "other_curated_urls", return_value=set()):
                fetch_agentic.main()
            data = json.loads(out.read_text())
            self.assertEqual(data["items"], [manual])
            self.assertEqual(data["trends"], [{"title": "Keep"}])

    def test_agentic_refresh_upgrades_auto_date(self):
        old = fetch_agentic.parse(XML, "agentic-rl")[0]
        old.update(year="2026-11", confidence="confirmed",
                   url="http://arxiv.org/abs/2611.02455v1")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "agentic.json"
            out.write_text(json.dumps({"items": [old]}))
            with patch.object(fetch_agentic, "OUT", str(out)), \
                 patch.object(fetch_agentic, "fetch", return_value=XML), \
                 patch.object(fetch_agentic, "other_curated_urls", return_value=set()):
                fetch_agentic.main()
            items = json.loads(out.read_text())["items"]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["year"], "2026-11-02")
            self.assertEqual(items[0]["confidence"], "self-reported")

    def test_live_feed_excludes_pinned_revision(self):
        pinned = {"title": "Reviewed", "url": "https://arxiv.org/abs/2611.02455"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "feed.json"
            with patch.object(fetch_arxiv, "OUT", str(out)), \
                 patch.object(fetch_arxiv, "load_pinned", return_value=[pinned]), \
                 patch.object(fetch_arxiv, "load_previous", return_value=None), \
                 patch.object(fetch_arxiv, "curated_urls", return_value=set()), \
                 patch.object(fetch_arxiv, "fetch", return_value=XML), \
                 patch("sys.argv", ["fetch_arxiv.py"]):
                self.assertEqual(fetch_arxiv.main(), 0)
            self.assertEqual(json.loads(out.read_text())["items"], [pinned])

    def test_failed_live_refresh_keeps_unique_uncurated_items(self):
        prev = {"updated": "2026-11-12", "items": [
            {"url": "http://arxiv.org/abs/2611.02455v1", "year": "2026-11-02", "tags": ["auto"]},
            {"url": "http://arxiv.org/abs/2611.14467v1", "year": "2026-11-10", "tags": ["auto"]},
            {"url": "https://arxiv.org/abs/2611.14467v2", "year": "2026-11-10", "tags": ["auto"]},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "feed.json"
            with patch.object(fetch_arxiv, "OUT", str(out)), \
                 patch.object(fetch_arxiv, "load_pinned", return_value=[]), \
                 patch.object(fetch_arxiv, "load_previous", return_value=prev), \
                 patch.object(fetch_arxiv, "curated_urls", return_value={"https://arxiv.org/abs/2611.02455"}), \
                 patch.object(fetch_arxiv, "fetch", side_effect=OSError("offline")), \
                 patch("sys.argv", ["fetch_arxiv.py"]):
                self.assertEqual(fetch_arxiv.main(), 1)
            data = json.loads(out.read_text())
            self.assertTrue(data["stale"])
            self.assertEqual(data["updated"], "2026-11-12")
            self.assertEqual(len(data["items"]), 1)
            self.assertEqual(data["items"][0]["url"], "https://arxiv.org/abs/2611.14467")


if __name__ == "__main__":
    unittest.main()
