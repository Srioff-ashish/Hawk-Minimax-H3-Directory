"""Story from the author's own novel: a PDF (or text) is split at its chapter headings, read for the cast and a
summary, and each part is drafted into a chapter from its own text, dialogue kept.

    cd tests_api && python3 -m unittest test_story_source
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawk_api import story_source  # noqa: E402

try:
    import pypdf  # noqa: F401
    from hawk_api.jobs import RequestError
    from hawk_api.story import StoryManager
    from test_story import DRAFT, WriterChat
    from test_manhwa import PipelineService
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"story source test dependencies missing: {exc}")


def make_pdf(pages: list[list[str]]) -> bytes:
    """A real text PDF: one Helvetica line per entry, a page per list."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        body = "BT /F1 11 Tf 14 TL 60 780 Td " + " ".join(
            "(" + line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
        content = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {content} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(len(objects))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for n, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


NOVEL = [
    ["The Last Train", "", "by A. Writer"],
    ["Chapter 1: Platform Nine", "", "Mina waited on platform nine as the rain came down over",
     "the old station roof. The clock said eleven.", "", "\"You are late again,\" she said when Jun ran up the stairs.",
     "", "\"The bus broke down,\" Jun said, out of breath.", "", "3"],
    ["Chapter 2: The Night Market", "", "They walked into the night market, past the noodle stalls.",
     "", "\"I have something to tell you,\" Jun said.", "", "Mina stopped walking and looked at him.", "", "4"],
]


class Splitting(unittest.TestCase):
    def test_a_pdf_is_read_and_split_at_its_chapter_headings(self):
        text, pages = story_source.extract_text(make_pdf(NOVEL * 1), "the_last_train.pdf")
        self.assertEqual(pages, 3)
        self.assertNotRegex(text, r"(?m)^\s*[34]\s*$", "bare page numbers are dropped")
        self.assertIn("rain came down over the old station roof", text, "a line broken by the PDF is joined")
        parts = story_source.split_parts(text)
        self.assertEqual([p["title"] for p in parts], ["Chapter 1: Platform Nine", "Chapter 2: The Night Market"],
                         "the title page is not a chapter")
        self.assertIn('"You are late again," she said', parts[0]["text"])
        self.assertIn("I have something to tell you", parts[1]["text"])

    def test_headings_in_other_forms_and_text_without_any(self):
        for heading in ("CHAPTER THREE", "Chapter Twenty-One: Home", "Ch. 12 - The End", "Part II", "Adhyay 4"):
            with self.subTest(heading=heading):
                self.assertTrue(story_source.HEADING.match(heading))
        for line in ("Chapter and verse were quoted all night long by the priest.", "Partly cloudy", "The chapter ended."):
            with self.subTest(line=line):
                self.assertFalse(story_source.HEADING.match(line))
        plain = "\n\n".join(f"Paragraph {n} " + "word " * 400 for n in range(40))
        parts = story_source.split_parts(plain)
        self.assertGreater(len(parts), 3)
        self.assertEqual(parts[0]["title"], "Part 1")
        self.assertTrue(all(len(p["text"]) <= story_source.PLAIN_PART_CHARS + 2500 for p in parts))

    def test_a_very_long_chapter_is_split_and_parts_group_into_chapters(self):
        long = "Chapter 1\n\n" + "\n\n".join("A long paragraph. " * 60 for _ in range(60)) + "\n\nChapter 2\n\nShort."
        parts = story_source.split_parts(long)
        self.assertGreater(len(parts), 2)
        self.assertTrue(parts[0]["title"].startswith("Chapter 1 (1 of "))
        self.assertEqual(parts[-1]["title"], "Chapter 2")
        self.assertEqual(story_source.group_parts([0, 1, 2], [5, 5, 5], 5), [[0], [1], [2]])
        groups = story_source.group_parts(list(range(10)), [10] * 10, 3)
        self.assertEqual(len(groups), 3)
        self.assertEqual(sum(groups, []), list(range(10)), "every part once, in order")

    def test_what_cannot_be_read_is_refused_plainly(self):
        with self.assertRaisesRegex(story_source.SourceError, "scanned"):
            story_source.extract_text(make_pdf([[""], [""]]), "scan.pdf")
        with self.assertRaisesRegex(story_source.SourceError, "not a PDF"):
            story_source.extract_text(b"PK\x03\x04 a docx", "story.docx")
        text, pages = story_source.extract_text("Chapter 1\n\nHello.\n\nChapter 2\n\nBye.".encode(), "story.txt")
        self.assertEqual((pages, len(story_source.split_parts(text))), (0, 2))


class Reader(WriterChat):
    """The co-writer, which also reads the novel: every part names Mina and Jun, the first describes Mina."""

    async def chat(self, model, messages, *, purpose="", json_mode=True, **kw):
        if purpose == "story reading":
            self.calls.append((model, purpose))
            self.sent.append(messages)
            return json.dumps({"summary": "Mina waits for Jun at a station in the rain; he has a secret.",
                               "characters": [{"name": "Mina", "description": "late 20s, short black hair, red coat"},
                                              {"name": "Jun", "description": "early 30s, tall"}]}), {}
        return await super().chat(model, messages, purpose=purpose, json_mode=json_mode, **kw)


class FromTheNovel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hawk_story_source_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.service = PipelineService(self.tmp)
        self.chat = Reader()
        self.service.atlas = self.chat
        self.manager = StoryManager(self.service)
        self.manager.poll = 0.01
        self.story = self.manager.create_story("The Last Train", "", "")

    async def asyncTearDown(self):
        await self.manager.stop()

    async def read(self, sid):
        for _ in range(500):
            source = self.manager.view(sid)["source"]
            if source["status"] != "reading":
                return source
            await asyncio.sleep(0.01)
        self.fail("still reading")

    async def test_import_reads_the_cast_and_each_part_drafts_a_chapter_from_its_own_text(self):
        sid = self.story["id"]
        view = await self.manager.import_source(sid, "the_last_train.pdf", make_pdf(NOVEL))
        self.assertEqual(view["source"]["status"], "reading")
        source = await self.read(sid)
        self.assertEqual((source["status"], source["pages"], len(source["parts"])), ("ready", 3, 2))
        view = self.manager.view(sid)
        self.assertIn("Mina waits for Jun", view["premise"], "an empty premise is filled from the summary")
        cast = {c["name"]: c for c in view["characters"]}
        self.assertEqual(cast["Mina"]["description"], "late 20s, short black hair, red coat")
        self.assertEqual(cast["Mina"]["role"], "cast")

        chapter = await self.manager.draft_chapter(sid)  # no chat: the first part of the novel
        self.assertEqual(chapter["source_parts"], [0])
        prompt = self.chat.sent[-1][0]["content"]
        self.assertIn("author's own novel: adapt it faithfully", prompt)
        self.assertIn('"You are late again," she said', prompt)
        self.assertNotIn("Night Market", prompt, "only this chapter's part")

        await self.manager.chat(sid, "Make the market scene end on a close-up of Mina.")
        second = await self.manager.draft_chapter(sid)
        self.assertEqual(second["source_parts"], [1])
        prompt = self.chat.sent[-1][0]["content"]
        self.assertIn("I have something to tell you", prompt)
        self.assertIn("THE USER'S NOTES FOR THIS CHAPTER", prompt, "the chat steers the adaptation")
        self.assertIn("close-up of Mina", prompt)
        parts = self.manager.view(sid)["source"]["parts"]
        self.assertEqual([p["chapter"] for p in parts], [0, 1])
        with self.assertRaisesRegex(RequestError, "already a chapter"):
            self.manager.start_draft(sid)

    async def test_the_users_own_cast_wins_and_a_new_upload_replaces_the_text(self):
        sid = self.story["id"]
        self.manager.add_character(sid, "Mina", "my own words for Mina", "cast")
        await self.manager.import_source(sid, "a.pdf", make_pdf(NOVEL))
        await self.read(sid)
        cast = {c["name"]: c for c in self.manager.view(sid)["characters"]}
        self.assertEqual(cast["Mina"]["description"], "my own words for Mina")
        await self.manager.draft_chapter(sid)
        await self.manager.import_source(sid, "b.txt", b"Chapter 1\n\nA new start.\n\nChapter 2\n\nAnd more.")
        view = self.manager.view(sid)
        self.assertEqual(view["source"]["filename"], "b.txt")
        self.assertEqual([p["chapter"] for p in view["source"]["parts"]], [None, None], "the old links are dropped")
        self.assertEqual(len(view["chapters"]), 1, "chapters already drafted stay")
        with self.assertRaisesRegex(RequestError, "scanned"):
            await self.manager.import_source(sid, "scan.pdf", make_pdf([[""]]))
        self.manager.drop_source(sid)
        self.assertIsNone(self.manager.view(sid)["source"])

    async def test_autopilot_makes_the_novel_into_chapters_with_no_premise(self):
        sid = self.story["id"]
        await self.manager.import_source(sid, "the_last_train.pdf", make_pdf(NOVEL))
        self.manager.start_autopilot(sid, chapters=5, portraits=False, videos=False)
        for _ in range(1500):
            pilot = self.manager.view(sid)["autopilot"]
            if pilot["status"] != "running":
                break
            await asyncio.sleep(0.02)
        self.assertEqual(pilot["status"], "done", pilot)
        self.assertEqual([o["parts"] for o in pilot["outline"]], [[0], [1]], "one chapter per part, not five")
        self.assertEqual([c["source_parts"] for c in self.manager.view(sid)["chapters"]], [[0], [1]])
        self.assertFalse(any(p == "story outline" for _, p in self.chat.calls), "the novel is the outline")


if __name__ == "__main__":
    unittest.main()
