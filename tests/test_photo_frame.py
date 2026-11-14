"""自訂外觀的照片框 (#35): settings/ui/frame.js gives the frames of tests/photo_frames.json and the photo's place of
tests/photo_places.json (the DLL's cpp/tip/photo_frame.h is checked against the same tables by tools/compare.py,
through sz_trace --frames and --places)."""
import json
import os
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "tests", "photo_frames.json")
PLACES = os.path.join(ROOT, "tests", "photo_places.json")


def run_js(call, rows):
    """frame.js, then call (a JS expression of r) for each row: the results."""
    with open(os.path.join(ROOT, "settings", "ui", "frame.js"), encoding="utf-8") as f:
        source = f.read()
    script = source + "\nconst rows = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" + \
        f"console.log(JSON.stringify(rows.map((r) => {call})));\n"
    return json.loads(subprocess.run(["node", "-e", script], input=json.dumps(rows), capture_output=True, text=True,
                                     encoding="utf-8", check=True).stdout)


class PhotoFrameTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_js_frames(self):
        with open(TABLE, encoding="utf-8") as f:
            table = json.load(f)
        for row, got in zip(table, run_js("lookFrame(r.window, r.keys, r.per_line, r.font)", table)):
            self.assertEqual(got, [row["w"], row["h"]], row)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_js_places(self):
        # (code-reviewer 2026-11-14) a window shorter than its frame (the last page, 3 left) shows the same picture's
        # top: the photo is clamped against the larger of window and frame
        with open(PLACES, encoding="utf-8") as f:
            table = json.load(f)
        call = "lookPlace(r.frame, r.zoom, r.x, r.y, r.window[0], r.window[1], r.photo[0], r.photo[1])"
        for row, got in zip(table, run_js(call, table)):
            for g, want in zip(got, row["place"]):
                self.assertAlmostEqual(g, want, places=5, msg=row)


if __name__ == "__main__":
    unittest.main()
