using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.IO;

// Redraws IME tray icons in the style of the Windows ones:
//   tile:  1px 50% black frame, white inside, black glyph
//   mode:  white glyph with a 1px 50% black outline (25% on the diagonals), no background
// Up to 48px the strokes are pixel-exact (hinted, 1 bit) with only the steps of diagonal strokes
// softened, like the hand-tuned originals; above that plain antialiasing.
public static class IconDraw {
    public static string FontName = "Microsoft JhengHei";

    // the glyph as coverage 0..1 on an S x S canvas, as large as fits in box x box, centred;
    // grow: pixels added to the right of and below every stroke (a heavier weight)
    public static float[,] Mask(string ch, int S, int box, FontStyle style, int grow) {
        bool exact = S <= 48;
        int C = S * 3, fit = box - grow;
        float lo = 2, hi = S * 3, em = 0;
        Rectangle bb = Rectangle.Empty;
        for (int it = 0; it < 18; it++) {
            float mid = (lo + hi) / 2;
            Rectangle r; Render(ch, C, mid, style, S, exact ? TextRenderingHint.SingleBitPerPixelGridFit : TextRenderingHint.AntiAlias, out r);
            if (r.Width == 0) lo = mid;
            else if (r.Width <= fit && r.Height <= fit) { lo = mid; em = mid; bb = r; }
            else hi = mid;
        }
        var res = new float[S, S];
        if (em == 0) return res;
        Rectangle unused;
        float[,] m = Render(ch, C, em, style, S, exact ? TextRenderingHint.SingleBitPerPixelGridFit : TextRenderingHint.AntiAlias, out unused);
        if (exact) {
            float[,] aa = Render(ch, C, em, style, S, TextRenderingHint.AntiAliasGridFit, out unused);
            var soft = new float[C, C];
            for (int y = 1; y < C - 1; y++)
                for (int x = 1; x < C - 1; x++) {
                    if (m[x, y] > 0) { soft[x, y] = 1; continue; }
                    bool l = m[x - 1, y] > 0, r = m[x + 1, y] > 0, u = m[x, y - 1] > 0, d = m[x, y + 1] > 0;
                    // the inside of a step of a diagonal stroke: one side and one of top / bottom, nothing opposite
                    bool step = (l ^ r) && (u ^ d);
                    if (step) soft[x, y] = Math.Min(0.6f, aa[x, y]);
                }
            m = soft;
        }
        int ox = (S - (bb.Width + grow)) / 2 - bb.X, oy = (S - (bb.Height + grow)) / 2 - bb.Y;
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++) {
                float v = 0;
                for (int dy = 0; dy <= grow; dy++)
                    for (int dx = 0; dx <= grow; dx++) {
                        int sx = x - ox - dx, sy = y - oy - dy;
                        if (sx < 0 || sy < 0 || sx >= C || sy >= C) continue;
                        if (m[sx, sy] > v) v = m[sx, sy];
                    }
                res[x, y] = v;
            }
        return res;
    }

    static float[,] Render(string ch, int C, float em, FontStyle style, int S, TextRenderingHint hint, out Rectangle bb) {
        var m = new float[C, C];
        using (var b = new Bitmap(C, C, PixelFormat.Format32bppArgb))
        using (var g = Graphics.FromImage(b))
        using (var f = new Font(FontName, em, style, GraphicsUnit.Pixel)) {
            g.Clear(Color.Black);
            g.TextRenderingHint = hint;
            g.DrawString(ch, f, Brushes.White, S / 2, S / 2, StringFormat.GenericTypographic);
            int x0 = C, y0 = C, x1 = -1, y1 = -1;
            BitmapData d = b.LockBits(new Rectangle(0, 0, C, C), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var px = new byte[d.Stride * C];
            System.Runtime.InteropServices.Marshal.Copy(d.Scan0, px, 0, px.Length);
            b.UnlockBits(d);
            for (int y = 0; y < C; y++)
                for (int x = 0; x < C; x++) {
                    float v = px[y * d.Stride + x * 4 + 1] / 255f;
                    m[x, y] = v;
                    if (v > 0.25f) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
                }
            bb = x1 < 0 ? Rectangle.Empty : new Rectangle(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        }
        return m;
    }

    // the weight per size, after the originals' strokes: 1px at 16, 2px at 24, 3px at 48
    static void Weight(int S, out FontStyle style, out int grow) {
        style = S == 24 ? FontStyle.Bold : FontStyle.Regular;
        grow = (S == 40 || S == 48) ? 1 : 0;
    }

    // white glyph (black for a light taskbar) with the outline
    public static Bitmap Mode(string ch, int S, bool darkGlyph) {
        int o = Math.Max(1, (int)Math.Round(S / 48.0));           // outline width
        FontStyle style; int grow; Weight(S, out style, out grow);
        float[,] m = Mask(ch, S, S - 2 * o, style, grow);
        var b = new Bitmap(S, S, PixelFormat.Format32bppArgb);
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++) {
                float edge = 0, corner = 0;
                for (int dy = -o; dy <= o; dy++)
                    for (int dx = -o; dx <= o; dx++) {
                        int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= S || yy >= S) continue;
                        float v = m[xx, yy];
                        if (dx == 0 || dy == 0) { if (v > edge) edge = v; } else if (v > corner) corner = v;
                    }
                float outline = Math.Max(0.5f * edge, 0.25f * corner);
                float g = m[x, y];
                float a = g + (1 - g) * outline;
                int c = a <= 0 ? 0 : (int)Math.Round(255 * g / a);
                if (darkGlyph) c = 255 - c;
                b.SetPixel(x, y, Color.FromArgb((int)Math.Round(a * 255), c, c, c));
            }
        return b;
    }

    // the framed white tile with a black glyph
    public static Bitmap Tile(string ch, int S) {
        // as heavy as the Korean IME's tile (3px strokes at 48) where a many-stroke character still
        // reads: from 40px up; below that heavier strokes run together
        return Tile(ch, S, S >= 64 ? FontStyle.Bold : FontStyle.Regular, (S == 40 || S == 48) ? 1 : 0);
    }

    public static Bitmap Tile(string ch, int S, FontStyle style, int grow) {
        int f = Math.Max(1, (int)Math.Round(S / 48.0));           // frame width
        int pad = Math.Max(1, (int)Math.Round(S / 16.0));
        float[,] m = Mask(ch, S, S - 2 * (f + pad), style, grow);
        var b = new Bitmap(S, S, PixelFormat.Format32bppArgb);
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++) {
                bool frame = x < f || y < f || x >= S - f || y >= S - f;
                if (frame) { b.SetPixel(x, y, Color.FromArgb(128, 0, 0, 0)); continue; }
                int c = (int)Math.Round(255 * (1 - m[x, y]));
                b.SetPixel(x, y, Color.FromArgb(255, c, c, c));
            }
        return b;
    }

    // .ico: 32bpp BMP entries below 256px, PNG for 256px (what rc.exe and old loaders expect)
    public static void SaveIco(string path, List<Bitmap> images) {
        var blobs = new List<byte[]>();
        foreach (Bitmap b in images) blobs.Add(b.Width >= 256 ? Png(b) : Dib(b));
        using (var w = new BinaryWriter(File.Create(path))) {
            w.Write((ushort)0); w.Write((ushort)1); w.Write((ushort)images.Count);
            int offset = 6 + 16 * images.Count;
            for (int i = 0; i < images.Count; i++) {
                int s = images[i].Width;
                w.Write((byte)(s >= 256 ? 0 : s)); w.Write((byte)(s >= 256 ? 0 : s));
                w.Write((byte)0); w.Write((byte)0); w.Write((ushort)1); w.Write((ushort)32);
                w.Write(blobs[i].Length); w.Write(offset);
                offset += blobs[i].Length;
            }
            foreach (byte[] blob in blobs) w.Write(blob);
        }
    }

    static byte[] Png(Bitmap b) { using (var ms = new MemoryStream()) { b.Save(ms, ImageFormat.Png); return ms.ToArray(); } }

    static byte[] Dib(Bitmap b) {
        int s = b.Width, maskStride = ((s + 31) / 32) * 4;
        using (var ms = new MemoryStream())
        using (var w = new BinaryWriter(ms)) {
            w.Write(40); w.Write(s); w.Write(s * 2); w.Write((ushort)1); w.Write((ushort)32);
            w.Write(0); w.Write(s * s * 4 + maskStride * s); w.Write(0); w.Write(0); w.Write(0); w.Write(0);
            for (int y = s - 1; y >= 0; y--)
                for (int x = 0; x < s; x++) { Color c = b.GetPixel(x, y); w.Write(c.B); w.Write(c.G); w.Write(c.R); w.Write(c.A); }
            for (int y = s - 1; y >= 0; y--) {
                var row = new byte[maskStride];
                for (int x = 0; x < s; x++) if (b.GetPixel(x, y).A == 0) row[x / 8] |= (byte)(0x80 >> (x % 8));
                w.Write(row);
            }
            return ms.ToArray();
        }
    }
}
