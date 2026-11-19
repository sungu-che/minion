//! Pixel kernels of the drawing pipeline (`drawing/ink.py`, `drawing/crops.py`,
//! `preprocess.py`).  Masks are one byte per pixel (0 / 1), images are
//! row-major RGB bytes.  Every kernel computes exactly what its numpy
//! counterpart computes.

use crate::par::{self, Cptr, Ptr};

/// `gray = (r·77 + g·150 + b·29) >> 8`, `chroma = max − min`, and the
/// histogram of `gray` (for the paper level percentile).
pub unsafe fn gray_chroma(rgb: *const u8, npix: usize, gray: *mut u8, chroma: *mut u8, hist: *mut u64) {
    let (src, g, c) = (Cptr(rgb), Ptr(gray), Ptr(chroma));
    let step = 1 << 16;
    let parts = npix.div_ceil(step);
    let mut hists = vec![[0u64; 256]; parts];
    let hp = Ptr(hists.as_mut_ptr());
    par::run(parts, |p| {
        let a = p * step;
        let b = (a + step).min(npix);
        let h = &mut *hp.get().add(p);
        for i in a..b {
            let px = src.get().add(3 * i);
            let (r, gg, bb) = (*px as u32, *px.add(1) as u32, *px.add(2) as u32);
            let v = ((r * 77 + gg * 150 + bb * 29) >> 8) as u8;
            *g.get().add(i) = v;
            h[v as usize] += 1;
            if !c.get().is_null() {
                *c.get().add(i) = (r.max(gg).max(bb) - r.min(gg).min(bb)) as u8;
            }
        }
    });
    if !hist.is_null() {
        for k in 0..256 {
            *hist.add(k) = hists.iter().map(|h| h[k]).sum();
        }
    }
}

/// `out = src < thr` (or `>=` when `ge` is set).
pub unsafe fn threshold(src: *const u8, npix: usize, thr: u8, ge: bool, out: *mut u8) {
    let (s, o) = (Cptr(src), Ptr(out));
    par::chunks(npix, 1 << 16, |a, b| {
        for i in a..b {
            let v = *s.get().add(i);
            *o.get().add(i) = ((v < thr) != ge) as u8;
        }
    });
}

/// Square dilation of a 0/1 mask with a zero border:
/// `maximum_filter(mask, size=2r+1, mode="constant", cval=0)`.
pub unsafe fn dilate(mask: *const u8, h: usize, w: usize, r: usize, out: *mut u8) {
    if r == 0 {
        std::ptr::copy_nonoverlapping(mask, out, h * w);
        return;
    }
    let tmp = vec![0u8; h * w];
    let (m, t, o) = (Cptr(mask), Ptr(tmp.as_ptr() as *mut u8), Ptr(out));
    // rows: sliding count over [x − r, x + r]
    par::chunks(h, 16, |a, b| {
        for y in a..b {
            let src = m.get().add(y * w);
            let dst = t.get().add(y * w);
            let mut cnt: i32 = 0;
            for x in 0..r.min(w) {
                cnt += *src.add(x) as i32;
            }
            for x in 0..w {
                if x + r < w {
                    cnt += *src.add(x + r) as i32;
                }
                if x > r {
                    cnt -= *src.add(x - r - 1) as i32;
                }
                *dst.add(x) = (cnt > 0) as u8;
            }
        }
    });
    // columns: the same count per column, rows processed in order
    par::chunks(w, 256, |a, b| {
        let n = b - a;
        let mut cnt = vec![0i32; n];
        for y in 0..r.min(h) {
            let src = t.get().add(y * w + a);
            for x in 0..n {
                cnt[x] += *src.add(x) as i32;
            }
        }
        for y in 0..h {
            if y + r < h {
                let add = t.get().add((y + r) * w + a);
                for x in 0..n {
                    cnt[x] += *add.add(x) as i32;
                }
            }
            if y > r {
                let sub = t.get().add((y - r - 1) * w + a);
                for x in 0..n {
                    cnt[x] -= *sub.add(x) as i32;
                }
            }
            let dst = o.get().add(y * w + a);
            for x in 0..n {
                *dst.add(x) = (cnt[x] > 0) as u8;
            }
        }
    });
}

/// Pixels of straight runs at least `min_len` long along `axis` (1 = rows,
/// 0 = columns).
pub unsafe fn long_runs(mask: *const u8, h: usize, w: usize, min_len: usize, axis: u32, out: *mut u8) {
    let (m, o) = (Cptr(mask), Ptr(out));
    if axis == 1 {
        par::chunks(h, 16, |a, b| {
            for y in a..b {
                let src = m.get().add(y * w);
                let dst = o.get().add(y * w);
                let mut x = 0;
                while x < w {
                    if *src.add(x) == 0 {
                        *dst.add(x) = 0;
                        x += 1;
                        continue;
                    }
                    let s = x;
                    while x < w && *src.add(x) != 0 {
                        x += 1;
                    }
                    let keep = (x - s >= min_len) as u8;
                    for k in s..x {
                        *dst.add(k) = keep;
                    }
                }
            }
        });
        return;
    }
    par::chunks(w, 64, |a, b| {
        let n = b - a;
        let mut start = vec![usize::MAX; n];
        let close = |x: usize, s: usize, end: usize| {
            let keep = (end - s >= min_len) as u8;
            for y in s..end {
                *o.get().add(y * w + a + x) = keep;
            }
        };
        for y in 0..h {
            let src = m.get().add(y * w + a);
            for x in 0..n {
                if *src.add(x) != 0 {
                    if start[x] == usize::MAX {
                        start[x] = y;
                    }
                } else {
                    if start[x] != usize::MAX {
                        close(x, start[x], y);
                        start[x] = usize::MAX;
                    }
                    *o.get().add(y * w + a + x) = 0;
                }
            }
        }
        for x in 0..n {
            if start[x] != usize::MAX {
                close(x, start[x], h);
            }
        }
    });
}

/// Line pixels that continue a text stroke from both sides along `axis`
/// within `reach` steps (`crops.bridges`): a pixel is kept when a run of line
/// pixels of at most `reach + 1` connects it to text on the near side, and the
/// same holds on the far side.
pub unsafe fn bridges(line: *const u8, text: *const u8, h: usize, w: usize, axis: u32, reach: usize, out: *mut u8) {
    let (l, t, o) = (Cptr(line), Cptr(text), Ptr(out));
    // (outer count, inner count, index of (outer, inner), step along the axis)
    let (outer, inner) = if axis == 0 { (w, h) } else { (h, w) };
    let idx = move |a: usize, i: usize| if axis == 0 { i * w + a } else { a * w + i };
    par::chunks(outer, 32, |a0, a1| {
        let mut fwd = vec![0u8; inner];
        for a in a0..a1 {
            // forward: distance since the last text pixel through line pixels
            let mut cnt = usize::MAX;
            for i in 0..inner {
                let k = idx(a, i);
                if *l.get().add(k) != 0 {
                    cnt = if i > 0 && *t.get().add(idx(a, i - 1)) != 0 { 0 } else { cnt.saturating_add(1) };
                    fwd[i] = (cnt <= reach) as u8;
                } else {
                    cnt = usize::MAX;
                    fwd[i] = 0;
                }
            }
            let mut cnt = usize::MAX;
            for i in (0..inner).rev() {
                let k = idx(a, i);
                let mut back = 0u8;
                if *l.get().add(k) != 0 {
                    cnt = if i + 1 < inner && *t.get().add(idx(a, i + 1)) != 0 { 0 } else { cnt.saturating_add(1) };
                    back = (cnt <= reach) as u8;
                } else {
                    cnt = usize::MAX;
                }
                *o.get().add(k) = fwd[i] & back;
            }
        }
    });
}

/// `out[i] = flags[labels[i]]` (a mask of the selected components; `flags[0]` is the background).
pub unsafe fn select(labels: *const i32, npix: usize, flags: *const u8, nflags: usize, out: *mut u8) {
    let (l, f, o) = (Cptr(labels), Cptr(flags), Ptr(out));
    par::chunks(npix, 1 << 16, |a, b| {
        for i in a..b {
            let k = *l.get().add(i) as usize;
            *o.get().add(i) = if k < nflags { *f.get().add(k) } else { 0 };
        }
    });
}

/// `rgb[mask] = max(rgb[mask], luma)` per channel, in place.
pub unsafe fn damp(rgb: *mut u8, mask: *const u8, npix: usize, luma: u8) {
    let (p, m) = (Ptr(rgb), Cptr(mask));
    par::chunks(npix, 1 << 16, |a, b| {
        for i in a..b {
            if *m.get().add(i) != 0 {
                for c in 0..3 {
                    let v = p.get().add(3 * i + c);
                    *v = (*v).max(luma);
                }
            }
        }
    });
}

/// 8-connected components numbered like `scipy.ndimage.label` (in raster
/// order of their first pixel).  Writes the labels (0 = background) and, when
/// `boxes` is given, `[y0, x0, y1, x1)` of every label (up to `max_boxes`);
/// when `counts` is given, the pixels of every label (same limit).
/// Returns the number of components.
pub unsafe fn label(mask: *const u8, h: usize, w: usize, labels: *mut i32, boxes: *mut i32, max_boxes: usize, counts: *mut i64) -> i64 {
    // runs per row
    struct Run {
        y: u32,
        x0: u32,
        x1: u32, // exclusive
    }
    let mut row_start = vec![0usize; h + 1];
    let mut per_row: Vec<Vec<Run>> = (0..h).map(|_| Vec::new()).collect();
    {
        let m = Cptr(mask);
        let rows = Ptr(per_row.as_mut_ptr());
        par::chunks(h, 32, |a, b| {
            for y in a..b {
                let src = m.get().add(y * w);
                let v = &mut *rows.get().add(y);
                let mut x = 0;
                while x < w {
                    if *src.add(x) == 0 {
                        x += 1;
                        continue;
                    }
                    let s = x;
                    while x < w && *src.add(x) != 0 {
                        x += 1;
                    }
                    v.push(Run { y: y as u32, x0: s as u32, x1: x as u32 });
                }
            }
        });
    }
    for y in 0..h {
        row_start[y + 1] = row_start[y] + per_row[y].len();
    }
    let runs: Vec<Run> = per_row.into_iter().flatten().collect();
    let n = runs.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut a: u32) -> u32 {
        while p[a as usize] != a {
            let g = p[p[a as usize] as usize];
            p[a as usize] = g;
            a = g;
        }
        a
    }
    for y in 1..h {
        let (mut i, ie) = (row_start[y], row_start[y + 1]);
        let (mut j, je) = (row_start[y - 1], row_start[y]);
        // a run of this row and one of the row above touch (8-connectivity)
        // when a.x0 <= b.x1 and b.x0 <= a.x1 (x1 is exclusive)
        while i < ie && j < je {
            let (a, b) = (&runs[i], &runs[j]);
            if b.x1 < a.x0 {
                j += 1;
                continue;
            }
            if a.x1 < b.x0 {
                i += 1;
                continue;
            }
            let ra = find(&mut parent, i as u32);
            let rb = find(&mut parent, j as u32);
            if ra != rb {
                // the smaller index stays the root: the first run of a component
                let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                parent[hi as usize] = lo;
            }
            if a.x1 < b.x1 {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    // number components in the order of their first run
    let mut id = vec![0i32; n];
    let mut next = 0i32;
    for r in 0..n {
        let root = find(&mut parent, r as u32) as usize;
        if root == r {
            next += 1;
            id[r] = next;
        } else {
            id[r] = id[root];
        }
    }
    let lp = Ptr(labels);
    par::chunks(h, 32, |a, b| {
        for y in a..b {
            let dst = lp.get().add(y * w);
            std::ptr::write_bytes(dst, 0, w);
            for r in row_start[y]..row_start[y + 1] {
                let run = &runs[r];
                for x in run.x0..run.x1 {
                    *dst.add(x as usize) = id[r];
                }
            }
        }
    });
    if !counts.is_null() {
        let k = (next as usize).min(max_boxes);
        std::ptr::write_bytes(counts, 0, k);
        for (r, run) in runs.iter().enumerate() {
            let l = id[r] as usize - 1;
            if l < k {
                *counts.add(l) += (run.x1 - run.x0) as i64;
            }
        }
    }
    if !boxes.is_null() {
        let k = (next as usize).min(max_boxes);
        for i in 0..k {
            let bx = boxes.add(4 * i);
            *bx = i32::MAX;
            *bx.add(1) = i32::MAX;
            *bx.add(2) = 0;
            *bx.add(3) = 0;
        }
        for (r, run) in runs.iter().enumerate() {
            let l = id[r] as usize - 1;
            if l >= k {
                continue;
            }
            let bx = boxes.add(4 * l);
            *bx = (*bx).min(run.y as i32);
            *bx.add(1) = (*bx.add(1)).min(run.x0 as i32);
            *bx.add(2) = (*bx.add(2)).max(run.y as i32 + 1);
            *bx.add(3) = (*bx.add(3)).max(run.x1 as i32);
        }
    }
    next as i64
}

/// `preprocess.to_patches`: RGB `[gh·14, gw·14, 3]` → `[gh·gw, 588]` in
/// merge-group order, `(v · inv255 − mean[c]) / std[c]` in f32.
#[allow(clippy::too_many_arguments)]
pub unsafe fn patchify(rgb: *const u8, gh: usize, gw: usize, inv255: f32, mean: *const f32, std: *const f32, out: *mut f32) {
    const P: usize = 14;
    let w = gw * P;
    let mean = [*mean, *mean.add(1), *mean.add(2)];
    let std = [*std, *std.add(1), *std.add(2)];
    let (src, o) = (Cptr(rgb), Ptr(out));
    // token order: group row, group col, i, j; values: C, y, x
    let groups = (gh / 2) * (gw / 2);
    par::chunks(groups, 8, |a, b| {
        for g in a..b {
            let gr = g / (gw / 2);
            let gc = g % (gw / 2);
            for i in 0..2 {
                for j in 0..2 {
                    let token = g * 4 + i * 2 + j;
                    let dst = o.get().add(token * 3 * P * P);
                    let py = (gr * 2 + i) * P;
                    let px = (gc * 2 + j) * P;
                    for c in 0..3 {
                        for y in 0..P {
                            let row = src.get().add(((py + y) * w + px) * 3 + c);
                            for x in 0..P {
                                let v = *row.add(3 * x) as f32;
                                *dst.add(c * P * P + y * P + x) = (v * inv255 - mean[c]) / std[c];
                            }
                        }
                    }
                }
            }
        }
    });
}
