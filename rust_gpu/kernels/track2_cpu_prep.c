/* Track 2 fast C CPU image-prep kernels (L2 tokenizer worker path, zero GIL, single-pass cache-hot).
 *
 * Compiled on-box inside track2-base:v1:
 *   gcc -O3 -shared -fPIC -o gravity_ocr/_native/libtrack2_cpu_prep.so rust_gpu/kernels/track2_cpu_prep.c -lm
 *
 * Functions:
 *   1. t2_glyph_size_u8    : exact equivalent of textscale.glyph_size() on any strided uint8 green plane
 *   2. t2_ink_box_u8       : exact equivalent of margin_crop._ink_box() on any strided uint8 [3, H, W] image
 *   3. t2_copy_crop_chw_u8 : single-pass crop + HWC->CHW (or CHW->CHW) copy straight into /dev/shm ring
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmp_int(const void* a, const void* b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

static int cmp_i16(const void* a, const void* b) {
    int16_t x = *(const int16_t*)a;
    int16_t y = *(const int16_t*)b;
    return (x > y) - (x < y);
}

/* Linear-interpolation percentile matching numpy.percentile(sorted_arr, q) */
static double pct_sorted_int(const int* arr, int n, double q) {
    if (n <= 0) return 0.0;
    if (n == 1) return (double)arr[0];
    double pos = (q / 100.0) * (double)(n - 1);
    int lo = (int)pos;
    int hi = lo + 1;
    if (hi >= n) return (double)arr[n - 1];
    double frac = pos - (double)lo;
    return (double)arr[lo] * (1.0 - frac) + (double)arr[hi] * frac;
}

static double pct_sorted_i16(const int16_t* arr, int n, double q) {
    if (n <= 0) return 0.0;
    if (n == 1) return (double)arr[0];
    double pos = (q / 100.0) * (double)(n - 1);
    int lo = (int)pos;
    int hi = lo + 1;
    if (hi >= n) return (double)arr[n - 1];
    double frac = pos - (double)lo;
    return (double)arr[lo] * (1.0 - frac) + (double)arr[hi] * frac;
}

static double robust_pct(int* ht, int n, double q) {
    qsort(ht, (size_t)n, sizeof(int), cmp_int);
    double med = pct_sorted_int(ht, n, 50.0);
    double lo_v = 0.35 * med;
    double hi_v = 3.0 * med;
    int a = 0;
    while (a < n && (double)ht[a] < lo_v) a++;
    int b = n;
    while (b > a && (double)ht[b - 1] > hi_v) b--;
    if (b <= a) return med;
    return pct_sorted_int(ht + a, b - a, q);
}

/* Exact match for textscale.glyph_size() on a uint8 green plane with strides (sh, sw).
 * Returns 1 on success (*out_size, *out_a, *out_b written), 0 if rejected (unreliable). */
int t2_glyph_size_u8(
    const uint8_t* ptr,
    int H,
    int W,
    int64_t sh,
    int64_t sw,
    double* out_size,
    double* out_a,
    double* out_b
) {
    int f = 1;
    int h = H, w = W;
    uint8_t* owned = NULL;
    const uint8_t* g = ptr;
    int64_t cur_sh = sh, cur_sw = sw;

    while ((int64_t)h * (int64_t)w > 2000000 && f < 4) {
        int h2 = h / 2, w2 = w / 2;
        uint8_t* nxt = (uint8_t*)malloc((size_t)h2 * (size_t)w2);
        if (!nxt) { free(owned); return 0; }
        for (int y = 0; y < h2; y++) {
            const uint8_t* r0 = g + (int64_t)(2 * y) * cur_sh;
            const uint8_t* r1 = r0 + cur_sh;
            uint8_t* dst = nxt + (size_t)y * (size_t)w2;
            if (cur_sw == 1) {
                for (int x = 0; x < w2; x++) {
                    int x2 = 2 * x;
                    dst[x] = (uint8_t)(((uint32_t)r0[x2] + r0[x2 + 1] + r1[x2] + r1[x2 + 1] + 2) >> 2);
                }
            } else {
                for (int x = 0; x < w2; x++) {
                    int64_t x2 = (int64_t)(2 * x) * cur_sw;
                    dst[x] = (uint8_t)(((uint32_t)r0[x2] + r0[x2 + cur_sw] + r1[x2] + r1[x2 + cur_sw] + 2) >> 2);
                }
            }
        }
        free(owned);
        owned = nxt;
        g = owned;
        h = h2; w = w2;
        cur_sh = w2; cur_sw = 1;
        f *= 2;
    }

    int n8 = (w - 2) / 8;
    if (h < 12 || n8 < 8) { free(owned); return 0; }

    int16_t* p8 = (int16_t*)malloc((size_t)h * (size_t)n8 * sizeof(int16_t));
    if (!p8) { free(owned); return 0; }

    int64_t edge_cnt = 0;
    int ew = w - 2;
    int n8_px = n8 * 8;
    for (int y = 0; y < h; y++) {
        const uint8_t* row = g + (int64_t)y * cur_sh;
        int16_t* prow = p8 + (size_t)y * (size_t)n8;
        if (cur_sw == 1) {
            for (int b = 0; b < n8; b++) {
                int s = 0;
                const uint8_t* rp = row + b * 8;
                for (int j = 0; j < 8; j++) {
                    int diff = (int)rp[j + 2] - (int)rp[j];
                    if (diff < 0) diff = -diff;
                    s += (diff > 32);
                }
                prow[b] = (int16_t)s;
                edge_cnt += s;
            }
            for (int x = n8_px; x < ew; x++) {
                int diff = (int)row[x + 2] - (int)row[x];
                if (diff < 0) diff = -diff;
                edge_cnt += (diff > 32);
            }
        } else {
            int64_t sw2 = 2 * cur_sw;
            for (int b = 0; b < n8; b++) {
                int s = 0;
                const uint8_t* rp = row + (int64_t)(b * 8) * cur_sw;
                for (int j = 0; j < 8; j++) {
                    int diff = (int)rp[sw2] - (int)rp[0];
                    if (diff < 0) diff = -diff;
                    s += (diff > 32);
                    rp += cur_sw;
                }
                prow[b] = (int16_t)s;
                edge_cnt += s;
            }
            const uint8_t* rp = row + (int64_t)n8_px * cur_sw;
            for (int x = n8_px; x < ew; x++) {
                int diff = (int)rp[sw2] - (int)rp[0];
                if (diff < 0) diff = -diff;
                edge_cnt += (diff > 32);
                rp += cur_sw;
            }
        }
    }

    double dens = (double)edge_cnt / ((double)h * (double)ew);
    if (dens < 0.004 || dens > 0.30) {
        free(p8); free(owned); return 0;
    }

    int hmax = (int)(0.8 * (double)h);
    if (hmax < 8) hmax = 8;

    const int ks[2] = {3, 8};
    const double dmaxs[2] = {0.80, 0.90};
    double res[2] = {0.0, 0.0};
    const int min_runs[2] = {8, 3};
    const double pcts[2] = {25.0, 50.0};

    int16_t* col_buf = (int16_t*)malloc((size_t)h * sizeof(int16_t));
    int max_hts = ((h / 2) + 2) * (n8 / 3 + 1);
    int* hts = (int*)malloc((size_t)max_hts * sizeof(int));
    if (!col_buf || !hts) {
        free(col_buf); free(hts); free(p8); free(owned); return 0;
    }

    int ok = 1;
    for (int pass = 0; pass < 2; pass++) {
        int k = ks[pass];
        int nk = n8 / k;
        int max_val = k * 8; /* 24 or 64 */
        int64_t on_cnt = 0;
        int n_ht = 0;

        for (int c = 0; c < nk; c++) {
            int c0 = c * k;
            int hist[65];
            memset(hist, 0, (size_t)(max_val + 1) * sizeof(int));
            for (int y = 0; y < h; y++) {
                const int16_t* prow = p8 + (size_t)y * (size_t)n8 + c0;
                int16_t sum = 0;
                for (int j = 0; j < k; j++) sum += prow[j];
                col_buf[y] = sum;
                hist[sum]++;
            }
            /* O(h + 65) exact linear-interpolated 10th percentile matching numpy.percentile */
            double pos = 0.10 * (double)(h - 1);
            int lo_i = (int)pos;
            int hi_i = lo_i + 1 < h ? lo_i + 1 : h - 1;
            double frac = pos - (double)lo_i;
            int acc = 0, v_lo = 0, v_hi = 0;
            for (int v = 0; v <= max_val; v++) {
                int prev = acc;
                acc += hist[v];
                if (prev <= lo_i && lo_i < acc) v_lo = v;
                if (prev <= hi_i && hi_i < acc) { v_hi = v; break; }
            }
            double thr = ((double)v_lo * (1.0 - frac) + (double)v_hi * frac) + 1.0;

            int run = 0;
            for (int y = 0; y < h; y++) {
                if ((double)col_buf[y] > thr) {
                    on_cnt++;
                    run++;
                } else if (run > 0) {
                    if (run >= 4 && run <= hmax) hts[n_ht++] = run;
                    run = 0;
                }
            }
            if (run >= 4 && run <= hmax) hts[n_ht++] = run;
        }

        double duty = (double)on_cnt / ((double)h * (double)nk);
        if (duty > dmaxs[pass] || n_ht < min_runs[pass]) {
            ok = 0;
            break;
        }
        res[pass] = robust_pct(hts, n_ht, pcts[pass]);
    }

    free(col_buf); free(hts); free(p8); free(owned);
    if (!ok) return 0;

    double a = res[0];
    double b = res[1];
    double s = a < (b / 1.31) ? a : (b / 1.31);
    *out_size = (double)f * s;
    if (out_a) *out_a = (double)f * a;
    if (out_b) *out_b = (double)f * b;
    return 1;
}

/* Check one horizontal row y in [x_lo, x_hi) for faint/strong ink.
 * Sets *has_strong = 1 if any pixel exceeds strong bounds; returns 1 if ink (strong or >= m faint). */
static inline int check_row_u8(
    const uint8_t* p0, const uint8_t* p1, const uint8_t* p2,
    int len, int64_t sw,
    const uint8_t lo_f[3], const uint8_t hi_f[3],
    const uint8_t lo_s[3], const uint8_t hi_s[3],
    int m, int* has_strong
) {
    int faint_cnt = 0;
    int st = 0;
    if (sw == 1) {
        for (int x = 0; x < len; x++) {
            uint8_t v0 = p0[x], v1 = p1[x], v2 = p2[x];
            if (v0 < lo_f[0] || v0 > hi_f[0] || v1 < lo_f[1] || v1 > hi_f[1] || v2 < lo_f[2] || v2 > hi_f[2]) {
                if (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]) {
                    st = 1;
                    *has_strong = 1;
                    return 1;
                }
                faint_cnt++;
                if (faint_cnt >= m && m == 1) {
                    /* still need to know if row has strong only if caller checks strong, so continue only when needed */
                }
            }
        }
    } else {
        for (int x = 0; x < len; x++) {
            int64_t off = (int64_t)x * sw;
            uint8_t v0 = p0[off], v1 = p1[off], v2 = p2[off];
            if (v0 < lo_f[0] || v0 > hi_f[0] || v1 < lo_f[1] || v1 > hi_f[1] || v2 < lo_f[2] || v2 > hi_f[2]) {
                if (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]) {
                    st = 1;
                    *has_strong = 1;
                    return 1;
                }
                faint_cnt++;
            }
        }
    }
    *has_strong = st;
    return (faint_cnt >= m);
}

static inline int row_is_strong_u8(
    const uint8_t* p0, const uint8_t* p1, const uint8_t* p2,
    int len, int64_t sw,
    const uint8_t lo_s[3], const uint8_t hi_s[3]
) {
    if (sw == 1) {
        for (int x = 0; x < len; x++) {
            uint8_t v0 = p0[x], v1 = p1[x], v2 = p2[x];
            if (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2])
                return 1;
        }
    } else {
        for (int x = 0; x < len; x++) {
            int64_t off = (int64_t)x * sw;
            uint8_t v0 = p0[off], v1 = p1[off], v2 = p2[off];
            if (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2])
                return 1;
        }
    }
    return 0;
}

/* Exact equivalent of margin_crop._ink_box() on a 3-channel uint8 view [3, H, W] with strides (sc, sh, sw)
 * inside sub-rectangle [y_start, y_end) x [x_start, x_end).
 * Writes (fy0, sy0, fy1, sy1, fx0, sx0, fx1, sx1) relative to (y_start, x_start).
 * Returns 1 if ink found, 0 if blank. */
int t2_ink_box_u8(
    const uint8_t* ptr,
    int y_start, int y_end, int x_start, int x_end,
    int64_t sc, int64_t sh, int64_t sw,
    const uint8_t lo_f[3], const uint8_t hi_f[3],
    const uint8_t lo_s[3], const uint8_t hi_s[3],
    int m, int reach,
    int out_coords[8]
) {
    int H = y_end - y_start;
    int W = x_end - x_start;
    if (H <= 0 || W <= 0) return 0;

    const uint8_t* base0 = ptr + (int64_t)y_start * sh + (int64_t)x_start * sw;
    const uint8_t* base1 = base0 + sc;
    const uint8_t* base2 = base1 + sc;

    /* 1. Top row scan */
    int fy0 = -1, sy0 = -1;
    for (int y = 0; y < H; y++) {
        int st = 0;
        if (check_row_u8(base0 + y * sh, base1 + y * sh, base2 + y * sh, W, sw, lo_f, hi_f, lo_s, hi_s, m, &st)) {
            fy0 = y;
            if (st) { sy0 = y; }
            else if (reach > 1) {
                int lim = y + reach < H ? y + reach : H;
                for (int yy = y + 1; yy < lim; yy++) {
                    if (row_is_strong_u8(base0 + yy * sh, base1 + yy * sh, base2 + yy * sh, W, sw, lo_s, hi_s)) {
                        sy0 = yy; break;
                    }
                }
            }
            break;
        }
    }
    if (fy0 < 0) return 0;

    /* 2. Bottom row scan */
    int fy1 = fy0, sy1 = -1;
    for (int y = H - 1; y >= fy0; y--) {
        int st = 0;
        if (check_row_u8(base0 + y * sh, base1 + y * sh, base2 + y * sh, W, sw, lo_f, hi_f, lo_s, hi_s, m, &st)) {
            fy1 = y;
            if (st) { sy1 = y; }
            else if (reach > 1) {
                int lim = y - reach + 1 > 0 ? y - reach + 1 : 0;
                for (int yy = y - 1; yy >= lim; yy--) {
                    if (row_is_strong_u8(base0 + yy * sh, base1 + yy * sh, base2 + yy * sh, W, sw, lo_s, hi_s)) {
                        sy1 = yy; break;
                    }
                }
            }
            break;
        }
    }

    /* 3. Outside-in blocked column scans over rows [fy0, fy1] (stops as soon as margins end — skips 90%+ interior!) */
    int mm = m > 255 ? 255 : m;
    int fx0 = -1, sx0 = -1, fx1 = -1, sx1 = -1;
    uint8_t col_f[512], col_s[512];

    /* Left-to-right scan in growing blocks (32, 64, 128, 256, 512...) */
    int pos = 0, step = 32;
    while (pos < W && fx0 < 0) {
        int nb = step;
        if (nb > 512) nb = 512;
        if (pos + nb > W) nb = W - pos;
        memset(col_f, 0, (size_t)nb);
        memset(col_s, 0, (size_t)nb);
        for (int y = fy0; y <= fy1; y++) {
            const uint8_t* p0 = base0 + (int64_t)y * sh + (int64_t)pos * sw;
            const uint8_t* p1 = base1 + (int64_t)y * sh + (int64_t)pos * sw;
            const uint8_t* p2 = base2 + (int64_t)y * sh + (int64_t)pos * sw;
            if (sw == 1) {
                for (int j = 0; j < nb; j++) {
                    uint8_t v0 = p0[j], v1 = p1[j], v2 = p2[j];
                    if (v0 < lo_f[0] || v0 > hi_f[0] || v1 < lo_f[1] || v1 > hi_f[1] || v2 < lo_f[2] || v2 > hi_f[2]) {
                        if (col_f[j] < mm) col_f[j]++;
                        if (!col_s[j] && (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]))
                            col_s[j] = 1;
                    }
                }
            } else {
                for (int j = 0; j < nb; j++) {
                    int64_t off = (int64_t)j * sw;
                    uint8_t v0 = p0[off], v1 = p1[off], v2 = p2[off];
                    if (v0 < lo_f[0] || v0 > hi_f[0] || v1 < lo_f[1] || v1 > hi_f[1] || v2 < lo_f[2] || v2 > hi_f[2]) {
                        if (col_f[j] < mm) col_f[j]++;
                        if (!col_s[j] && (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]))
                            col_s[j] = 1;
                    }
                }
            }
        }
        for (int j = 0; j < nb; j++) {
            if (col_s[j] || col_f[j] >= mm) {
                fx0 = pos + j;
                if (reach > 0) {
                    int lim = fx0 + reach < W ? fx0 + reach : W;
                    int lim_in = lim < pos + nb ? lim : pos + nb;
                    for (int xx = fx0; xx < lim_in; xx++) {
                        if (col_s[xx - pos]) { sx0 = xx; break; }
                    }
                    for (int xx = lim_in; sx0 < 0 && xx < lim; xx++) {
                        for (int y = fy0; y <= fy1; y++) {
                            int64_t off = (int64_t)y * sh + (int64_t)xx * sw;
                            uint8_t v0 = base0[off], v1 = base1[off], v2 = base2[off];
                            if (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]) {
                                sx0 = xx; break;
                            }
                        }
                    }
                }
                break;
            }
        }
        pos += nb;
        if (step < 512) step *= 2;
    }
    if (fx0 < 0) return 0;

    /* Right-to-left scan in growing blocks */
    int rpos = W;
    step = 32;
    while (rpos > fx0 && fx1 < 0) {
        int nb = step;
        if (nb > 512) nb = 512;
        if (rpos - nb < fx0) nb = rpos - fx0;
        int x0b = rpos - nb;
        memset(col_f, 0, (size_t)nb);
        memset(col_s, 0, (size_t)nb);
        for (int y = fy0; y <= fy1; y++) {
            const uint8_t* p0 = base0 + (int64_t)y * sh + (int64_t)x0b * sw;
            const uint8_t* p1 = base1 + (int64_t)y * sh + (int64_t)x0b * sw;
            const uint8_t* p2 = base2 + (int64_t)y * sh + (int64_t)x0b * sw;
            if (sw == 1) {
                for (int j = 0; j < nb; j++) {
                    uint8_t v0 = p0[j], v1 = p1[j], v2 = p2[j];
                    if (v0 < lo_f[0] || v0 > hi_f[0] || v1 < lo_f[1] || v1 > hi_f[1] || v2 < lo_f[2] || v2 > hi_f[2]) {
                        if (col_f[j] < mm) col_f[j]++;
                        if (!col_s[j] && (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]))
                            col_s[j] = 1;
                    }
                }
            } else {
                for (int j = 0; j < nb; j++) {
                    int64_t off = (int64_t)j * sw;
                    uint8_t v0 = p0[off], v1 = p1[off], v2 = p2[off];
                    if (v0 < lo_f[0] || v0 > hi_f[0] || v1 < lo_f[1] || v1 > hi_f[1] || v2 < lo_f[2] || v2 > hi_f[2]) {
                        if (col_f[j] < mm) col_f[j]++;
                        if (!col_s[j] && (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]))
                            col_s[j] = 1;
                    }
                }
            }
        }
        for (int j = nb - 1; j >= 0; j--) {
            if (col_s[j] || col_f[j] >= mm) {
                fx1 = x0b + j;
                if (reach > 0) {
                    int lim = fx1 - reach + 1 > 0 ? fx1 - reach + 1 : 0;
                    int lim_in = lim > x0b ? lim : x0b;
                    for (int xx = fx1; xx >= lim_in; xx--) {
                        if (col_s[xx - x0b]) { sx1 = xx; break; }
                    }
                    for (int xx = lim_in - 1; sx1 < 0 && xx >= lim; xx--) {
                        for (int y = fy0; y <= fy1; y++) {
                            int64_t off = (int64_t)y * sh + (int64_t)xx * sw;
                            uint8_t v0 = base0[off], v1 = base1[off], v2 = base2[off];
                            if (v0 < lo_s[0] || v0 > hi_s[0] || v1 < lo_s[1] || v1 > hi_s[1] || v2 < lo_s[2] || v2 > hi_s[2]) {
                                sx1 = xx; break;
                            }
                        }
                    }
                }
                break;
            }
        }
        rpos = x0b;
        if (step < 512) step *= 2;
    }

    out_coords[0] = fy0; out_coords[1] = sy0;
    out_coords[2] = fy1; out_coords[3] = sy1;
    out_coords[4] = fx0; out_coords[5] = sx0;
    out_coords[6] = fx1; out_coords[7] = sx1;
    return 1;
}

/* Single-pass crop + transpose/copy from any strided [3, H, W] uint8 tensor straight into contiguous CHW dst.
 * Fast-paths both HWC (sc=1, sw=3) and planar CHW (sw=1). */
void t2_copy_crop_chw_u8(
    const uint8_t* src,
    int64_t sc, int64_t sh, int64_t sw,
    int y0, int y1, int x0, int x1,
    uint8_t* dst
) {
    int ch = y1 - y0;
    int cw = x1 - x0;
    size_t plane = (size_t)ch * (size_t)cw;
    uint8_t* d0 = dst;
    uint8_t* d1 = dst + plane;
    uint8_t* d2 = dst + 2 * plane;

    if (sw == 1) {
        const uint8_t* s0 = src + (int64_t)y0 * sh + x0;
        const uint8_t* s1 = s0 + sc;
        const uint8_t* s2 = s1 + sc;
        for (int y = 0; y < ch; y++) {
            memcpy(d0 + (size_t)y * cw, s0 + (int64_t)y * sh, (size_t)cw);
            memcpy(d1 + (size_t)y * cw, s1 + (int64_t)y * sh, (size_t)cw);
            memcpy(d2 + (size_t)y * cw, s2 + (int64_t)y * sh, (size_t)cw);
        }
    } else if (sc == 1 && sw == 3) {
        /* torchvision read_image() raw HWC layout: deinterleave RGB -> RRR/GGG/BBB in one cache-hot sweep */
        for (int y = 0; y < ch; y++) {
            const uint8_t* row = src + (int64_t)(y0 + y) * sh + (int64_t)x0 * 3;
            uint8_t* r0 = d0 + (size_t)y * cw;
            uint8_t* r1 = d1 + (size_t)y * cw;
            uint8_t* r2 = d2 + (size_t)y * cw;
            for (int x = 0; x < cw; x++) {
                r0[x] = row[3 * x];
                r1[x] = row[3 * x + 1];
                r2[x] = row[3 * x + 2];
            }
        }
    } else {
        for (int y = 0; y < ch; y++) {
            const uint8_t* s0 = src + (int64_t)(y0 + y) * sh + (int64_t)x0 * sw;
            const uint8_t* s1 = s0 + sc;
            const uint8_t* s2 = s1 + sc;
            uint8_t* r0 = d0 + (size_t)y * cw;
            uint8_t* r1 = d1 + (size_t)y * cw;
            uint8_t* r2 = d2 + (size_t)y * cw;
            for (int x = 0; x < cw; x++) {
                int64_t off = (int64_t)x * sw;
                r0[x] = s0[off];
                r1[x] = s1[off];
                r2[x] = s2[off];
            }
        }
    }
}
