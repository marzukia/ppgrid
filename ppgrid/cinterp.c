/* Bit-identical np.interp replacement for ppgrid (W2 priority 3).
 *
 * Clones numpy's interp (numpy/_core/src/multiarray/compiled_base.c,
 * numpy 2.5.x, BSD-3-Clause) element arithmetic so the produced floats are
 * bit-identical to np.interp(x, xp, fp) with default left/right for ANY xp
 * (monotonic, tied, or non-monotonic), any fp, and any x (f64, or f32 with
 * the exact f32 -> f64 promotion numpy's internal cast performs).
 *
 * Why this is faster than np.interp:
 * numpy's per-element binary_search_with_guess only hits when consecutive
 * elements land in the same/adjacent node bin. On real interpolated blocks
 * the percentile-rank index j moves by many nodes per element (measured:
 * 98%+ of elements fall through to the ~10-step bisection even on smooth
 * 2-D fields), costing ~50-60 ns/element. Here the input is processed in
 * 256-element chunks: two bisections per chunk bound the whole chunk's j
 * range [jlo, jhi]; every element then searches only that window (0 steps
 * when the chunk sits in a single node bin). The window bound is exact, so
 * each element's final j equals numpy's and the (cloned) arithmetic is
 * unchanged: output bits are identical for every input.
 *
 * j convention note: numpy returns j = len for key > xp[-1]; a lower_bound
 * search gives len - 1. With default left/right the output is fp[-1] in
 * both cases, so the convention difference is invisible.
 *
 * Design notes:
 * - Buffer protocol only (no numpy C API): the JIT build is one cc line
 *   and only needs the CPython headers.
 * - The GIL is released around the whole call so block worker threads
 *   interpolate in parallel (numpy's arr_interp releases it too).
 * - Build with -O3 -ffp-contract=off and NO -march: x86-64 baseline has
 *   no FMA, so slope*(x-dx)+dy compiles to mulsd+addsd, the same two IEEE
 *   ops numpy performs. No reassociation (plain -O3, no -ffast-math).
 * - Slopes are precomputed by the caller (ppgrid._cinterp) with numpy
 *   ufuncs; the per-element formula (dy[j+1]-dy[j])/(dx[j+1]-dx[j]) is the
 *   same IEEE ops numpy uses, so precompute vs per-element is bit-identical
 *   (numpy itself precomputes when lenxp <= lenx).
 *
 * Public functions (METH_FASTCALL, 5 positional args):
 *   interp_f64(x, xp, fp, slopes, out)   x: C-contiguous f64, any shape
 *   interp_f32(x, xp, fp, slopes, out)   x: C-contiguous f32, any shape
 *   xp/fp: C-contiguous f64, 1-D, len(xp) == len(fp)
 *   slopes: C-contiguous f64, len == len(xp) - 1
 *   out: C-contiguous WRITABLE f64, same number of elements as x
 * Returns None on success; the result is in out.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

typedef Py_ssize_t intp;

/* Chunk size for windowing: 256 elements (4 KB f64, L1-friendly). */
#define INTERP_CHUNK 256

/* numpy's binary_search_with_guess cache-window constant (cloned). */
#define LIKELY_IN_CACHE_SIZE 8

/* lower_bound: first i in [lo, hi] with arr[i] > key, or hi if none.
 * Invariant: lo == 0 or arr[lo - 1] <= key. O(log(hi - lo)). */
static intp
lb_gt(const double *arr, intp lo, intp hi, const double key)
{
    while (lo < hi) {
        const intp mid = lo + ((hi - lo) >> 1);
        if (arr[mid] > key) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }
    return lo;
}

/* numpy's final j for one key: last i with arr[i] <= key, with
 * -1 when key < arr[0] and len-1 when key > arr[-1] (see header note). */
static intp
j_of(const double *arr, const intp len, const double key)
{
    return lb_gt(arr, 0, len, key) - 1;
}

/* Element arithmetic, verbatim numpy arr_interp branch for a known j
 * (j in [-1, lenxp - 1]; j == lenxp folds into j == lenxp - 1 with the
 * same default-left/right output). */
static double
interp_at(const double x_val, const intp j, const double *dx, const double *dy,
          const double *slopes, const double lval, const double rval, const intp lenxp)
{
    if (j == -1) {
        return lval;
    }
    if (j == lenxp - 1) {
        return dy[j];
    }
    if (dx[j] == x_val) {
        /* Avoid potential non-finite interpolation */
        return dy[j];
    }
    const double slope = slopes[j];
    /* If we get nan in one direction, try the other */
    double r = slope * (x_val - dx[j]) + dy[j];
    if (r != r) {
        r = slope * (x_val - dx[j + 1]) + dy[j + 1];
        if (r != r && dy[j] == dy[j + 1]) {
            r = dy[j];
        }
    }
    return r;
}


/* Verbatim numpy clone (original search with persistent guess). Bit-identical
 * to np.interp for ANY xp (including non-monotonic, where the search result
 * itself is data/guess dependent); used when xp is not non-decreasing. */
static intp
bs_guess(double key, const double *arr, const intp len, intp guess)
{
    intp imin = 0, imax = len;

    if (key > arr[len - 1]) {
        return len;
    }
    else if (key < arr[0]) {
        return -1;
    }

    if (len <= 4) {
        intp i;
        for (i = 1; i < len && key >= arr[i]; i++) {
        }
        return i - 1;
    }

    /*
     * Clamp the guess so that guess-1, guess and guess+2 are valid array
     * indexes. This avoids a boundary check in the loop below.
     */
    if (guess > len - 3) {
        guess = len - 3;
    }
    if (guess < 1) {
        guess = 1;
    }

    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            /*
             * Set the search range to [0, guess-1], but
             * restrict the range to LIKELY_IN_CACHE_SIZE elements
             * if we don't expect the key to be outside of
             * the cache
             */
            imax = guess - 1;
            if (guess > LIKELY_IN_CACHE_SIZE &&
                    key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
                imin = guess - LIKELY_IN_CACHE_SIZE;
            }
        }
        else {
            /* guess-1 <= key < guess */
            return guess - 1;
        }
    }
    else {
        /*
         * Set the search range to [guess, len], but
         * restrict the range to LIKELY_IN_CACHE_SIZE elements
         * if we don't expect the key to be outside of
         * the cache
         */
        if (key < arr[guess + 1]) {
            /* guess <= key < guess+1 */
            return guess;
        }
        else {
            imin = guess + 2;
            if (guess < len - LIKELY_IN_CACHE_SIZE - 1 &&
                    key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
                imax = guess + LIKELY_IN_CACHE_SIZE;
            }
            if (key < arr[guess + 2]) {
                /* guess+1 <= key < guess+2 */
                return guess + 1;
            }
        }
    }

    while (imin < imax) {
        const intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

#define INTERP_LOOP_VERBATIM(XLOAD)                                          \
    do {                                                                     \
        intp j = 0;                                                          \
        for (intp i = 0; i < lenx; ++i) {                                   \
            const double x_val = (XLOAD);                                   \
            if (x_val != x_val) {                                           \
                dres[i] = x_val;                                            \
                continue;                                                   \
            }                                                               \
            const intp jj = bs_guess(x_val, dx, lenxp, j);                  \
            j = jj;                                                          \
            if (jj == -1) {                                                 \
                dres[i] = lval;                                             \
            }                                                               \
            else if (jj == lenxp) {                                         \
                dres[i] = rval;                                             \
            }                                                               \
            else if (jj == lenxp - 1) {                                     \
                dres[i] = dy[jj];                                           \
            }                                                               \
            else if (dx[jj] == x_val) {                                     \
                /* Avoid potential non-finite interpolation */               \
                dres[i] = dy[jj];                                           \
            }                                                               \
            else {                                                           \
                const double slope = slopes[jj];                            \
                /* If we get nan in one direction, try the other */          \
                double r = slope * (x_val - dx[jj]) + dy[jj];               \
                if (r != r) {                                               \
                    r = slope * (x_val - dx[jj + 1]) + dy[jj + 1];          \
                    if (r != r && dy[jj] == dy[jj + 1]) {                   \
                        r = dy[jj];                                         \
                    }                                                       \
                }                                                           \
                dres[i] = r;                                                \
            }                                                               \
        }                                                                   \
    } while (0)


static int
is_nondecreasing(const double *dx, const intp lenxp)
{
    for (intp i = 1; i < lenxp; ++i) {
        /* NaN-aware: !(a >= b) is true when either operand is NaN, so a
         * NaN-containing xp routes to the verbatim numpy search clone. */
        if (!(dx[i] >= dx[i - 1])) {
            return 0;
        }
    }
    return 1;
}

/* Chunked windowed kernel. XLOAD(i) loads element i as a double
 * (f64: dz[i]; f32: (double) dz32[i]). */
#define INTERP_CHUNKED_KERNEL(XLOAD)                                        \
    do {                                                                    \
        for (intp c0 = 0; c0 < lenx; c0 += INTERP_CHUNK) {                  \
            const intp c1 = c0 + INTERP_CHUNK < lenx ? c0 + INTERP_CHUNK : lenx; \
            double xmin = INFINITY, xmax = -INFINITY;                          \
            int has_finite = 0;                                             \
            for (intp i = c0; i < c1; ++i) {                                \
                const double v = (XLOAD);                                \
                if (v == v) {                                               \
                    has_finite = 1;                                         \
                    if (v < xmin) {                                         \
                        xmin = v;                                           \
                    }                                                       \
                    if (v > xmax) {                                         \
                        xmax = v;                                           \
                    }                                                       \
                }                                                           \
            }                                                               \
            if (!has_finite) {                                              \
                for (intp i = c0; i < c1; ++i) {                            \
                    dres[i] = (XLOAD);                                   \
                }                                                           \
                continue;                                                   \
            }                                                               \
            const intp jlo = j_of(dx, lenxp, xmin);                         \
            const intp jhi = j_of(dx, lenxp, xmax);                         \
            const intp a0 = jlo + 1;                                        \
            const intp b0 = jhi + 1;                                        \
            if (a0 == b0) {                                                 \
                /* Whole chunk in one node bin: no per-element search. */   \
                const intp jj = a0 - 1;                                     \
                for (intp i = c0; i < c1; ++i) {                            \
                    const double x_val = (XLOAD);                        \
                    dres[i] = (x_val != x_val) ? x_val :                    \
                               interp_at(x_val, jj, dx, dy, slopes, lval, rval, lenxp); \
                }                                                           \
            }                                                               \
            else {                                                          \
                intp jprev = jlo;                                           \
                for (intp i = c0; i < c1; ++i) {                            \
                    const double x_val = (XLOAD);                        \
                    if (x_val != x_val) {                                   \
                        dres[i] = x_val;                                    \
                        continue;                                           \
                    }                                                       \
                    intp jj;                                                \
                    {                                                       \
                        intp g = jprev;                                     \
                        if (g < a0) {                                       \
                            g = a0;                                         \
                        }                                                   \
                        if (g > b0 - 1) {                                   \
                            g = b0 - 1;                                     \
                        }                                                   \
                        if (x_val < dx[g]) {                                \
                            if (g == a0 || x_val >= dx[g - 1]) {            \
                                jj = g - 1;                                 \
                            }                                               \
                            else {                                          \
                                jj = lb_gt(dx, a0, g, x_val) - 1;           \
                            }                                               \
                        }                                                   \
                        else if (g == b0 - 1) {                             \
                            jj = g;                                         \
                        }                                                   \
                        else if (x_val < dx[g + 1]) {                       \
                            jj = g;                                         \
                        }                                                   \
                        else if (g + 2 < b0 && x_val < dx[g + 2]) {         \
                            jj = g + 1;                                     \
                        }                                                   \
                        else {                                              \
                            jj = lb_gt(dx, g + 2 < b0 ? g + 2 : g + 1, b0, x_val) - 1; \
                        }                                                   \
                    }                                                       \
                    jprev = jj;                                             \
                    dres[i] = interp_at(x_val, jj, dx, dy, slopes, lval, rval, lenxp); \
                }                                                           \
            }                                                               \
        }                                                                   \
    } while (0)

/* numpy's arr_interp lenxp == 1 branch (note: no NaN check, so NaN in ->
 * fp_val out; cloned deliberately). */
#define INTERP_LOOP_NX1(XLOAD)                                              \
    do {                                                                    \
        const double xp_val = dx[0];                                        \
        const double fp_val = dy[0];                                        \
        for (intp i = 0; i < lenx; ++i) {                                  \
            const double x_val = (XLOAD);                                  \
            dres[i] = (x_val < xp_val) ? lval :                            \
                                         ((x_val > xp_val) ? rval : fp_val);\
        }                                                                   \
    } while (0)

static int
get_buf(PyObject *obj, Py_buffer *vb, const char *fmt, int writable, const char *name)
{
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) {
        flags |= PyBUF_WRITABLE;
    }
    if (PyObject_GetBuffer(obj, vb, flags) != 0) {
        PyErr_Format(PyExc_TypeError, "%s: expected a C-contiguous float array", name);
        return -1;
    }
    if (vb->format == NULL || strcmp(vb->format, fmt) != 0) {
        PyBuffer_Release(vb);
        PyErr_Format(PyExc_TypeError, "%s: expected dtype %s, got format %s", name, fmt,
                     vb->format ? vb->format : "(none)");
        return -1;
    }
    return 0;
}

#define D64 8
#define D32 4

static PyObject *
interp_impl(PyObject *self, PyObject *const *args, Py_ssize_t nargs, int xbits)
{
    PyObject *ox = args[0], *oxp = args[1], *ofp = args[2], *oslopes = args[3], *oout = args[4];
    Py_buffer bx, bxp, bfp, bslopes, bout;
    intp lenx, lenxp;
    const char *xfmt = xbits == 8 ? "d" : "f";

    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "usage: interp(x, xp, fp, slopes, out)");
        return NULL;
    }

    if (get_buf(ox, &bx, xfmt, 0, "x") < 0) {
        return NULL;
    }
    if (get_buf(oxp, &bxp, "d", 0, "xp") < 0) {
        PyBuffer_Release(&bx);
        return NULL;
    }
    if (get_buf(ofp, &bfp, "d", 0, "fp") < 0) {
        PyBuffer_Release(&bxp);
        PyBuffer_Release(&bx);
        return NULL;
    }
    if (get_buf(oslopes, &bslopes, "d", 0, "slopes") < 0) {
        PyBuffer_Release(&bfp);
        PyBuffer_Release(&bxp);
        PyBuffer_Release(&bx);
        return NULL;
    }
    if (get_buf(oout, &bout, "d", 1, "out") < 0) {
        PyBuffer_Release(&bslopes);
        PyBuffer_Release(&bfp);
        PyBuffer_Release(&bxp);
        PyBuffer_Release(&bx);
        return NULL;
    }

    lenx = (intp)(bx.len / xbits);
    lenxp = (intp)(bxp.len / D64);

    if (lenxp == 0 && lenx != 0) {
        PyErr_SetString(PyExc_ValueError, "array of sample points is empty");
        goto fail;
    }
    if (bfp.len != bxp.len) {
        PyErr_SetString(PyExc_ValueError, "fp and xp are not of the same length.");
        goto fail;
    }
    if (bslopes.len != (Py_ssize_t)(lenxp - 1) * D64) {
        PyErr_SetString(PyExc_ValueError, "slopes must have len(xp) - 1 elements");
        goto fail;
    }
    if (bout.len != (Py_ssize_t)lenx * D64) {
        PyErr_SetString(PyExc_ValueError, "out must have the same size as x");
        goto fail;
    }
    if (lenx == 0) {
        goto ok;
    }

    {
        const double *dx = (const double *)bxp.buf;
        const double *dy = (const double *)bfp.buf;
        const double *slopes = (const double *)bslopes.buf;
        double *dres = (double *)bout.buf;
        const double *dz64 = (const double *)bx.buf;
        const float *dz32 = (const float *)bx.buf;
        const double lval = dy[0];
        const double rval = dy[lenxp - 1];

        Py_BEGIN_ALLOW_THREADS
        if (lenxp == 1) {
            if (xbits == 8) {
                INTERP_LOOP_NX1(dz64[i]);
            }
            else {
                INTERP_LOOP_NX1((double) dz32[i]);
            }
        }
        else if (!is_nondecreasing(dx, lenxp)) {
            /* Windowing needs monotonic xp; use the exact clone instead. */
            if (xbits == 8) {
                INTERP_LOOP_VERBATIM(dz64[i]);
            }
            else {
                INTERP_LOOP_VERBATIM((double) dz32[i]);
            }
        }
        else if (xbits == 8) {
            INTERP_CHUNKED_KERNEL(dz64[i]);
        }
        else {
            INTERP_CHUNKED_KERNEL((double) dz32[i]);
        }
        Py_END_ALLOW_THREADS
    }

ok:
    PyBuffer_Release(&bout);
    PyBuffer_Release(&bslopes);
    PyBuffer_Release(&bfp);
    PyBuffer_Release(&bxp);
    PyBuffer_Release(&bx);
    Py_RETURN_NONE;

fail:
    PyBuffer_Release(&bout);
    PyBuffer_Release(&bslopes);
    PyBuffer_Release(&bfp);
    PyBuffer_Release(&bxp);
    PyBuffer_Release(&bx);
    return NULL;
}

static PyObject *
interp_f64(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return interp_impl(self, args, nargs, 8);
}

static PyObject *
interp_f32(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return interp_impl(self, args, nargs, 4);
}

static PyMethodDef cinterp_methods[] = {
    {"interp_f64", (PyCFunction)(void (*)(void))interp_f64, METH_FASTCALL,
     "interp_f64(x, xp, fp, slopes, out) -> None\n"
     "Fill out with np.interp(x, xp, fp) bits (x f64, any shape)."},
    {"interp_f32", (PyCFunction)(void (*)(void))interp_f32, METH_FASTCALL,
     "interp_f32(x, xp, fp, slopes, out) -> None\n"
     "As interp_f64 but x is f32 (exact promotion, no intermediate cast)."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef cinterp_module = {
    PyModuleDef_HEAD_INIT,
    "_cinterp_c",
    "Bit-identical np.interp replacement for ppgrid (buffer protocol, chunked window).",
    -1,
    cinterp_methods,
};

PyMODINIT_FUNC
PyInit__cinterp_c(void)
{
    return PyModule_Create(&cinterp_module);
}
