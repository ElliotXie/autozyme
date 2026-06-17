#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "scanpy_kernels.h"

#include <stdint.h>
#include <stdlib.h>

static int get_buffer(PyObject *obj, Py_buffer *view, int writable,
                      Py_ssize_t itemsize, Py_ssize_t min_items,
                      const char *name) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj, view, flags) != 0) return 0;
    if (view->itemsize != itemsize ||
        view->len < min_items * itemsize ||
        view->buf == NULL) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a C-contiguous buffer with itemsize %zd and at least %zd items",
                     name, itemsize, min_items);
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

static PyObject *py_knn_descent_f32(PyObject *self, PyObject *args) {
    PyObject *x_obj, *idx_obj, *dist_obj;
    int n, d, k, n_trees, n_iters, leaf_size, n_threads;
    unsigned long long seed;
    Py_buffer x = {0}, idx = {0}, dist = {0};

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOiiiiiKii",
                          &x_obj, &idx_obj, &dist_obj,
                          &n, &d, &k, &n_trees, &n_iters,
                          &seed, &leaf_size, &n_threads)) {
        return NULL;
    }
    if (n <= 0 || d <= 0 || k <= 0) {
        PyErr_SetString(PyExc_ValueError, "n, d, and k must be positive");
        return NULL;
    }
    if (!get_buffer(x_obj, &x, 0, 4, (Py_ssize_t)n * d, "x")) return NULL;
    if (!get_buffer(idx_obj, &idx, 1, 4, (Py_ssize_t)n * k, "idx")) {
        PyBuffer_Release(&x);
        return NULL;
    }
    if (!get_buffer(dist_obj, &dist, 1, 4, (Py_ssize_t)n * k, "dist")) {
        PyBuffer_Release(&idx);
        PyBuffer_Release(&x);
        return NULL;
    }

    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = scblas_knn_descent_f32(n, d, k, (const float *)x.buf,
                                n_trees, n_iters, leaf_size,
                                (uint64_t)seed,
                                (int32_t *)idx.buf, (float *)dist.buf,
                                n_threads);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&dist);
    PyBuffer_Release(&idx);
    PyBuffer_Release(&x);
    return PyLong_FromLong((long)rc);
}

static PyObject *py_umap_graph_f32(PyObject *self, PyObject *args) {
    PyObject *idx_obj, *dist_obj, *out_rows_obj, *out_cols_obj, *out_vals_obj;
    int n_samples, n_neighbors;
    Py_buffer idx = {0}, dist = {0}, out_rows = {0}, out_cols = {0}, out_vals = {0};

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOii",
                          &idx_obj, &dist_obj,
                          &out_rows_obj, &out_cols_obj, &out_vals_obj,
                          &n_samples, &n_neighbors)) {
        return NULL;
    }
    if (n_samples <= 0 || n_neighbors <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_samples and n_neighbors must be positive");
        return NULL;
    }

    const Py_ssize_t flat = (Py_ssize_t)n_samples * n_neighbors;
    if (!get_buffer(idx_obj, &idx, 0, 4, flat, "idx")) return NULL;
    if (!get_buffer(dist_obj, &dist, 0, 4, flat, "dist")) goto fail_idx;
    if (!get_buffer(out_rows_obj, &out_rows, 1, 4, flat * 2, "out_rows")) goto fail_dist;
    if (!get_buffer(out_cols_obj, &out_cols, 1, 4, flat * 2, "out_cols")) goto fail_rows;
    if (!get_buffer(out_vals_obj, &out_vals, 1, 4, flat * 2, "out_vals")) goto fail_cols;

    float *sigmas = (float *)malloc((size_t)n_samples * sizeof(float));
    float *rhos = (float *)malloc((size_t)n_samples * sizeof(float));
    int32_t *rows = (int32_t *)malloc((size_t)flat * sizeof(int32_t));
    int32_t *cols = (int32_t *)malloc((size_t)flat * sizeof(int32_t));
    float *vals = (float *)malloc((size_t)flat * sizeof(float));
    float *tmp_dists = (float *)malloc((size_t)flat * sizeof(float));
    if (sigmas == NULL || rhos == NULL || rows == NULL || cols == NULL ||
        vals == NULL || tmp_dists == NULL) {
        free(sigmas); free(rhos); free(rows); free(cols); free(vals); free(tmp_dists);
        PyErr_NoMemory();
        goto fail_vals;
    }

    Py_BEGIN_ALLOW_THREADS
    scblas_umap_smooth_knn_dist_f32(n_samples, n_neighbors,
                                    (const float *)dist.buf,
                                    (float)n_neighbors, 64, 1.0f, 1.0f,
                                    sigmas, rhos);
    scblas_umap_membership_strengths_f32(n_samples, n_neighbors,
                                         (const int32_t *)idx.buf,
                                         (const float *)dist.buf,
                                         sigmas, rhos, 0, 0,
                                         rows, cols, vals, tmp_dists);
    scblas_umap_symmetrize_fuzzy_graph_f32(n_samples, n_neighbors,
                                           rows, cols, vals, 1.0f,
                                           (int32_t *)out_rows.buf,
                                           (int32_t *)out_cols.buf,
                                           (float *)out_vals.buf);
    Py_END_ALLOW_THREADS

    free(sigmas); free(rhos); free(rows); free(cols); free(vals); free(tmp_dists);
    PyBuffer_Release(&out_vals);
    PyBuffer_Release(&out_cols);
    PyBuffer_Release(&out_rows);
    PyBuffer_Release(&dist);
    PyBuffer_Release(&idx);
    Py_RETURN_NONE;

fail_vals:
    PyBuffer_Release(&out_vals);
fail_cols:
    PyBuffer_Release(&out_cols);
fail_rows:
    PyBuffer_Release(&out_rows);
fail_dist:
    PyBuffer_Release(&dist);
fail_idx:
    PyBuffer_Release(&idx);
    return NULL;
}

static PyObject *py_umap_layout_euclidean_f32(PyObject *self, PyObject *args) {
    PyObject *embedding_obj, *head_obj, *tail_obj, *eps_obj;
    int n_samples, dim, n_epochs, n_threads;
    double a, b, gamma, initial_alpha, negative_sample_rate;
    unsigned long long seed;
    Py_buffer embedding = {0}, head = {0}, tail = {0}, eps = {0};

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOiiidddddKi",
                          &embedding_obj, &head_obj, &tail_obj, &eps_obj,
                          &n_samples, &dim, &n_epochs,
                          &a, &b, &gamma, &initial_alpha,
                          &negative_sample_rate, &seed, &n_threads)) {
        return NULL;
    }
    if (n_samples <= 0 || dim <= 0 || n_epochs <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_samples, dim, and n_epochs must be positive");
        return NULL;
    }

    if (!get_buffer(embedding_obj, &embedding, 1, 4,
                    (Py_ssize_t)n_samples * dim, "embedding")) return NULL;
    if (!get_buffer(head_obj, &head, 0, 4, 0, "head")) goto fail_embedding;
    if (!get_buffer(tail_obj, &tail, 0, 4, 0, "tail")) goto fail_head;
    if (!get_buffer(eps_obj, &eps, 0, 4, 0, "epochs_per_sample")) goto fail_tail;

    Py_ssize_t n_edges = head.len / 4;
    if (tail.len / 4 != n_edges || eps.len / 4 != n_edges) {
        PyErr_SetString(PyExc_ValueError, "head, tail, and epochs_per_sample lengths must match");
        goto fail_eps;
    }

    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = scblas_umap_optimize_layout_euclidean_f32_parallel(
        (float *)embedding.buf, n_samples, dim,
        (const int32_t *)head.buf, (const int32_t *)tail.buf,
        (int64_t)n_edges, (const float *)eps.buf,
        n_epochs, (float)a, (float)b, (float)gamma,
        (float)initial_alpha, (float)negative_sample_rate,
        (uint64_t)seed, n_threads);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&eps);
    PyBuffer_Release(&tail);
    PyBuffer_Release(&head);
    PyBuffer_Release(&embedding);
    return PyLong_FromLong((long)rc);

fail_eps:
    PyBuffer_Release(&eps);
fail_tail:
    PyBuffer_Release(&tail);
fail_head:
    PyBuffer_Release(&head);
fail_embedding:
    PyBuffer_Release(&embedding);
    return NULL;
}

static PyMethodDef methods[] = {
    {"knn_descent_f32", py_knn_descent_f32, METH_VARARGS,
     "Fill kNN indices and squared distances with the vendored Scanpy kNN kernel."},
    {"umap_graph_f32", py_umap_graph_f32, METH_VARARGS,
     "Build symmetrized UMAP fuzzy graph COO buffers from dense kNN arrays."},
    {"umap_layout_euclidean_f32", py_umap_layout_euclidean_f32, METH_VARARGS,
     "Optimize a UMAP layout in-place with the vendored pthread kernel."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native_scanpy",
    "Minimal vendored native kernels for autozyme.scanpy.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__native_scanpy(void) {
    return PyModule_Create(&module);
}
