#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>

typedef struct {
    double hpwl;
    double min_x;
    double max_x;
    double min_y;
    double max_y;
} hpwl_stats_t;

static int extract_coords(PyObject *coords_seq, double **xs_out, double **ys_out, Py_ssize_t *size_out) {
    PyObject *fast = PySequence_Fast(coords_seq, "coordinates must be a sequence");
    if (fast == NULL) {
        return -1;
    }

    Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
    *size_out = size;
    if (size < 2) {
        Py_DECREF(fast);
        *xs_out = NULL;
        *ys_out = NULL;
        return 0;
    }

    double *xs = PyMem_Malloc(sizeof(double) * size);
    double *ys = PyMem_Malloc(sizeof(double) * size);
    if (xs == NULL || ys == NULL) {
        Py_DECREF(fast);
        PyMem_Free(xs);
        PyMem_Free(ys);
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        PyObject *point = PySequence_Fast(item, "each coordinate must be a sequence");
        if (point == NULL) {
            Py_DECREF(fast);
            PyMem_Free(xs);
            PyMem_Free(ys);
            return -1;
        }
        if (PySequence_Fast_GET_SIZE(point) < 2) {
            Py_DECREF(point);
            Py_DECREF(fast);
            PyMem_Free(xs);
            PyMem_Free(ys);
            PyErr_SetString(PyExc_ValueError, "each coordinate must have at least two elements");
            return -1;
        }
        PyObject *x_obj = PySequence_Fast_GET_ITEM(point, 0);
        PyObject *y_obj = PySequence_Fast_GET_ITEM(point, 1);
        double x = PyFloat_AsDouble(x_obj);
        double y = PyFloat_AsDouble(y_obj);
        Py_DECREF(point);
        if (PyErr_Occurred()) {
            Py_DECREF(fast);
            PyMem_Free(xs);
            PyMem_Free(ys);
            return -1;
        }
        xs[i] = x;
        ys[i] = y;
    }

    Py_DECREF(fast);
    *xs_out = xs;
    *ys_out = ys;
    return 1;
}

static void compute_stats(double *xs, double *ys, Py_ssize_t size, hpwl_stats_t *out_stats) {
    double min_x = xs[0];
    double max_x = xs[0];
    double min_y = ys[0];
    double max_y = ys[0];

    for (Py_ssize_t i = 1; i < size; ++i) {
        double x = xs[i];
        double y = ys[i];
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }

    out_stats->min_x = min_x;
    out_stats->max_x = max_x;
    out_stats->min_y = min_y;
    out_stats->max_y = max_y;
    out_stats->hpwl = (max_x - min_x) + (max_y - min_y);
}

static PyObject *build_stats_tuple(const hpwl_stats_t *stats) {
    return Py_BuildValue("(ddddd)",
                         stats->hpwl,
                         stats->min_x,
                         stats->max_x,
                         stats->min_y,
                         stats->max_y);
}

static PyObject *hpwl_stats(PyObject *self, PyObject *args) {
    PyObject *coords_seq;
    if (!PyArg_ParseTuple(args, "O:hpwl_stats", &coords_seq)) {
        return NULL;
    }

    double *xs = NULL;
    double *ys = NULL;
    Py_ssize_t size = 0;
    int extract_status = extract_coords(coords_seq, &xs, &ys, &size);
    if (extract_status < 0) {
        return NULL;
    }

    if (size < 2) {
        return Py_BuildValue("(ddddd)", 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    hpwl_stats_t stats;

    Py_BEGIN_ALLOW_THREADS
    compute_stats(xs, ys, size, &stats);
    Py_END_ALLOW_THREADS

    PyMem_Free(xs);
    PyMem_Free(ys);

    return build_stats_tuple(&stats);
}

static PyObject *hpwl_stats_batch(PyObject *self, PyObject *args) {
    PyObject *batch_seq;
    if (!PyArg_ParseTuple(args, "O:hpwl_stats_batch", &batch_seq)) {
        return NULL;
    }

    PyObject *fast_batch = PySequence_Fast(batch_seq, "batch must be a sequence");
    if (fast_batch == NULL) {
        return NULL;
    }

    Py_ssize_t batch_size = PySequence_Fast_GET_SIZE(fast_batch);
    PyObject *result_list = PyList_New(batch_size);
    if (result_list == NULL) {
        Py_DECREF(fast_batch);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < batch_size; ++i) {
        PyObject *coords_seq = PySequence_Fast_GET_ITEM(fast_batch, i);
        double *xs = NULL;
        double *ys = NULL;
        Py_ssize_t size = 0;
        int status = extract_coords(coords_seq, &xs, &ys, &size);
        if (status < 0) {
            Py_DECREF(fast_batch);
            Py_DECREF(result_list);
            return NULL;
        }
        if (size < 2) {
            PyList_SET_ITEM(result_list, i, Py_BuildValue("(ddddd)", 0.0, 0.0, 0.0, 0.0, 0.0));
            continue;
        }

        hpwl_stats_t stats;
        Py_BEGIN_ALLOW_THREADS
        compute_stats(xs, ys, size, &stats);
        Py_END_ALLOW_THREADS

        PyMem_Free(xs);
        PyMem_Free(ys);

        PyObject *tuple = build_stats_tuple(&stats);
        if (tuple == NULL) {
            Py_DECREF(fast_batch);
            Py_DECREF(result_list);
            return NULL;
        }
        PyList_SET_ITEM(result_list, i, tuple);
    }

    Py_DECREF(fast_batch);
    return result_list;
}

static PyMethodDef HPWLMethods[] = {
    {"hpwl_stats", hpwl_stats, METH_VARARGS, "Compute HPWL and bbox for a net."},
    {"hpwl_stats_batch", hpwl_stats_batch, METH_VARARGS, "Batch HPWL stats."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hpwlmodule = {
    PyModuleDef_HEAD_INIT,
    "_hpwl",
    "Fast HPWL helpers",
    -1,
    HPWLMethods
};

PyMODINIT_FUNC PyInit__hpwl(void) {
    return PyModule_Create(&hpwlmodule);
}
