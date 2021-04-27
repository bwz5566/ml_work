# distutils: language=c++
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange
from libcpp.map cimport map

def hello():
    print("hello")


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result

cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result

cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()

    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    i=0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)



######################################################################
# homework
cpdef target_mean_v4(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int)
    target_mean_v4_impl(result, y, x, nrow)
    return result

@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void target_mean_v4_impl(double[:] result, double[:] y, int[:] x, const long nrow):
    cdef map[int, double] value_dict
    cdef map[int, double] count_dict
    cdef long i
    for i in range(nrow):
        if  value_dict.find(x[i]) == value_dict.end():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1.0
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1.0
    i=0
    for i in prange(nrow,nogil=True,schedule='static', chunksize=1):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)
