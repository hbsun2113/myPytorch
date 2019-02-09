"""Module for executors."""
# pylint: disable=invalid-name
from __future__ import absolute_import

from abc import abstractmethod
import functools
import operator

from ... import backend as F
from ...frame import FrameRef, Frame
from ... import utils

from .program import get_current_prog
from . import var
from .var import VarType
from .registry import IR_REGISTRY

__all__ = [
    'OpCode', 'Executor',
    'NodeUDFExecutor', 'NODE_UDF',
    'EdgeUDFExecutor', 'EDGE_UDF',
    'SPMVExecutor', 'SPMV',
    'SPMVWithDataExecutor', 'SPMV_WITH_DATA',
    'ReadExecutor', 'READ',
    'ReadColExecutor', 'READ_COL',
    'ReadRowExecutor', 'READ_ROW',
    'MergeRowExecutor', 'MERGE_ROW',
    'UpdateDictExecutor', 'UPDATE_DICT',
    'NewDictExecutor', 'NEW_DICT',
    'Write_Executor', 'WRITE_',
    'WriteCol_Executor', 'WRITE_COL_',
    'WriteRow_Executor', 'WRITE_ROW_',
    'WriteDict_Executor', 'WRITE_DICT_',
    'AppendRow_Executor', 'APPEND_ROW_',
    'WriteRowInplace_Executor', 'WRITE_ROW_INPLACE_',
    'ClearFrame_Executor', 'CLEAR_FRAME_',
]

class OpCode(object):
    """Opcode for all the executor types."""
    # immutable op
    NODE_UDF = 0
    EDGE_UDF = 1
    SPMV = 2
    SPMV_WITH_DATA = 3
    READ = 4
    READ_COL = 5
    READ_ROW = 6
    MERGE_ROW = 7
    UPDATE_DICT = 8
    NEW_DICT = 9
    # mutable op (no return)
    # remember the name is suffixed with "_"
    WRITE_ = 21
    WRITE_COL_ = 22
    WRITE_ROW_ = 23
    WRITE_DICT_ = 24
    APPEND_ROW_ = 25
    WRITE_ROW_INPLACE_ = 26
    CLEAR_FRAME_ = 27

class Executor(object):
    """Base executor class.

    An executor is similar to a basic operator in dataflow-based framework.
    The executor can be evaluated by the ``run`` function.
    """
    @abstractmethod
    def opcode(self):
        """Return the opcode of this executor."""
        raise NotImplementedError

    @abstractmethod
    def arg_vars(self):
        """Return the argument variable list of this executor."""
        raise NotImplementedError

    @abstractmethod
    def ret_var(self):
        """Return the result variable of this executor."""
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """Evaluate this executor.

        The function takes no argument and returns none, which means all the
        argument and result variables must be pre-bound.
        """
        raise NotImplementedError

class NodeUDFExecutor(Executor):
    """Executor for Node UDF call.

    Parameters
    ----------
    fn : var.Var
        The UDF.
    fdnode : var.Var
        The node feature dict.
    fdmail : var.Var
        The mailbox data dict.
    ret : var.Var
        The return new node feature dict.
    """
    def __init__(self, fn, fdnode, fdmail, ret):
        self.fn = fn
        self.fdnode = fdnode
        self.fdmail = fdmail
        self.ret = ret

    def opcode(self):
        return OpCode.NODE_UDF

    def arg_vars(self):
        if self.fdmail is None:
            return [self.fn, self.fdnode]
        else:
            return [self.fn, self.fdnode, self.fdmail]

    def ret_var(self):
        return self.ret

    def run(self):
        fn_data = self.fn.data
        node_data = self.fdnode.data
        if self.fdmail is None:
            udf_ret = fn_data(node_data)
        else:
            mail_data = self.fdmail.data
            udf_ret = fn_data(node_data, mail_data)
        self.ret.data = FrameRef(Frame(udf_ret))

IR_REGISTRY[OpCode.NODE_UDF] = {
    'name' : 'NODE_UDF',
    'args_type' : [VarType.FUNC, VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : NodeUDFExecutor,
}

def NODE_UDF(fn, fdnode, fdmail=None, ret=None):
    """Apply the node UDF and get the new node feature symbolically.

    Parameters
    ----------
    fn : var.Var
        The UDF.
    fdnode : var.Var
        The node feature dict.
    fdmail : var.Var
        The mailbox data dict.
    ret : var.Var, optional
        The return variable for new node feature dict. If not give,
        a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.NODE_UDF]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fn, fdnode, fdmail, ret))
    return ret

class EdgeUDFExecutor(Executor):
    """Executor for edge UDF call.

    Parameters
    ----------
    fn : var.Var
        The UDF.
    fdsrc : var.Var
        The src node feature dict.
    fdedge : var.Var
        The edge feature dict.
    fddst : var.Var
        The dst node feature dict.
    ret : var.Var
        The return new edge feature dict.
    """
    def __init__(self, fn, fdsrc, fdedge, fddst, ret):
        self.fn = fn
        self.fdsrc = fdsrc
        self.fdedge = fdedge
        self.fddst = fddst
        self.ret = ret

    def opcode(self):
        return OpCode.EDGE_UDF

    def arg_vars(self):
        return [self.fn, self.fdsrc, self.fdedge, self.fddst]

    def ret_var(self):
        return self.ret

    def run(self):
        fn_data = self.fn.data
        src_data = self.fdsrc.data
        edge_data = self.fdedge.data
        dst_data = self.fddst.data
        udf_ret = fn_data(src_data, edge_data, dst_data)
        self.ret.data = FrameRef(Frame(udf_ret))

IR_REGISTRY[OpCode.EDGE_UDF] = {
    'name' : 'EDGE_UDF',
    'args_type' : [VarType.FUNC, VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : EdgeUDFExecutor,
}
def EDGE_UDF(fn, fdsrc, fdedge, fddst, ret=None):
    """Apply the edge UDF and get the new edge feature symbolically.

    Parameters
    ----------
    fn : var.Var
        The UDF.
    fdsrc : var.Var
        The src node feature dict.
    fdedge : var.Var
        The edge feature dict.
    fddst : var.Var
        The dst node feature dict.
    ret : var.Var, optional
        The return variable for new node feature dict. If not give,
        a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.EDGE_UDF]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fn, fdsrc, fdedge, fddst, ret))
    return ret

class ReadExecutor(Executor):
    """Executor for read data from feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    col : var.Var
        The column name.
    ret : var.Var
        The return feature tensor.
    """
    def __init__(self, fd, row, col, ret):
        self.fd = fd
        self.row = row
        self.col = col
        self.ret = ret

    def opcode(self):
        return OpCode.READ

    def arg_vars(self):
        return [self.fd, self.row, self.col]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        col_data = self.col.data  # key str
        self.ret.data = fd_data[row_data][col_data]

IR_REGISTRY[OpCode.READ] = {
    'name' : 'READ',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.STR],
    'ret_type' : VarType.FEAT,
    'executor_cls' : ReadExecutor,
}

def READ(fd, row, col, ret=None):
    """Read the feature data from the dictionary specified by the row and column symbolically.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    col : var.Var
        The column name.
    ret : var.Var, optional
        The return feature tensor. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.READ]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd, row, col, ret))
    return ret

class ReadColExecutor(Executor):
    """Executor for read column data from feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    col : var.Var
        The column name.
    ret : var.Var
        The return feature tensor.
    """
    def __init__(self, fd, col, ret):
        self.fd = fd
        self.col = col
        self.ret = ret

    def opcode(self):
        return OpCode.READ_COL

    def arg_vars(self):
        return [self.fd, self.col]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_data = self.fd.data
        col_data = self.col.data
        self.ret.data = fd_data[col_data]

IR_REGISTRY[OpCode.READ_COL] = {
    'name' : 'READ_COL',
    'args_type' : [VarType.FEAT_DICT, VarType.STR],
    'ret_type' : VarType.FEAT,
    'executor_cls' : ReadColExecutor,
}

def READ_COL(fd, col, ret=None):
    """Read the column data from the dictionary.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    col : var.Var
        The column name.
    ret : var.Var, optional
        The return feature tensor. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.READ_COL]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd, col, ret))
    return ret

class ReadRowExecutor(Executor):
    """Executor for read row data from feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    ret : var.Var
        The return feature tensor.
    """
    def __init__(self, fd, row, ret):
        self.fd = fd
        self.row = row
        self.ret = ret

    def opcode(self):
        return OpCode.READ_ROW

    def arg_vars(self):
        return [self.fd, self.row]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_data = self.fd.data
        row_data = self.row.data  # idx
        self.ret.data = fd_data[row_data]

IR_REGISTRY[OpCode.READ_ROW] = {
    'name' : 'READ_ROW',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : ReadRowExecutor,
}

def READ_ROW(fd, row, ret=None):
    """Read the row data from the dictionary.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    ret : var.Var, optional
        The return feature tensor. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.READ_ROW]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd, row, ret))
    return ret

class SPMVExecutor(Executor):
    """Executor for sparse-matrix-dense-matrix multiply.

    Parameters
    ----------
    spA : var.Var
        Variable for sparse matrix lambda. The lambda returns the sparse matrix
        given a context object.
    B : var.Var
        Variable for the dense feature tensor.
    ret : var.Var
        Variable for the result.
    """
    def __init__(self, spA, B, ret):
        self.spA = spA
        self.B = B
        self.ret = ret

    def opcode(self):
        return OpCode.SPMV

    def arg_vars(self):
        return [self.spA, self.B]

    def ret_var(self):
        return self.ret

    def run(self):
        spA_ctx_fn = self.spA.data
        B = self.B.data
        ctx = F.context(B)
        spA = spA_ctx_fn(ctx)
        if F.ndim(B) == 1:
            # B is a vector, append a (1,) dim at the end
            B = F.unsqueeze(B, 1)
            C = F.spmm(spA, B)
            C = F.squeeze(C, 1)
        elif F.ndim(B) > 2:
            # Flatten the dim 1:~
            B_shape = F.shape(B)
            feat_shape = B_shape[1:]
            tmp_B_shape = (B_shape[0], functools.reduce(operator.mul, feat_shape, 1))
            B = F.reshape(B, tmp_B_shape)
            C = F.spmm(spA, B)
            C_shape = (F.shape(C)[0],) + feat_shape
            C = F.reshape(C, C_shape)
        else:
            C = F.spmm(spA, B)
        self.ret.data = C

IR_REGISTRY[OpCode.SPMV] = {
    'name' : 'SPMV',
    'args_type' : [VarType.SPMAT, VarType.FEAT],
    'ret_type' : VarType.FEAT,
    'executor_cls' : SPMVExecutor,
}

def SPMV(spA, B, ret=None):
    """Perform sparse-matrix-dense-matrix multiply symbolically.

    Parameters
    ----------
    spA : var.Var
        Variable for sparse matrix lambda. The lambda returns the sparse matrix
        given a context object.
    B : var.Var
        Variable for the dense feature tensor.
    ret : var.Var, optional
        Variable for the result. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.SPMV]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](spA, B, ret))
    return ret

class SPMVWithDataExecutor(Executor):
    """Executor for sparse-matrix-dense-matrix multiply with provided sparse data.

    Parameters
    ----------
    spA : var.Var
        Variable for sparse matrix lambda. The lambda returns the sparse matrix
        given a context object.
    A_data : var.Var
        Variable for the sparse matrix data.
    B : var.Var
        Variable for the dense feature tensor.
    ret : var.Var
        Variable for the result.
    """
    def __init__(self, spA, A_data, B, ret):
        self.spA = spA
        self.A_data = A_data
        self.B = B
        self.ret = ret

    def opcode(self):
        return OpCode.SPMV_WITH_DATA

    def arg_vars(self):
        return [self.spA, self.A_data, self.B]

    def ret_var(self):
        return self.ret

    def run(self):
        spA_ctx_fn = self.spA.data
        A_data = self.A_data.data
        if F.ndim(A_data) > 1:
            # A_data is of shape (E, 1). Squeeze the last dim.
            A_data = F.squeeze(A_data, 1)
        B = self.B.data

        ctx = F.context(B)
        spA = spA_ctx_fn(ctx)
        spidx = F.sparse_matrix_indices(spA)
        shape = F.shape(spA)
        # shuffle index is not used
        spA, _ = F.sparse_matrix(A_data, spidx, shape)

        if F.ndim(B) == 1:
            # B is a vector, append a (1,) dim at the end
            B = F.unsqueeze(B, 1)
            C = F.spmm(spA, B)
            C = F.squeeze(C, 1)
        elif F.ndim(B) > 2:
            # Flatten the dim 1:~
            B_shape = F.shape(B)
            feat_shape = B_shape[1:]
            tmp_B_shape = (B_shape[0], functools.reduce(operator.mul, feat_shape, 1))
            B = F.reshape(B, tmp_B_shape)
            C = F.spmm(spA, B)
            C_shape = (F.shape(C)[0],) + feat_shape
            C = F.reshape(C, C_shape)
        else:
            C = F.spmm(spA, B)
        self.ret.data = C

IR_REGISTRY[OpCode.SPMV_WITH_DATA] = {
    'name' : 'SPMV_WITH_DATA',
    'args_type' : [VarType.SPMAT, VarType.FEAT, VarType.FEAT],
    'ret_type' : VarType.FEAT,
    'executor_cls' : SPMVWithDataExecutor,
}

def SPMV_WITH_DATA(spA, A_data, B, ret=None):
    """Perform sparse-matrix-dense-matrix multiply with sparse data symbolically.

    Parameters
    ----------
    spA : var.Var
        Variable for sparse matrix lambda. The lambda returns the sparse matrix
        given a context object.
    A_data : var.Var
        Variable for the sparse matrix data.
    B : var.Var
        Variable for the dense feature tensor.
    ret : var.Var, optional
        Variable for the result. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.SPMV_WITH_DATA]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](spA, A_data, B, ret))
    return ret

class MergeRowExecutor(Executor):
    """Executor for merge row data according to the given order.

    Parameters
    ----------
    order : var.Var
        The order index.
    fd_list : list of var.Var
        The list of row data variables. Each represents a feature dict.
    ret : var.Var
        Variable for the result.
    """
    def __init__(self, order, fd_list, ret):
        self.order = order
        self.fd_list = fd_list
        self.ret = ret

    def opcode(self):
        return OpCode.MERGE_ROW

    def arg_vars(self):
        return [self.order] + self.fd_list

    def ret_var(self):
        return self.ret

    def run(self):
        # merge buckets according to the ascending order of the node ids.
        order_data = self.order.data
        fd_data = [fd.data for fd in self.fd_list]
        keys = fd_data[0].keys()
        all_fd = {key : F.cat([fd[key] for fd in fd_data], dim=0)
                  for key in keys}
        ret_fd = utils.reorder(all_fd, order_data)
        self.ret.data = ret_fd

IR_REGISTRY[OpCode.MERGE_ROW] = {
    'name' : 'MERGE_ROW',
    'args_type' : [VarType.IDX, VarType.IDX, '*', VarType.FEAT_DICT, '*'],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : MergeRowExecutor,
}

def MERGE_ROW(idx_list, fd_list, ret=None):
    """Merge row data according to the given order symbolically.

    Parameters
    ----------
    order : var.Var
        The order index.
    fd_list : list of var.Var
        The list of row data variables. Each represents a feature dict.
    ret : var.Var, optional
        Variable for the result. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.MERGE_ROW]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](idx_list, fd_list, ret))
    return ret

class UpdateDictExecutor(Executor):
    """Executor for update feature dictionary with another one.

    Similar to python dict's update but return a new dictionary.

    Parameters
    ----------
    fd1 : var.Var
        Variable for the feature dict to be updated.
    fd2 : var.Var
        Variable for the provided feature dict.
    ret : var.Var
        Variable for the result.
    """
    def __init__(self, fd1, fd2, ret):
        self.fd1 = fd1
        self.fd2 = fd2
        self.ret = ret

    def opcode(self):
        return OpCode.UPDATE_DICT

    def arg_vars(self):
        return [self.fd1, self.fd2]

    def ret_var(self):
        return self.ret

    def run(self):
        fd1_data = self.fd1.data
        fd2_data = self.fd2.data
        if (isinstance(fd1_data, utils.LazyDict)
                or isinstance(fd2_data, utils.LazyDict)):
            # NOTE: fd2 has higher priority
            ret_data = utils.HybridDict(fd2_data, fd1_data)
        else:
            ret_data = {k : v for k, v in fd1_data.items()}
            ret_data.update(fd2_data)
        self.ret.data = ret_data

IR_REGISTRY[OpCode.UPDATE_DICT] = {
    'name' : 'UPDATE_DICT',
    'args_type' : [VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : UpdateDictExecutor,
}

def UPDATE_DICT(fd1, fd2, ret=None):
    """Executor for update feature dictionary with another one.

    Similar to python dict's update but return a new dictionary.

    Parameters
    ----------
    fd1 : var.Var
        Variable for the feature dict to be updated.
    fd2 : var.Var
        Variable for the provided feature dict.
    ret : var.Var, optional
        Variable for the result. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.UPDATE_DICT]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd1, fd2, ret))
    return ret

class NewDictExecutor(Executor):
    """Executor for creating new feature dictionary.

    Parameters
    ----------
    fd_init : var.Var
        The feat dict to borrow initializer.
    idx : var.Var
        The index to look for number or rows.
    fd_scheme : var.Var
        The feat dict to look for column scheme.
    ret : var.Var
        Variable for the result.
    """
    def __init__(self, fd_init, idx, fd_scheme, ret):
        self.fd_init = fd_init  # the feat dict to borrow initializer
        self.idx = idx  # the index to look for number or rows
        self.fd_scheme = fd_scheme  # the feat dict to look for column scheme
        self.ret = ret  # the result

    def opcode(self):
        return OpCode.NEW_DICT

    def arg_vars(self):
        return [self.fd_init, self.idx, self.fd_scheme]

    def ret_var(self):
        return self.ret

    def run(self):
        fd_init_data = self.fd_init.data
        idx_data = self.idx.data
        fd_scheme_data = self.fd_scheme.data
        schemes = fd_scheme_data.schemes
        ret_dict = {}
        for key, sch in schemes.items():
            initializer = fd_init_data.get_initializer(key)
            ctx = F.context(fd_scheme_data[key])
            shape = (len(idx_data),) + sch.shape
            # FIXME: the last argument here can only be idx; range
            #   is meaningless. Need to rethink the signature.
            ret_dict[key] = initializer(shape, sch.dtype, ctx, idx_data)
        self.ret.data = FrameRef(Frame(ret_dict))

IR_REGISTRY[OpCode.NEW_DICT] = {
    'name' : 'NEW_DICT',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.FEAT_DICT],
    'ret_type' : VarType.FEAT_DICT,
    'executor_cls' : NewDictExecutor,
}

def NEW_DICT(fd_init, idx, fd_scheme, ret=None):
    """Create a new dictionary symbolically.

    Parameters
    ----------
    fd_init : var.Var
        The feat dict to borrow initializer.
    idx : var.Var
        The index to look for number or rows.
    fd_scheme : var.Var
        The feat dict to look for column scheme.
    ret : var.Var
        Variable for the result. If not give, a new variable will be created.

    Returns
    -------
    var.Var
        Variable for the result.
    """
    reg = IR_REGISTRY[OpCode.NEW_DICT]
    ret = var.new(reg['ret_type']) if ret is None else ret
    get_current_prog().issue(reg['executor_cls'](fd_init, idx, fd_scheme, ret))
    return ret

class Write_Executor(Executor):
    """Executor for writing the given data to the feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    col : var.Var
        The column name.
    val : var.Var
        The given feature data.
    """
    def __init__(self, fd, row, col, val):
        self.fd = fd
        self.row = row
        self.col = col
        self.val = val

    def opcode(self):
        return OpCode.WRITE_

    def arg_vars(self):
        return [self.fd, self.row, self.col, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        col_data = self.col.data  # key str
        val_data = self.val.data
        fd_data[col_data][row_data] = val_data

IR_REGISTRY[OpCode.WRITE_] = {
    'name' : 'WRITE_',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.STR, VarType.FEAT],
    'ret_type' : None,
    'executor_cls' : Write_Executor,
}

def WRITE_(fd, row, col, val):
    """Write the given data to the feature dict symbolically.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    col : var.Var
        The column name.
    val : var.Var
        The given feature data.
    """
    reg = IR_REGISTRY[OpCode.WRITE_]
    get_current_prog().issue(reg['executor_cls'](fd, row, col, val))

class WriteCol_Executor(Executor):
    """Executor for writing the given column data to the feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    col : var.Var
        The column name.
    val : var.Var
        The given feature data.
    """
    def __init__(self, fd, col, val):
        self.fd = fd
        self.col = col
        self.val = val

    def opcode(self):
        return OpCode.WRITE_COL_

    def arg_vars(self):
        return [self.fd, self.col, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        col_data = self.col.data  # key str
        val_data = self.val.data
        fd_data[col_data] = val_data

IR_REGISTRY[OpCode.WRITE_COL_] = {
    'name' : 'WRITE_COL_',
    'args_type' : [VarType.FEAT_DICT, VarType.STR, VarType.FEAT],
    'ret_type' : None,
    'executor_cls' : WriteCol_Executor,
}

def WRITE_COL_(fd, col, val):
    """Writing the given column data to the feature dict symbolically.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    col : var.Var
        The column name.
    val : var.Var
        The given feature data.
    """
    reg = IR_REGISTRY[OpCode.WRITE_COL_]
    get_current_prog().issue(reg['executor_cls'](fd, col, val))

class WriteRow_Executor(Executor):
    """Executor for writing the given row data to the feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    val : var.Var
        The given feature data.
    """
    def __init__(self, fd, row, val):
        self.fd = fd
        self.row = row
        self.val = val

    def opcode(self):
        return OpCode.WRITE_ROW_

    def arg_vars(self):
        return [self.fd, self.row, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        val_data = self.val.data
        fd_data[row_data] = val_data

IR_REGISTRY[OpCode.WRITE_ROW_] = {
    'name' : 'WRITE_ROW_',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : WriteRow_Executor,
}

def WRITE_ROW_(fd, row, val):
    """Write the given row data to the feature dict symbolically.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    val : var.Var
        The given feature data.
    """
    reg = IR_REGISTRY[OpCode.WRITE_ROW_]
    get_current_prog().issue(reg['executor_cls'](fd, row, val))

class WriteRowInplace_Executor(Executor):
    """Executor for writing the given row data to the feature dict in-place.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    val : var.Var
        The given feature data.
    """
    def __init__(self, fd, row, val):
        self.fd = fd
        self.row = row
        self.val = val

    def opcode(self):
        return OpCode.WRITE_ROW_INPLACE_

    def arg_vars(self):
        return [self.fd, self.row, self.val]

    def ret_var(self):
        return None

    def run(self):
        fd_data = self.fd.data  # feature dict
        row_data = self.row.data  # idx
        val_data = self.val.data
        fd_data.update_data(row_data, val_data, inplace=True)

IR_REGISTRY[OpCode.WRITE_ROW_INPLACE_] = {
    'name' : 'WRITE_ROW_INPLACE_',
    'args_type' : [VarType.FEAT_DICT, VarType.IDX, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : WriteRowInplace_Executor,
}

def WRITE_ROW_INPLACE_(fd, row, val):
    """Write the given row data to the feature dict in-place symbolically.

    Parameters
    ----------
    fd : var.Var
        The feature dict.
    row : var.Var
        The row index.
    val : var.Var
        The given feature data.
    """
    reg = IR_REGISTRY[OpCode.WRITE_ROW_INPLACE_]
    get_current_prog().issue(reg['executor_cls'](fd, row, val))

class WriteDict_Executor(Executor):
    """Executor for writing the given feature dict data into the another one.

    Parameters
    ----------
    fd1 : var.Var
        The feature dict to be mutated.
    fd2 : var.Var
        The feature dict data.
    """
    def __init__(self, fd1, fd2):
        self.fd1 = fd1
        self.fd2 = fd2

    def opcode(self):
        return OpCode.WRITE_DICT_

    def arg_vars(self):
        return [self.fd1, self.fd2]

    def ret_var(self):
        return None

    def run(self):
        fd1_data = self.fd1.data
        fd2_data = self.fd2.data
        for k, v in fd2_data.items():
            fd1_data[k] = v

IR_REGISTRY[OpCode.WRITE_DICT_] = {
    'name' : 'WRITE_DICT_',
    'args_type' : [VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : WriteDict_Executor,
}

def WRITE_DICT_(fd1, fd2):
    """Writing the given feature dict data into the another one symbolically.

    Parameters
    ----------
    fd1 : var.Var
        The feature dict to be mutated.
    fd2 : var.Var
        The feature dict data.
    """
    reg = IR_REGISTRY[OpCode.WRITE_DICT_]
    get_current_prog().issue(reg['executor_cls'](fd1, fd2))

class AppendRow_Executor(Executor):
    """Executor for appending one feature dict to another.

    Parameters
    ----------
    fd1 : var.Var
        The feature dict in the front.
    fd2 : var.Var
        The feature dict in the back.
    """
    def __init__(self, fd1, fd2):
        self.fd1 = fd1
        self.fd2 = fd2

    def opcode(self):
        return OpCode.APPEND_ROW_

    def arg_vars(self):
        return [self.fd1, self.fd2]

    def ret_var(self):
        return None

    def run(self):
        fd1_data = self.fd1.data
        fd2_data = self.fd2.data
        fd1_data.append(fd2_data)

IR_REGISTRY[OpCode.APPEND_ROW_] = {
    'name' : 'APPEND_ROW_',
    'args_type' : [VarType.FEAT_DICT, VarType.FEAT_DICT],
    'ret_type' : None,
    'executor_cls' : AppendRow_Executor,
}
def APPEND_ROW_(fd1, fd2):
    """Append one feature dict to another symbolically.

    Parameters
    ----------
    fd1 : var.Var
        The feature dict in the front.
    fd2 : var.Var
        The feature dict in the back.
    """
    reg = IR_REGISTRY[OpCode.APPEND_ROW_]
    get_current_prog().issue(reg['executor_cls'](fd1, fd2))

class ClearFrame_Executor(Executor):
    """Executor for clear the feature dict.

    Parameters
    ----------
    fd : var.Var
        The feature dict to be cleared.
    """
    def __init__(self, fd):
        self.fd = fd

    def opcode(self):
        return OpCode.CLEAR_FRAME_

    def arg_vars(self):
        return [self.fd]

    def ret_var(self):
        return None

    def run(self):
        frame = self.fd.data
        num_rows = frame.num_rows
        frame.clear()
        frame.add_rows(num_rows)

IR_REGISTRY[OpCode.CLEAR_FRAME_] = {
    'name': 'CLEAR_FRAME_',
    'args_type': [VarType.FEAT_DICT],
    'ret_type': None,
    'executor_cls': ClearFrame_Executor,
}

def CLEAR_FRAME_(fd):
    """Clear the feature dict symbolically.

    Parameters
    ----------
    fd : var.Var
        The feature dict to be cleared.
    """
    reg = IR_REGISTRY[OpCode.CLEAR_FRAME_]
    get_current_prog().issue(reg['executor_cls'](fd))
