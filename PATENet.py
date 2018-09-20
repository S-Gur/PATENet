# This code includes an implementation of the PATENet family of algorithms as described in
#
# Gur, S., & Honavar, V. G. (2018, July). PATENet: Pairwise Alignment of Time Evolving Networks. In International
# Conference on Machine Learning and Data Mining in Pattern Recognition (pp. 85-98). Springer, Cham.
#
# Please cite this paper if you use this code.


import sys
import numpy as np
from scipy import sparse
from inspect import signature
from typing import Callable, Any, Iterable, Tuple
import collections
from enum import Enum

class MV(Enum):
    END, DIAG, UP, LEFT = range(0, 4)


def PATENet(i_fnSim: Callable, i_fnTrns: Callable, i_fMtchTH: float, i_arrTMLG1: Iterable[Any],
            i_arrTMLG2: Iterable[Any]) -> (str, str, float):
    """PATENet: find the best local alignment between two temporal sequences of objects (networks; TMLNs) based on
       provided similarity measure between the objects comprising the sequences (networks), a monotone transform
       function, and an object-match threshold.

    :param i_fnSim: (Callable) Similarity function between the two objects of the type making the sequences. Should
           take two Args for two objects; any additional Args should have default values.
    :param i_fnTrns: (Callable) Monotone transform function, taking float Arg for the value to be transformed, and
           float Arg for the match threshold; any additional Args should have default values.
    :param i_fMtchTH: (float) Match threshold for SM's construction.
    :param i_arrTMLG1: (Iterable[Any]) The first temporal sequence of objects (networks; TMLNs) to be aligned.
    :param i_arrTMLG2: (Iterable[Any]) The second temporal sequence of objects (networks; TMLNs) to be aligned.

    :return str: The aligned segment within the first TMLN (i_arrTMLG1), tab separated.
    :return str: The aligned segment within the second TMLN (i_arrTMLG2), tab separated.
    :return float: The similarity score (according to the dynamic programming algorithm).

    """

    # Arguments validity checks:
    if not callable(i_fnSim):
        sys.exit("First argument to PATENet must be a valid function.")
    elif not callable(i_fnTrns):
        sys.exit("Second argument to PATENet must be a valid function.")
    sig = signature(i_fnSim)
    if 2 > len(sig.parameters):
        sys.exit("First argument to PATENet must be a similarity function between two objects, each of which should be "
                 "a separate argument to the function.")
    elif 2 < len(sig.parameters):
        lstPrms = list(sig.parameters)
        for i in range(2, np.size(lstPrms)):
            strExtraPrm = str(sig.parameters[lstPrms[i]])
            if -1 == strExtraPrm.find('='):
                sys.exit("First argument to PATENet must be a similarity function between two objects; any additional "
                         "arguments beyond the two objects must have default values.")
    sig = signature(i_fnTrns)
    if 2 > len(sig.parameters):
        sys.exit("Second argument to PATENet must be a monotone transform function that takes the orinigal value as its"
                 "first argument and a threshold as its second argument.")
    elif 2 < len(sig.parameters):
        lstPrms = list(sig.parameters)
        for i in range(2, np.size(lstPrms)):
            strExtraPrm = str(sig.parameters[lstPrms[i]])
            if -1 == strExtraPrm.find('='):
                sys.exit("Second argument to PATENet must be a monotone transform function that takes the orinigal "
                         "value as its first argument and a threshold as its second argument; any additional arguments "
                         "beyond these two must have default values.")
    if not isinstance(i_fMtchTH, float) or (0. > i_fMtchTH) or (1. < i_fMtchTH):
        sys.exit("Third argument to PATENet must be float between 0 and 1.")
    if not isinstance(i_arrTMLG1, collections.Iterable):
        sys.exit("Fourth argument to PATENet must be the first sequence to be aligned and should be iterable.")
    if not isinstance(i_arrTMLG2, collections.Iterable):
        sys.exit("Fifth argument to PATENet must be the second sequence to be aligned and should be iterable.")

    # PATENet's algorithm:
    mtxSM = pn_genSimMtrx(i_fnSim, i_fnTrns, i_fMtchTH, i_arrTMLG1, i_arrTMLG2)
    mtxScrMtrx, tplMaxPos, fMaxScr = pn_genScrMtrx(i_arrTMLG1, i_arrTMLG2, mtxSM)
    strAlgnSeq1, strAlgnSeq2 = pn_traceback(mtxScrMtrx, tplMaxPos, np.size(i_arrTMLG1, axis=0),
                                            np.size(i_arrTMLG2, axis=0), mtxSM)

    return strAlgnSeq1, strAlgnSeq2, fMaxScr

# Assist-functions:


def pn_genSimMtrx(i_fnSim: Callable, i_fnTrns: Callable, i_fMtchTH: float, i_arrTMLG1: Iterable[Any],
                  i_arrTMLG2: Iterable[Any]) -> Iterable[Iterable[float]]:
    """Generating SM for the elemets of the two sequences based on the provided similarity measure between, monotone
       transform function, and object-match threshold.

    :param i_fnSim: (Callable) Similarity function between the two objects of the type making the sequences. Should
           take two Args for two objects; any additional Args should have default values.
    :param i_fnTrns: (Callable) Monotone transform function, taking float Arg for the value to be transformed, and
           float Arg for the match threshold; any additional Args should have default values.
    :param i_fMtchTH: (float) Match threshold for SM's construction.
    :param i_arrTMLG1: (Iterable[Any]) The first temporal sequence of objects (networks; TMLNs) to be aligned.
    :param i_arrTMLG2: (Iterable[Any]) The second temporal sequence of objects (networks; TMLNs) to be aligned.

    :return Iterable[Iterable[float]]: SM for the elements of the two sequences.

    """

    if sparse.issparse(i_arrTMLG1[0]):
        nofNodes = sparse.coo_matrix.get_shape(i_arrTMLG1[0])[0]
    else:
        nofNodes = np.size(i_arrTMLG1[0], axis=0)
    # Validate all layers in both sequences have the same number of nodes:
    for obLayer in i_arrTMLG1:
        if sparse.issparse(obLayer):
            if sparse.coo_matrix.get_shape(obLayer)[0] != nofNodes:
                sys.exit("PATENet::pn_genSimMtrx: All layers of both sequences must have the same number of nodes.")
        elif np.size(obLayer, axis=0) != nofNodes:
            sys.exit("PATENet::pn_genSimMtrx: All layers of both sequences must have the same number of nodes.")
    for obLayer in i_arrTMLG2:
        if sparse.issparse(obLayer):
            if sparse.coo_matrix.get_shape(obLayer)[0] != nofNodes:
                sys.exit("PATENet::pn_genSimMtrx: All layers of both sequences must have the same number of nodes.")
        elif np.size(obLayer, axis=0) != nofNodes:
            sys.exit("PATENet::pn_genSimMtrx: All layers of both sequences must have the same number of nodes.")

    iLenTMLG1 = np.size(i_arrTMLG1, axis=0)
    iLenTMLG2 = np.size(i_arrTMLG2, axis=0)
    # Generate SM:
    mtxSM = np.reshape(np.zeros(iLenTMLG1*iLenTMLG2), [iLenTMLG1, iLenTMLG2])
    for i in range(1, np.size(i_arrTMLG1, axis=0) + 1):
        for j in range(1, np.size(i_arrTMLG2, axis=0) + 1):
            fSim = i_fnSim(i_arrTMLG1[i - 1], i_arrTMLG2[j - 1])
            mtxSM[i - 1][j - 1] = i_fnTrns(fSim, i_fMtchTH)

    return mtxSM


def pn_genScrMtrx(i_arrTMLG1: Iterable[Any], i_arrTMLG2: Iterable[Any], i_mtxSM: Iterable[Iterable[float]]) -> \
        (Iterable[Iterable[float]], Tuple, float):
    """Create a matrix of scores representing trial alignments of the two sequences. This function creates the score
    (2D) matrix between two sequences, optimizing cumulative score based on the similarities between their elements.

    :param i_arrTMLG1: (Iterable[Any]) The first temporal sequence of objects (networks; TMLNs) to be aligned.
    :param i_arrTMLG2: (Iterable[Any]) The second temporal sequence of objects (networks; TMLNs) to be aligned.
    :param i_mtxSM: Iterable[Iterable[float]] SM for the elements of the two sequences.

    :return Iterable[Iterable[float]]: The scoring matrix between the two sequences.
    :return Tuple: A location (i,j) of the maximum value in the scoring matrix
    :return float: The similarity score for the two sequences (the maximum value in the scoring matrix).

    """

    iLenTMLG1 = np.size(i_arrTMLG1, axis=0)
    iLenTMLG2 = np.size(i_arrTMLG2, axis=0)

    # Initialize the scoring matrix:
    mtxScrMtrx = np.reshape(np.zeros((iLenTMLG1+1)*(iLenTMLG2+1)), [iLenTMLG1+1, iLenTMLG2+1])
    # Fill the scoring matrix:
    fMaxScr = 0
    tplMaxPos = None  # The row and column of the highest score in matrix.
    for i in range(1, iLenTMLG1 + 1):
        for j in range(1, iLenTMLG2 + 1):
            fScr = pn_clcScr(mtxScrMtrx, i, j, i_mtxSM)
            if fScr > fMaxScr:
                fMaxScr = fScr
                tplMaxPos = (i, j)

            mtxScrMtrx[i][j] = fScr

    if tplMaxPos is None:
        tplMaxPos = (0, 0)

    return mtxScrMtrx, tplMaxPos, fMaxScr


def pn_clcScr(i_mtxScrMtrx: Iterable[Iterable[float]], i_i: int, i_j: int, i_mtxSM: Iterable[Iterable[float]]) -> float:
    """Calculate score for a given position (i_i,i_j) in the scoring matrix, based on the upper, left, and upper-left
    neighbors of the cell.

    :param i_mtxScrMtrx: (Iterable[Iterable[float]]) Current state of the scoring matrix (during dynamic programming).
    :param i_i: (int) Row index within the scoring matrix.
    :param i_j: (int) Column index within the scoring matrix.
    :param i_mtxSM: (Iterable[Iterable[float]]) SM for the elements of the two sequences.

    :return: float: The best cumulative score for an alignment ending with the i_i element of the first sequence being
             aligned with the i_j element of the second sequence

    """

    fSim = i_mtxSM[i_i - 1][i_j - 1]
    gap_pen = 0

    diag_score = i_mtxScrMtrx[i_i - 1][i_j - 1] + fSim
    up_score = i_mtxScrMtrx[i_i - 1][i_j] - gap_pen
    left_score = i_mtxScrMtrx[i_i][i_j - 1] - gap_pen

    return max(0, diag_score, up_score, left_score)


def pn_traceback(i_mtxScrMtrx: Iterable[Iterable[float]], i_tplStrtPos: Tuple, i_iLenTMLG1: int, i_iLenTMLG2: int,
                 i_mtxSM: Iterable[Iterable[float]]) -> (str, str):
    """Find the optimal path according to the scoring matrix. This function traces a path from a position of the
    maximum value in the scoring matrix. Each move corresponds to a match or gap in one of the sequences being aligned.
    Three possible moves are allowed: upper-left diagonal, up, or left. Each of these moves represent the following:
        diagonal: match/mismatch
        up:       gap in sequence 2
        left:     gap in sequence 1

    :param i_mtxScrMtrx: (Iterable[Iterable[float]]) The scoring matrix between the two sequences.
    :param i_tplStrtPos: (Tuple) The position (i,j) to start trace-backing from.
    :param i_iLenTMLG1: (int) The number of elements (networks) in the first sequence.
    :param i_iLenTMLG2: (int) The number of elements (networks) in the second sequence.
    :param i_mtxSM: (Iterable[Iterable[float]]) SM for the elements of the two sequences.

    :return str: The aligned segment within the first TMLN (i_arrTMLG1), tab separated.
    :return str: The aligned segment within the second TMLN (i_arrTMLG2), tab separated.

    """

    lstSeq1 = list(map(str, range(i_iLenTMLG1)))
    lstSeq2 = list(map(str, range(i_iLenTMLG2)))

    arrAlgSeq1 = []
    arrAlgSeq2 = []
    x, y = i_tplStrtPos
    move = pn_nxtMv(i_mtxScrMtrx, x, y, i_mtxSM)

    while move != MV.END:
        if move == MV.DIAG:
            arrAlgSeq1.append("\t" + lstSeq1[x - 1])
            arrAlgSeq2.append("\t" + lstSeq2[y - 1])
            x -= 1
            y -= 1
        elif move == MV.UP:
            arrAlgSeq1.append("\t" + lstSeq1[x - 1])
            arrAlgSeq2.append("\t-")
            x -= 1
        else:
            arrAlgSeq1.append("\t-")
            arrAlgSeq2.append("\t" + lstSeq2[y - 1])
            y -= 1

        if (x > 0) and (y > 0):
            move = pn_nxtMv(i_mtxScrMtrx, x, y, i_mtxSM)
        else:
            move = MV.END

    return ''.join(reversed(arrAlgSeq1)), ''.join(reversed(arrAlgSeq2))


def pn_nxtMv(i_mtxScrMtrx: Iterable[Iterable[float]], i_i: int, i_j: int, i_mtxSM: Iterable[Iterable[float]]) -> int:
    """Part of the traceback process - from a current position (i_i,i_j) in the complete scoring matrix, decide which
       position to continue from next (upper-left diagonal neighbor, upper neighbor or left neighbor) or whether the
       end of the process process (position with 0) has been reached.

    :param i_mtxScrMtrx: (Iterable[Iterable[float]]) The scoring matrix between the two sequences.
    :param i_i: (int) Row index within the scoring matrix.
    :param i_j: (int) Column index within the scoring matrix.
    :param i_mtxSM: (Iterable[Iterable[float]]) SM for the elements of the two sequences.

    :return int: 1 for diagonal move (MV.DIAG), 2 for upwards move (MV.UP), 3 for left move (MV.LEFT), and 0 for
            process termination (MV.END)

    """

    fSim = i_mtxSM[i_i - 1][i_j - 1]
    fCurrPosVal = i_mtxScrMtrx[i_i][i_j]
    fDiagVal = i_mtxScrMtrx[i_i - 1][i_j - 1]
    fUpVal = i_mtxScrMtrx[i_i - 1][i_j]
    fLeftVal = i_mtxScrMtrx[i_i][i_j - 1]

    if fCurrPosVal == fDiagVal + fSim:
        return MV.DIAG
    elif fCurrPosVal == fUpVal:
        return MV.UP
    elif fCurrPosVal == fLeftVal:
        return MV.LEFT
    else:
        return MV.END
