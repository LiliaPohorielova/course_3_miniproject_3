package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

public class MatrixMult {
    public static void parallelMatrixMultiply(Matrix matrixA, Matrix matrixB, Matrix resultMatrix,
                                              final MPI mpi) throws MPIException {
        int processRank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);
        int processCount = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);
        int totalRows = resultMatrix.getNRows();
        int rowsPerProcess = (totalRows + processCount - 1) / processCount;
        int startRow = processRank * rowsPerProcess;
        int endRow = Math.min((processRank + 1) * rowsPerProcess, totalRows);

        mpi.MPI_Bcast(matrixA.getValues(), 0, matrixA.getNRows() * matrixA.getNCols(), 0, mpi.MPI_COMM_WORLD);
        mpi.MPI_Bcast(matrixB.getValues(), 0, matrixB.getNRows() * matrixB.getNCols(), 0, mpi.MPI_COMM_WORLD);

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < resultMatrix.getNCols(); j++) {
                resultMatrix.set(i, j, 0.0);

                for (int k = 0; k < matrixB.getNRows(); k++) {
                    resultMatrix.incr(i, j, matrixA.get(i, k) * matrixB.get(k, j));
                }
            }
        }

        if (processRank == 0) {
            MPI.MPI_Request[] receiveRequests = new MPI.MPI_Request[processCount - 1];
            for (int i = 1; i < processCount; i++) {
                int rankStartRow = i * rowsPerProcess;
                int rankEndRow = Math.min((i + 1) * rowsPerProcess, totalRows);

                final int rowOffset = rankStartRow * resultMatrix.getNCols();
                final int elementCount = (rankEndRow - rankStartRow) * resultMatrix.getNCols();

                receiveRequests[i - 1] = mpi.MPI_Irecv(resultMatrix.getValues(), rowOffset, elementCount, i, i, mpi.MPI_COMM_WORLD);
            }
            mpi.MPI_Waitall(receiveRequests);
        } else {
            mpi.MPI_Send(resultMatrix.getValues(), startRow * resultMatrix.getNCols(), (endRow - startRow) * resultMatrix.getNCols(), 0, processRank, mpi.MPI_COMM_WORLD);
        }
    }
}
