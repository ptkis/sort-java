package quannm

import org.opencv.core.Mat

class HungarianAlgorithm {
    fun Solve(DistMatrix: Mat, Assignment: ArrayList<Int>): Double {
//        System.out.println("solve");
        val nRows = DistMatrix.size().height.toInt()
        val nCols = DistMatrix.size().width.toInt()
        //        System.out.println("hung rowcol "+ nRows+" "+nCols);
        val distMatrixIn = DoubleArray(nRows * nCols)
        val assignment = IntArray(nRows)
        val cost = 0.0
        for (i in 0 until nRows) for (j in 0 until nCols) distMatrixIn[i + nRows * j] = DistMatrix[i, j][0]
        assignmentoptimal(assignment, cost, distMatrixIn, nRows, nCols)
        Assignment.clear()
        for (r in 0 until nRows) {
//            System.out.println(assignment[r]);
            Assignment.add(assignment[r])
        }
        return cost
    }

    fun assignmentoptimal(
        assignment: IntArray,
        cost: Double,
        distMatrixIn: DoubleArray,
        nOfRows: Int,
        nOfColumns: Int
    ) {
        /* initialization */
//        System.out.println("assignment optimal");
        var cost = cost
        cost = 0.0
        for (row in 0 until nOfRows) assignment[row] = -1
        /* generate working copy of distance Matrix */
        /* check if all matrix elements are positive */
        val nOfElements = nOfRows * nOfColumns
        val distMatrix = DoubleArray(nOfElements)
        for (row in 0 until nOfElements) {
            val value = distMatrixIn[row]
            assert(value >= 0) { "All matrix elements have to be non-negative.\n" }
            distMatrix[row] = value
        }
        val coveredColumns = BooleanArray(nOfColumns)
        val coveredRows = BooleanArray(nOfRows)
        val starMatrix = BooleanArray(nOfElements)
        val primeMatrix = BooleanArray(nOfElements)
        val newStarMatrix = BooleanArray(nOfElements)
        val minDim: Int
        /* preliminary steps */if (nOfRows <= nOfColumns) {
            minDim = nOfRows
            var minValue: Double
            var value: Double
            for (row in 0 until nOfRows) {
                var current_index = row
                minValue = distMatrix[current_index]
                current_index += nOfRows
                while (current_index < nOfElements) {
                    value = distMatrix[current_index]
                    if (value < minValue) {
                        minValue = value
                    }
                    current_index += nOfRows
                }
                current_index = row
                while (current_index < nOfElements) {
                    distMatrix[current_index] -= minValue
                    current_index += nOfRows
                }
            }
            /* Steps 1 and 2a */for (row in 0 until nOfRows) {
                for (col in 0 until nOfColumns) {
                    if (Math.abs(distMatrix[row + nOfRows * col]) < DBL_EPSILON) {
                        if (!coveredColumns[col]) {
                            starMatrix[row + nOfRows * col] = true
                            coveredColumns[col] = true
                            break
                        }
                    }
                }
            }
        } else  /* if(nOfRows > nOfColumns) */ {
            minDim = nOfColumns
            var value: Double
            var current_index: Int
            for (col in 0 until nOfColumns) {
                /* find the smallest element in the column */
                current_index = nOfRows * col
                val columnEnd = nOfRows * col + nOfRows
                var minValue = distMatrix[current_index]
                while (current_index < columnEnd) {
                    value = distMatrix[current_index]
                    current_index++
                    if (value < minValue) minValue = value
                }
                /* subtract the smallest element from each element of the column */current_index = nOfRows * col
                while (current_index < columnEnd) {
                    distMatrix[current_index] -= minValue
                    current_index++
                }
            }
            /* Steps 1 and 2a */for (col in 0 until nOfColumns) {
                for (row in 0 until nOfRows) {
                    if (Math.abs(distMatrix[row + nOfRows * col]) < DBL_EPSILON) {
                        if (!coveredRows[row]) {
                            starMatrix[row + nOfRows * col] = true
                            coveredColumns[col] = true
                            coveredRows[row] = true
                            break
                        }
                    }
                }
            }
            for (row in 0 until nOfRows) coveredRows[row] = false
        }

        /* move to step 2b */step2b(
            assignment,
            distMatrix,
            starMatrix,
            newStarMatrix,
            primeMatrix,
            coveredColumns,
            coveredRows,
            nOfRows,
            nOfColumns,
            minDim
        )
        /* compute cost and remove invalid assignments */computeassignmentcost(assignment, cost, distMatrixIn, nOfRows)
        return
    }

    fun buildassignmentvector(assignment: IntArray, starMatrix: BooleanArray, nOfRows: Int, nOfColumns: Int) {
//        System.out.println("buildassignmentvector");
        var row: Int
        var col: Int
        row = 0
        while (row < nOfRows) {
            col = 0
            while (col < nOfColumns) {
                if (starMatrix[row + nOfRows * col]) {
                    assignment[row] = col
                    break
                }
                col++
            }
            row++
        }
    }

    fun computeassignmentcost(assignment: IntArray, cost: Double, distMatrix: DoubleArray, nOfRows: Int) {
//        System.out.println("computeassignmentcost");
        var cost = cost
        var row: Int
        var col: Int
        row = 0
        while (row < nOfRows) {
            col = assignment[row]
            if (col >= 0) cost += distMatrix[row + nOfRows * col]
            row++
        }
    }

    fun step2a(
        assignment: IntArray,
        distMatrix: DoubleArray,
        starMatrix: BooleanArray,
        newStarMatrix: BooleanArray,
        primeMatrix: BooleanArray,
        coveredColumns: BooleanArray,
        coveredRows: BooleanArray,
        nOfRows: Int,
        nOfColumns: Int,
        minDim: Int
    ) {
//        System.out.println("step2a");
        var columnEnd: Int
        var col: Int
        /* cover every column containing a starred zero */col = 0
        while (col < nOfColumns) {
            var currentIndex = nOfRows * col
            columnEnd = nOfRows * col + nOfRows
            while (currentIndex < columnEnd) {
                if (starMatrix[currentIndex]) {
                    coveredColumns[col] = true
                    break
                }
                currentIndex++
            }
            col++
        }
        //        System.out.println("step2a1");
        /* move to step 3 */step2b(
            assignment,
            distMatrix,
            starMatrix,
            newStarMatrix,
            primeMatrix,
            coveredColumns,
            coveredRows,
            nOfRows,
            nOfColumns,
            minDim
        )
    }

    fun step2b(
        assignment: IntArray,
        distMatrix: DoubleArray,
        starMatrix: BooleanArray,
        newStarMatrix: BooleanArray,
        primeMatrix: BooleanArray,
        coveredColumns: BooleanArray,
        coveredRows: BooleanArray,
        nOfRows: Int,
        nOfColumns: Int,
        minDim: Int
    ) {
//        System.out.println("step2b");
        var col: Int
        var nOfCoveredColumns: Int

        /* count covered columns */nOfCoveredColumns = 0
        col = 0
        while (col < nOfColumns) {
            if (coveredColumns[col]) nOfCoveredColumns++
            col++
        }
        if (nOfCoveredColumns == minDim) {
            /* algorithm finished */
            buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns)
        } else {
            /* move to step 3 */
            step3(
                assignment,
                distMatrix,
                starMatrix,
                newStarMatrix,
                primeMatrix,
                coveredColumns,
                coveredRows,
                nOfRows,
                nOfColumns,
                minDim
            )
        }
    }

    fun step3(
        assignment: IntArray,
        distMatrix: DoubleArray,
        starMatrix: BooleanArray,
        newStarMatrix: BooleanArray,
        primeMatrix: BooleanArray,
        coveredColumns: BooleanArray,
        coveredRows: BooleanArray,
        nOfRows: Int,
        nOfColumns: Int,
        minDim: Int
    ) {
//        System.out.println("step3");
        var zerosFound: Boolean
        var row: Int
        var col: Int
        var starCol: Int
        zerosFound = true
        while (zerosFound) {
            zerosFound = false
            col = 0
            while (col < nOfColumns) {
                if (!coveredColumns[col]) {
                    row = 0
                    while (row < nOfRows) {
                        if (!coveredRows[row] && Math.abs(distMatrix[row + nOfRows * col]) < DBL_EPSILON) {
                            /* prime zero */
                            primeMatrix[row + nOfRows * col] = true

                            /* find starred zero in current row */starCol = 0
                            while (starCol < nOfColumns) {
                                if (starMatrix[row + nOfRows * starCol]) break
                                starCol++
                            }
                            if (starCol == nOfColumns) /* no starred zero found */ {
                                /* move to step 4 */
                                step4(
                                    assignment,
                                    distMatrix,
                                    starMatrix,
                                    newStarMatrix,
                                    primeMatrix,
                                    coveredColumns,
                                    coveredRows,
                                    nOfRows,
                                    nOfColumns,
                                    minDim,
                                    row,
                                    col
                                )
                                return
                            } else {
                                coveredRows[row] = true
                                coveredColumns[starCol] = false
                                zerosFound = true
                                break
                            }
                        }
                        row++
                    }
                }
                col++
            }
        }
        /* move to step 5 */step5(
            assignment,
            distMatrix,
            starMatrix,
            newStarMatrix,
            primeMatrix,
            coveredColumns,
            coveredRows,
            nOfRows,
            nOfColumns,
            minDim
        )
    }

    fun step4(
        assignment: IntArray,
        distMatrix: DoubleArray,
        starMatrix: BooleanArray,
        newStarMatrix: BooleanArray,
        primeMatrix: BooleanArray,
        coveredColumns: BooleanArray,
        coveredRows: BooleanArray,
        nOfRows: Int,
        nOfColumns: Int,
        minDim: Int,
        row: Int,
        col: Int
    ) {
//        System.out.println("step4");
        var n: Int
        var starRow: Int
        var starCol: Int
        var primeRow: Int
        var primeCol: Int
        val nOfElements = nOfRows * nOfColumns

        /* generate temporary copy of starMatrix */n = 0
        while (n < nOfElements) {
            newStarMatrix[n] = starMatrix[n]
            n++
        }

        /* star current zero */newStarMatrix[row + nOfRows * col] = true

        /* find starred zero in current column */starCol = col
        starRow = 0
        while (starRow < nOfRows) {
            if (starMatrix[starRow + nOfRows * starCol]) break
            starRow++
        }
        while (starRow < nOfRows) {
            /* unstar the starred zero */
            newStarMatrix[starRow + nOfRows * starCol] = false

            /* find primed zero in current row */primeRow = starRow
            primeCol = 0
            while (primeCol < nOfColumns) {
                if (primeMatrix[primeRow + nOfRows * primeCol]) break
                primeCol++
            }

            /* star the primed zero */newStarMatrix[primeRow + nOfRows * primeCol] = true

            /* find starred zero in current column */starCol = primeCol
            starRow = 0
            while (starRow < nOfRows) {
                if (starMatrix[starRow + nOfRows * starCol]) break
                starRow++
            }
        }

        /* use temporary copy as new starMatrix */
        /* delete all primes, uncover all rows */n = 0
        while (n < nOfElements) {
            primeMatrix[n] = false
            starMatrix[n] = newStarMatrix[n]
            n++
        }
        n = 0
        while (n < nOfRows) {
            coveredRows[n] = false
            n++
        }

        /* move to step 2a */step2a(
            assignment,
            distMatrix,
            starMatrix,
            newStarMatrix,
            primeMatrix,
            coveredColumns,
            coveredRows,
            nOfRows,
            nOfColumns,
            minDim
        )
    }

    fun step5(
        assignment: IntArray,
        distMatrix: DoubleArray,
        starMatrix: BooleanArray,
        newStarMatrix: BooleanArray,
        primeMatrix: BooleanArray,
        coveredColumns: BooleanArray,
        coveredRows: BooleanArray,
        nOfRows: Int,
        nOfColumns: Int,
        minDim: Int
    ) {
//        System.out.println("step5");
        var h: Double
        var value: Double
        var row: Int
        var col: Int
        /* find smallest uncovered element h */h = 1.79769313486231570814527423731704357e+308
        row = 0
        while (row < nOfRows) {
            if (!coveredRows[row]) {
                col = 0
                while (col < nOfColumns) {
                    if (!coveredColumns[col]) {
                        value = distMatrix[row + nOfRows * col]
                        if (value < h) h = value
                    }
                    col++
                }
            }
            row++
        }

        /* add h to each covered row */row = 0
        while (row < nOfRows) {
            if (coveredRows[row]) {
                col = 0
                while (col < nOfColumns) {
                    distMatrix[row + nOfRows * col] += h
                    col++
                }
            }
            row++
        }

        /* subtract h from each uncovered column */col = 0
        while (col < nOfColumns) {
            if (!coveredColumns[col]) {
                row = 0
                while (row < nOfRows) {
                    distMatrix[row + nOfRows * col] -= h
                    row++
                }
            }
            col++
        }
        /* move to step 3 */step3(
            assignment,
            distMatrix,
            starMatrix,
            newStarMatrix,
            primeMatrix,
            coveredColumns,
            coveredRows,
            nOfRows,
            nOfColumns,
            minDim
        )
    }

    companion object {
        private const val DBL_EPSILON = 2.22044604925031308084726333618164062e-16
    }
}
