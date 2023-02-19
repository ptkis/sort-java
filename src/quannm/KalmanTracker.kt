package quannm

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.video.KalmanFilter
import java.util.concurrent.atomic.AtomicInteger

private val kf_count = AtomicInteger()
class KalmanTracker(initRect: Rect) {
    var m_time_since_update: Int
    var m_hits: Int
    var m_hit_streak: Int
    var m_age: Int
    var m_id: Int
    private var kf = KalmanFilter()
    private val measurement: Mat? = null
    private val m_history = ArrayList<Rect>()

    init {
        init_kf(initRect)
        m_time_since_update = 0
        m_hits = 0
        m_hit_streak = 0
        m_age = 0
        m_id = kf_count.getAndIncrement()
    }

    private fun init_kf(stateMat: Rect) {
        val stateNum = 7
        val measureNum = 4
        kf = KalmanFilter(stateNum, measureNum, 0, CvType.CV_32F)
        val transitionMatrix = Mat(7, 7, CvType.CV_32F, Scalar(0.0))
        val tM = floatArrayOf(
            1f, 0f, 0f, 0f, 1f, 0f, 0f,
            0f, 1f, 0f, 0f, 0f, 1f, 0f,
            0f, 0f, 1f, 0f, 0f, 0f, 1f,
            0f, 0f, 0f, 1f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 1f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 1f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f, 1f
        )
        transitionMatrix.put(0, 0, tM)
        kf._transitionMatrix = transitionMatrix
        kf._measurementMatrix = Mat.eye(4, 7, CvType.CV_32F)
        var processNoiseCov = Mat.eye(7, 7, CvType.CV_32F)
        processNoiseCov = processNoiseCov.mul(processNoiseCov, 1e-2)
        kf._processNoiseCov = processNoiseCov
        var id1 = Mat.eye(4, 4, CvType.CV_32F)
        id1 = id1.mul(id1, 1e-1)
        kf._measurementNoiseCov = id1
        val id2 = Mat.eye(7, 7, CvType.CV_32F)
        //id2 = id2.mul(id2,0.1);
        kf._errorCovPost = id2

//        System.out.println("stateMat "+stateMat);
//        System.out.println("stateMatx "+stateMat.x);
//        System.out.println("stateMaty "+stateMat.y);
//        System.out.println("stateMatwidth "+stateMat.width);
//        System.out.println("stateMatheight "+stateMat.height);
//        System.out.println("stateMatarea "+stateMat.area());
//        System.out.println("stateMatratio "+(float)stateMat.width / (float) stateMat.height);
        val statePost = Mat(7, 1, CvType.CV_32F, Scalar(0.0))
        statePost.put(0, 0, (stateMat.x + stateMat.width / 2).toDouble())
        statePost.put(1, 0, (stateMat.y + stateMat.height / 2).toDouble())
        statePost.put(2, 0, stateMat.area())
        statePost.put(3, 0, (stateMat.width.toFloat() / stateMat.height.toFloat()).toDouble())
        kf._statePost = statePost

//        System.out.println("transitionmat "+ kf.get_transitionMatrix().size());
//        System.out.println("transitionmat\n "+ kf.get_transitionMatrix().dump());
//        System.out.println("measurementMatrix "+ kf.get_measurementMatrix().size());
//        System.out.println("measurementMatrix\n "+ kf.get_measurementMatrix().dump());
//        System.out.println("processNoiseCov "+ kf.get_processNoiseCov().size());
//        System.out.println("processNoiseCov\n "+ kf.get_processNoiseCov().dump());
//        System.out.println("measurementNoiseCov "+ kf.get_measurementNoiseCov().size());
//        System.out.println("measurementNoiseCov\n "+ kf.get_measurementNoiseCov().dump());
//        System.out.println("errorCovPost "+ kf.get_errorCovPost().size());
//        System.out.println("errorCovPost\n "+ kf.get_errorCovPost().dump());
//        System.out.println("statePre "+ kf.get_statePre().size());
//        System.out.println("statePre\n "+ kf.get_statePre().dump());
//        System.out.println("statePost "+ kf.get_statePost().size());
//        System.out.println("statePost\n "+ kf.get_statePost().dump());
    }

    fun predict(): Rect {
        val p = kf.predict()
        //        System.out.println(" p "+ p.dump());
        m_age += 1
        if (m_time_since_update > 0) m_hit_streak = 0
        m_time_since_update += 1
        val predictBox =
            get_rect_xysr(p[0, 0][0].toFloat(), p[1, 0][0].toFloat(), p[2, 0][0].toFloat(), p[3, 0][0].toFloat())
        m_history.add(predictBox)
        return m_history[m_history.size - 1]
    }

    fun update(stateMat: Rect) {
        m_time_since_update = 0
        m_history.clear()
        m_hits += 1
        m_hit_streak += 1

        //measurement
//        float[][] floatMeasurementArray = new float[][]{{stateMat.x+stateMat.width/2},{stateMat.y + stateMat.height/2},{(int) stateMat.area()},{(float)stateMat.width/(float)stateMat.height}};
//        measurement = setMatrix(4,0,new Mat(4,0,CvType.CV_32F),floatMeasurementArray);
        val measurement = Mat(4, 1, CvType.CV_32F, Scalar(0.0))
        measurement.put(0, 0, (stateMat.x + stateMat.width / 2).toDouble())
        measurement.put(1, 0, (stateMat.y + stateMat.height / 2).toDouble())
        measurement.put(2, 0, stateMat.area())
        measurement.put(3, 0, (stateMat.width.toFloat() / stateMat.height.toFloat()).toDouble())
        // update
        kf.correct(measurement)
    }

    fun get_state(): Rect {
        val s = kf._statePost
        return get_rect_xysr(s[0, 0][0].toFloat(), s[1, 0][0].toFloat(), s[2, 0][0].toFloat(), s[3, 0][0].toFloat())
    }

    private fun setMatrix(rowNum: Int, colNum: Int, tempMatrix: Mat, intArray: Array<FloatArray>): Mat {
        for (row in 0 until rowNum) {
            for (col in 0 until colNum) tempMatrix.put(row, col, intArray[row][col].toDouble())
        }
        return tempMatrix
    }

    private fun get_rect_xysr(cx: Float, cy: Float, s: Float, r: Float): Rect {
        val w = Math.sqrt((s * r).toDouble()).toInt()
        val h = (s / w).toInt()
        var x = (cx - w / 2).toInt()
        var y = (cy - h / 2).toInt()
        if (x < 0 && cx > 0) x = 0
        if (y < 0 && cy > 0) y = 0
        return Rect(x, y, w, h)
    }
}
