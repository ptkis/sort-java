package quannm

import nu.pattern.OpenCV
import org.apache.commons.io.FileUtils
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Color
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.util.*
import java.util.concurrent.atomic.AtomicInteger

//import com.google.common.collect.Sets;
object Main {
    var total_frames = 0
    var total_time = 0.0
    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        OpenCV.loadLocally()
        val Sequences = ArrayList(
            mutableListOf(
                "PETS09-S2L1",
                "TUD-Campus",
                "TUD-Stadtmitte",
                "ETH-Bahnhof",
                "ETH-Sunnyday",
                "ETH-Pedcross2",
                "KITTI-13",
                "KITTI-17",
                "ADL-Rundle-6",
                "ADL-Rundle-8",
                "Venice-2"
            )
        )
        // Loop through all datasets
        for (sequence in Sequences) {
            TestSORT(sequence, true) // display variable
        }
    }

    @Throws(IOException::class)
    fun TestSORT(seqName: String, display: Boolean) {
        var display1 = display
        var currTrackerIndex: Int
        val randColor: MutableList<Color> = ArrayList<Color>()
        var r: Float
        var g: Float
        var b: Float
        var randomColor: Color
        val rand = Random()
        for (i in 0..19) {
            r = rand.nextFloat()
            g = rand.nextFloat()
            b = rand.nextFloat()
            randomColor = Color(r, g, b)
            randColor.add(randomColor)
        }
        println("Processing $seqName ....")
        val imgPath = "src/mot_benchmark/train/$seqName/img1/"
        if (display1) {
            if (!Files.exists(Path.of(imgPath))) {
                println("Image path not found")
                display1 = false
            }
        }
        // 1 . Read detection file
        val detFileName = "src/data/$seqName/det.txt"
        val detFile = File(detFileName)
        var detLine: String
        val detData = ArrayList<TrackingBox>()
        var ch: Char
        var tpx: Float
        var tpy: Float
        var tpw: Float
        var tph: Float
        FileUtils.lineIterator(detFile, "UTF-8").use { it ->
            while (it.hasNext()) {
                val tb = TrackingBox()
                val line = it.nextLine()
                val values = line.split(",".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                tb.frame = values[0].toInt()
                tb.id = values[1].toInt()
                tb.box = Rect(
                    values[2].toFloat().toInt(),
                    values[3].toFloat().toInt(),
                    values[4].toFloat().toInt(),
                    values[5].toFloat()
                        .toInt()
                )
                detData.add(tb)
            }
        }
        // 2. group detData by frame
        var maxFrame = 0
        for (tb in detData) {
            if (maxFrame < tb.frame) maxFrame = tb.frame
        }
        val detFrameData = ArrayList<ArrayList<TrackingBox>>()
        val tempVec = ArrayList<TrackingBox>()
        for (fi in 0 until maxFrame) {
            for (tb in detData) {
                if (tb.frame == fi + 1) {
                    tempVec.add(tb)
                }
            }
            detFrameData.add(tempVec.clone() as ArrayList<TrackingBox>)
            tempVec.clear()
        }

        // 3. update across frames
        var frame_count = 0
        val max_age = 1
        val min_hits = 3
        val iouThreshold = 0.3
        val trackers = ArrayList<KalmanTracker>()
        // variables used in the for-loop
        val predictedBoxes = ArrayList<Rect>()
        var iouMatrix: Mat
        val assignment = ArrayList<Int>()
        var unmatchedDetections: MutableSet<Int?> = HashSet()
        val unmatchedTrajectories: MutableSet<Int> = HashSet()
        val allItems: MutableSet<Int?> = HashSet()
        val matchedItems: MutableSet<Int?> = HashSet()
        val matchedPairs = ArrayList<Point>()
        val frameTrackingResult = ArrayList<TrackingBox>()
        var trkNum = 0
        var detNum = 0
        var cycle_time = 0.0
        val start_time: Long = 0
        val fstream = FileWriter("src/output/$seqName.txt", false)

        val kfCount = AtomicInteger()
        // main loop
        for (fi in 0 until maxFrame) {
            total_frames++
            frame_count++

//            start_time = getTickCount();
            if (trackers.size == 0) // the first frame met
            {
                // initialize kalman trackers using first detections.
                for (i in detFrameData[fi].indices) {
                    val trk = KalmanTracker(detFrameData[fi][i].box!!, kfCount)
                    trackers.add(trk)
                }

                // output the first frame detections
                for (id in detFrameData[fi].indices) {
                    val tb = detFrameData[fi][id]
                    fstream.write(
                        """
    ${tb.frame},${id + 1},${tb.box?.x},${tb.box?.y},${tb.box?.width},${tb.box?.height},1,-1,-1,-1
    
    """.trimIndent()
                    )
                }
                continue
            }

            // 3.1. get predicted locations from existing trackers.
            predictedBoxes.clear()

//            System.out.println("trackers size "+trackers.size());
            currTrackerIndex = 0
            while (currTrackerIndex < trackers.size) {
                val pBox = trackers[currTrackerIndex].predict()
                //                try
//                {
//                    Thread.sleep(1000); // Sleep for one second
//                }
//                catch (InterruptedException e)
//                {
//                    Thread.currentThread().interrupt();
//                }
//                System.out.println("current index "+currTrackerIndex);
//                System.out.println("predicted box "+pBox);
                if (pBox.x >= 0 && pBox.y >= 0) {
                    predictedBoxes.add(pBox)
                    currTrackerIndex++
                } else {
                    trackers.removeAt(currTrackerIndex)
                }
            }

            // 3.2. associate detections to tracked object (both represented as bounding boxes)
            // dets : detFrameData[fi]
            trkNum = predictedBoxes.size
            detNum = detFrameData[fi].size
            iouMatrix = Mat(trkNum, detNum, CvType.CV_64F)
            for (i in 0 until trkNum)  // compute iou matrix as a distance matrix
            {
                for (j in 0 until detNum) {
                    // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                    iouMatrix.put(i, j, 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box!!))
                }
            }

            // solve the assignment problem using hungarian algorithm.
            // the resulting assignment is [track(prediction) : detection], with len=preNum
            val HungAlgo = HungarianAlgorithm()
            assignment.clear()
            HungAlgo.Solve(iouMatrix, assignment)

            // find matches, unmatched_detections and unmatched_predictions
            unmatchedTrajectories.clear()

//            System.out.println("unmatched det "+ unmatchedDetections);

//            System.out.println("unmatched det "+ unmatchedDetections.size());
            unmatchedDetections.clear()
            allItems.clear()
            matchedItems.clear()
            if (detNum > trkNum) //	there are unmatched detections
            {
                for (n in 0 until detNum) allItems.add(n)
                for (i in 0 until trkNum) matchedItems.add(assignment[i])
                unmatchedDetections = HashSet(allItems)
                unmatchedDetections.removeAll(matchedItems)
            } else if (detNum < trkNum) // there are unmatched trajectory/predictions
            {
                for (i in 0 until trkNum) if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.add(i)
            }


            // filter out matched with low IOU
            matchedPairs.clear()
            for (i in 0 until trkNum) {
                if (assignment[i] == -1) // pass over invalid values
                    continue
                if (1.0 - iouMatrix[i, assignment[i]][0] < iouThreshold) {
                    unmatchedTrajectories.add(i)
                    unmatchedDetections.add(assignment[i])
                } else matchedPairs.add(Point(i.toDouble(), assignment[i].toDouble()))
            }
            ///////////////////////////////////////
            // 3.3. updating trackers

            // update matched trackers with assigned detections.
            // each prediction is corresponding to a tracker
//            System.out.println("3.3");
            var detIdx: Int
            var trkIdx: Int
            for (i in matchedPairs.indices) {
                trkIdx = matchedPairs[i].x.toInt()
                detIdx = matchedPairs[i].y.toInt()
                trackers[trkIdx].update(detFrameData[fi][detIdx].box!!)
            }

            // create and initialise new trackers for unmatched detections
            for (umd in unmatchedDetections) {
//                System.out.println(detFrameData.get(fi).get(umd).getBox());
                val tracker = KalmanTracker(detFrameData[fi][umd!!].box!!, kfCount)
                trackers.add(tracker)
            }
            // get trackers' output
            frameTrackingResult.clear()
            currTrackerIndex = 0
            while (currTrackerIndex < trackers.size) {
                if (trackers[currTrackerIndex].m_time_since_update < 1 && (trackers[currTrackerIndex].m_hit_streak >= min_hits || frame_count <= min_hits)) {
                    val res = TrackingBox()
                    res.box = trackers[currTrackerIndex].get_state()
                    res.id = trackers[currTrackerIndex].m_id + 1
                    res.frame = frame_count
                    frameTrackingResult.add(res)
                }
                currTrackerIndex++
                // remove dead tracklet
                if (currTrackerIndex != trackers.size && trackers[currTrackerIndex].m_time_since_update > max_age) trackers.removeAt(
                    currTrackerIndex
                )
            }
            cycle_time = (Core.getTickCount() - start_time).toDouble()
            total_time += cycle_time / Core.getTickFrequency()
            for (tb in frameTrackingResult) {
                fstream.write(
                    """
    ${tb.frame},${tb.id},${tb.box?.x},${tb.box?.y},${tb.box?.width},${tb.box?.height},1,-1,-1,-1
    
    """.trimIndent()
                )
            }
            if (display1) {
                val oss = imgPath + String.format("%06d", fi + 1) + ".jpg"
                println(oss)
                val img = Imgcodecs.imread(oss)
                if (img.empty()) continue
                for (tb in frameTrackingResult) {
                    val color = randColor[tb.id % 20]
                    //                    System.out.println(color);
                    val opencv_color = Scalar(color.blue.toDouble(), color.green.toDouble(), color.red.toDouble())
                    Imgproc.rectangle(img, tb.box, opencv_color, 2, 8, 0)
                    Imgproc.putText(img, tb.id.toString(), Point(tb.box!!.x.toDouble(), tb.box!!.y.toDouble()), Imgproc.FONT_HERSHEY_SIMPLEX,1.0, opencv_color)
                }
                HighGui.imshow(seqName, img)
                HighGui.waitKey(40)
            }
        }
        fstream.close()
        if (display1) HighGui.destroyAllWindows()
    }

    fun GetIOU(bb_test: Rect, bb_gt: Rect): Double {
        val DBL_EPSILON = 2.22044604925031308084726333618164062e-16
        val intersectionMinX = Math.max(bb_test.x, bb_gt.x).toFloat()
        val intersectionMinY = Math.max(bb_test.y, bb_gt.y).toFloat()
        val intersectionMaxX = Math.min(bb_test.x + bb_test.width, bb_gt.x + bb_gt.width).toFloat()
        val intersectionMaxY = Math.min(bb_test.y + bb_test.height, bb_gt.y + bb_gt.height).toFloat()
        val `in` = Math.max(intersectionMaxY - intersectionMinY, 0f) *
                Math.max(intersectionMaxX - intersectionMinX, 0f)
        val un = (bb_test.area() + bb_gt.area() - `in`).toFloat()
        return if (un < DBL_EPSILON) 0.0 else (`in` / un).toDouble()
    }
}
