package com.example.productsrecognition

import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import org.opencv.android.*
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.core.*
import org.opencv.features2d.DescriptorExtractor
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.FeatureDetector
import org.opencv.features2d.Features2d
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.util.*

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private var w: Int = 0
    private var h: Int = 0
    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private var RED = Scalar(255.0, 0.0, 0.0)
    private var GREEN = Scalar(0.0, 255.0, 0.0)
    private lateinit var detector: FeatureDetector
    private lateinit var descriptor: DescriptorExtractor
    private lateinit var matcher: DescriptorMatcher
    private lateinit var descriptors2: Mat
    private lateinit var descriptors1: Mat
    private lateinit var img1: Mat
    private lateinit var keypoints1: MatOfKeyPoint
    private lateinit var keypoints2: MatOfKeyPoint

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView.enableView()
                    try {
                        initializeOpenCVDependencies()
                    } catch (e: IOException) {
                        e.printStackTrace()
                    }
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun initializeOpenCVDependencies() {
        mOpenCvCameraView.enableView()
        detector = FeatureDetector.create(FeatureDetector.ORB)
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB)
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        img1 = Mat()
        val assetManager = assets
        val istr = assetManager.open("a.jpeg")
        val bitmap = BitmapFactory.decodeStream(istr)
        Utils.bitmapToMat(bitmap, img1)
        Imgproc.cvtColor(img1, img1, Imgproc.COLOR_RGB2GRAY)
        img1.convertTo(img1, 0) //converting the image to match with the type of the cameras image
        descriptors1 = Mat()
        keypoints1 = MatOfKeyPoint()
        detector.detect(img1, keypoints1)
        descriptor.compute(img1, keypoints1, descriptors1)
    }

    init {
        Log.i(TAG, "Instantiated new " + this.javaClass)
    }

    public override fun onCreate(savedInstanceState: Bundle?) {

        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        mOpenCvCameraView.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)

    }

    public override fun onPause() {
        super.onPause()
            mOpenCvCameraView.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        w = width
        h = height
    }

    override fun onCameraViewStopped() {}

    private fun recognize(aInputFrame: Mat): Mat {

        Imgproc.cvtColor(aInputFrame, aInputFrame, Imgproc.COLOR_RGB2GRAY)
        descriptors2 = Mat()
        keypoints2 = MatOfKeyPoint()
        detector.detect(aInputFrame, keypoints2)
        descriptor.compute(aInputFrame, keypoints2, descriptors2)

        // Matching
        val matches = MatOfDMatch()
        if (img1.type() == aInputFrame.type()) {
            matcher.match(descriptors1, descriptors2, matches)
        } else {
            return aInputFrame
        }
        val matchesList = matches.toList()

        var max_dist = 0.0
        var min_dist = 100.0

        for (i in matchesList.indices) {
            val dist = matchesList[i].distance.toDouble()
            if (dist < min_dist)
                min_dist = dist
            if (dist > max_dist)
                max_dist = dist
        }

        val good_matches = LinkedList<DMatch>()
        for (i in matchesList.indices) {
            if (matchesList[i].distance <= 1.5 * min_dist)
                good_matches.addLast(matchesList[i])
        }

        val goodMatches = MatOfDMatch()
        goodMatches.fromList(good_matches)
        val outputImg = Mat()
        val drawnMatches = MatOfByte()
        if (aInputFrame.empty() || aInputFrame.cols() < 1 || aInputFrame.rows() < 1) {
            return aInputFrame
        }
        Features2d.drawMatches(img1, keypoints1, aInputFrame, keypoints2, goodMatches, outputImg, GREEN, RED, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS)
        Imgproc.resize(outputImg, outputImg, aInputFrame.size())

        return outputImg
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        return recognize(inputFrame.rgba())

    }

    companion object {

        private val TAG = "OCVSample::Activity"

        init {
            if (!OpenCVLoader.initDebug())
                Log.d("ERROR", "Unable to load OpenCV")
            else
                Log.d("SUCCESS", "OpenCV loaded")
        }
    }
}