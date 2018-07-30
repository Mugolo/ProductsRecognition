package com.example.productsrecognition

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.support.v4.content.ContextCompat
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.View
import android.view.WindowManager
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.features2d.DescriptorExtractor
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.FeatureDetector
import org.opencv.features2d.Features2d
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.util.*


class MainActivity : AppCompatActivity(), /*CameraBridgeViewBase.CvCameraViewListener2,*/ View.OnClickListener {

    private var w: Int = 0
    private var h: Int = 0
    private var RED = Scalar(255.0, 0.0, 0.0)
    private var GREEN = Scalar(0.0, 255.0, 0.0)
    private lateinit var detector: FeatureDetector
    private lateinit var descriptor: DescriptorExtractor
    private lateinit var matcher: DescriptorMatcher
    private lateinit var descriptors2: Mat
    private lateinit var descriptors1: Mat
    private lateinit var img1: Mat
    private lateinit var sampleBitmap: Bitmap
    private lateinit var keypoints1: MatOfKeyPoint
    private lateinit var keypoints2: MatOfKeyPoint

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.e(TAG, "OpenCV loaded successfully")
//                    mOpenCvCameraView.enableView()
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
//        mOpenCvCameraView.enableView()
        detector = FeatureDetector.create(FeatureDetector.ORB)
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB)
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        img1 = Mat()
        val istr = assets.open("a.jpg")
        sampleBitmap = BitmapFactory.decodeStream(istr)
        Utils.bitmapToMat(sampleBitmap, img1)
        Imgproc.cvtColor(img1, img1, Imgproc.COLOR_RGB2GRAY)
        img1.convertTo(img1, 0) //converting the image to match with the type of the cameras image
        descriptors1 = Mat()
        keypoints1 = MatOfKeyPoint()
        detector.detect(img1, keypoints1)
        descriptor.compute(img1, keypoints1, descriptors1)
    }

    init {
        Log.e(TAG, "Instantiated new " + this.javaClass)
    }

    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        requestPermissions()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback)
        } else {
            Log.e(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

//        mOpenCvCameraView.setCvCameraViewListener(this)
        pictureTaken.setOnClickListener(this)

    }

    private fun requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(Array(1) { Manifest.permission.CAMERA }, 1234)
            }
        }
    }

    public override fun onPause() {
        super.onPause()
//        mOpenCvCameraView.disableView()
    }

    public override fun onResume() {
        super.onResume()

    }


    override fun onClick(p0: View?) {
        when (p0) {
            pictureTaken -> {
                val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                if (takePictureIntent.resolveActivity(packageManager) != null) {
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode === REQUEST_IMAGE_CAPTURE && resultCode === Activity.RESULT_OK) {
            val extras = data?.extras
            val imageBitmap = extras?.get("data") as Bitmap
            val inputFrame = Mat()
//            Utils.bitmapToMat(imageBitmap, inputFrame)
//            Utils.matToBitmap(recognize(inputFrame), imageBitmap)
//            pictureTaken.setImageBitmap(imageBitmap)
            compareBitmaps(sampleBitmap, imageBitmap)

        }
    }

    private fun compareBitmaps(bitmap1: Bitmap, bitmap2: Bitmap) {
        val mat1 = Mat(bitmap1.width, bitmap1.height, CvType.CV_8UC3)
        Utils.bitmapToMat(bitmap1, mat1)

        val mat2 = Mat(bitmap2.width, bitmap2.height, CvType.CV_8UC3)
        Utils.bitmapToMat(bitmap2, mat2)

        compareMats(mat1, mat2)

    }

    private fun compareMats(img1: Mat, img2: Mat): Unit {
        val detector = FeatureDetector.create(FeatureDetector.ORB)
        val extractor = DescriptorExtractor.create(DescriptorExtractor.ORB)
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)

        val descriptors1 = Mat()
        val keypoints1 = MatOfKeyPoint()
        detector.detect(img1, keypoints1)
        extractor.compute(img1, keypoints1, descriptors1)

        //second image
        // Mat img2 = Imgcodecs.imread(path2);
        val descriptors2 = Mat()
        val keypoints2 = MatOfKeyPoint()
        detector.detect(img2, keypoints2)
        extractor.compute(img2, keypoints2, descriptors2)


        //matcher image descriptors
        val matches = MatOfDMatch()
        matcher.matchEdited(descriptors1, descriptors2, matches)

        // Filter matches by distance
        val filtered = filterMatchesByDistance(matches)

        val total: Int = matches.size().height.toInt()
        val match: Int = filtered.size().height.toInt()
        Log.d("LOG", "total:" + total + " Match:" + match)
    }

    private fun filterMatchesByDistance(matches: MatOfDMatch): MatOfDMatch {
        var matches_original: List<DMatch> = matches.toList()
        var matches_filtered = ArrayList<DMatch>()

        val DIST_LIMIT = 30;
        // Check all the matches distance and if it passes add to list of filtered matches
        Log.d("DISTFILTER", "ORG SIZE:" + matches_original.size + "")
        for (d: DMatch in matches_original) {
            if (Math.abs(d.distance) <= DIST_LIMIT) {
                matches_filtered.add(d)
            }
        }
        Log.d("DISTFILTER", "FIL SIZE:" + matches_filtered.size + "")

        val mat: MatOfDMatch = MatOfDMatch()
        mat.fromList(matches_filtered)
        return mat
    }

//    override fun onCameraViewStarted(width: Int, height: Int) {
//        w = width
//        h = height
//    }
//
//    override fun onCameraViewStopped() {
//        Log.e(TAG, "onCameraViewStopped")
//    }

    private fun recognize(aInputFrame: Mat): Mat {

        Imgproc.cvtColor(aInputFrame, aInputFrame, Imgproc.COLOR_RGB2GRAY)
        descriptors2 = Mat()
        keypoints2 = MatOfKeyPoint()
        detector.detect(aInputFrame, keypoints2)
        descriptor.compute(aInputFrame, keypoints2, descriptors2)

        // Matching
        val matches = MatOfDMatch()
        if (img1.type() == aInputFrame.type()) {
            matcher.matchEdited(descriptors1, descriptors2, matches)
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

//    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
//        return recognize(inputFrame.rgba())
//    }

    companion object {

        private val TAG = "OCVSample::Activity"
        private val REQUEST_IMAGE_CAPTURE: Int = 101

        init {
            if (!OpenCVLoader.initDebug())
                Log.e("ERROR", "Unable to load OpenCV")
            else
                Log.e("SUCCESS", "OpenCV loaded")
        }
    }
}