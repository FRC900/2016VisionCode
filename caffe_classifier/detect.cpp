#include <iostream>
#include "CaffeBatchPrediction.hpp"
#include "scalefactor.hpp"
#include "fast_nms.hpp"
#include "detect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

static double gtod_wrapper(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


// TODO :: Make a call for GPU Mat input.
// Simple multi-scale detect.  Take a single image, scale it into a number
// of diffent sized images. Run a fixed-size detection window across each
// of them.  Keep track of the scale of each scaled image to map the
// detected rectangles back to the correct location and size on the
// original input images
template<class MatT>
void NNDetect<MatT>::detectMultiscale(const Mat&            inputImg,
                                      const Mat&            depthMat,
                                      const Size&           minSize,
                                      const Size&           maxSize,
                                      double                scaleFactor,
                                      const vector<double>& nmsThreshold,
                                      const vector<double>& detectThreshold,
                                      vector<Rect>&         rectsOut)
{
    // Size of the first level classifier. Others are an integer multiple
    // of this initial size (2x and maybe 4x if we need it)
    int wsize = d12_.getInputGeometry().width;

    // scaled images which allow us to find images between min and max
    // size for the given classifier input window
    vector<pair<MatT, double> > scaledImages12;
    vector<pair<MatT, double> > scaledImages24;
    // Maybe later ? vector<pair<MatT, double> > scaledImages48;

    // list of windows to work with.
    // A window is a rectangle from a given scaled image along
    // with the index of the scaled image it corresponds with.
    vector<Window> windowsIn;
    vector<Window> windowsOut;

    // Confidence scores (0.0 - 1.0) for each detected rectangle
    vector<float> scores;

    // Generate a list of initial windows to search. Each window will be 12x12 from a scaled image
    // These scaled images let us search for variable sized objects using a fixed-width detector
    MatT f32Img;

    MatT(inputImg).convertTo(f32Img, CV_32FC3);     // classifier runs on float pixel data
    generateInitialWindows(f32Img, depthMat, minSize, maxSize, wsize, scaleFactor, scaledImages12, windowsIn);

    // Generate scaled images for the larger detect sizes as well. Subsequent passes will use larger
    // input sizes. These images will let us grab higher res input as the detector size goes up (as
    // opposed to just scaling up the 12x12 images to a larger size).
    scalefactor(f32Img, Size(wsize * 2, wsize * 2), minSize, maxSize, scaleFactor, scaledImages24);
    // not yet - scalefactor(f32Img, Size(wsize*4,wsize*4), minSize, maxSize, scaleFactor, scaledImages48);

    // Do 1st level of detection. This takes the initial list of windows
    // and returns the list which have a score for "ball" above the
    // threshold listed.
    cout << "d12 windows in = " << windowsIn.size() << endl;
    runDetection(d12_, scaledImages12, windowsIn, detectThreshold[0], "ball", windowsOut, scores);
    cout << "d12 windows out = " << windowsOut.size() << endl;
    runNMS(windowsOut, scores, scaledImages12, nmsThreshold[0], windowsIn);
    cout << "d12 nms windows out = " << windowsIn.size() << endl;

    // Double the size of the rects to get from a 12x12 to 24x24
    // detection window.  Use scaledImages24 for the detection call
    // since that has the scales appropriate for the 24x24 detector
    for (auto it = windowsIn.begin(); it != windowsIn.end(); ++it)
    {
        it->first = Rect(it->first.x * 2, it->first.y * 2,
                         it->first.width * 2, it->first.height * 2);
    }

    if ((detectThreshold.size() > 1) && (detectThreshold[1] > 0.0))
    {
        cout << "d24 windows in = " << windowsIn.size() << endl;
        runDetection(d24_, scaledImages24, windowsIn, detectThreshold[1], "ball", windowsOut, scores);
        cout << "d24 windows out = " << windowsOut.size() << endl;
        runNMS(windowsOut, scores, scaledImages24, nmsThreshold[1], windowsIn);
		cout << "d24 nms windows out = " << windowsIn.size() << endl;
    }

    // Final result - scale the output rectangles back to the
    // correct scale for the original sized image
    rectsOut.clear();
    for (auto it = windowsIn.cbegin(); it != windowsIn.cend(); ++it)
    {
        double   scale = scaledImages24[it->second].second;
        Rect rect(it->first);
        Rect scaledRect(Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale));
        rectsOut.push_back(scaledRect);
    }
}


template<class MatT>
void NNDetect<MatT>::runNMS(const vector<Window>& windows,
                            const vector<float>& scores,
                            const vector<pair<MatT, double> >& scaledImages,
                            double nmsThreshold,
                            vector<Window>& windowsOut)
{
    if ((nmsThreshold > 0.0) && (nmsThreshold < 1.0))
    {
        // Detected is a rect, score pair.
        vector<Detected> detected;

        // Need to scale each rect to the correct mapping to the
        // original image, since rectangles from multiple different
        // scales might overlap
        for (size_t i = 0; i < windows.size(); i++)
        {
            double scale = scaledImages[windows[i].second].second;
            Rect   rect(windows[i].first);
            Rect   scaledRect(Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale));
            detected.push_back(Detected(scaledRect, scores[i]));
        }

        vector<size_t> nmsOut;
        fastNMS(detected, nmsThreshold, nmsOut);
        // Each entry of nmsOut is the index of a saved rect/scales
        // pair.  Save the entries from those indexes as the output
        windowsOut.clear();
        for (auto it = nmsOut.cbegin(); it != nmsOut.cend(); ++it)
        {
            windowsOut.push_back(windows[*it]);
        }
    }
    else
    {
        // If not running NMS, output is the same as the input
        windowsOut = windows;
    }
}


template<class MatT>
void NNDetect<MatT>::generateInitialWindows(
    const MatT& input,
    const Mat&  depthIn,
    const Size& minSize,
    const Size& maxSize,
    int wsize,
    double scaleFactor,
    vector<pair<MatT, double> >& scaledImages,
    vector<Window>& windows)
{
    windows.clear();
    size_t windowsChecked = 0;

    // How many pixels to move the window for each step
    // We use 4 - the calibration step can adjust +/- 2 pixels
    // in each direction, which means they will correct for
    // anything which is actually centered in one of the
    // pixels we step over.
    const int step = 4;

    // Create array of scaled images
    vector<pair<MatT, double> > scaledDepth;
    if (!depthIn.empty())
    {
        MatT depthGpu = MatT(depthIn);
        scalefactor(depthGpu, Size(wsize, wsize), minSize, maxSize, scaleFactor, scaledDepth);
    }
    scalefactor(input, Size(wsize, wsize), minSize, maxSize, scaleFactor, scaledImages);
    // Main loop.  Look at each scaled image in turn
    for (size_t scale = 0; scale < scaledImages.size(); ++scale)
    {
		float depth_multiplier = 0.2;
		float ball_real_size = 247.6; // ball is 9.75in diameter = 247.6 mm
        float percent_image = (float)wsize / scaledImages[scale].first.cols;
		float size_fov = percent_image * hfov_; //TODO fov size
		float depth_avg = (ball_real_size / (2.0 * tanf(size_fov / 2.0))) - (4.572 * 25.4);
		
        float depth_min = depth_avg - depth_avg * depth_multiplier;
        float depth_max = depth_avg + depth_avg * depth_multiplier;
        cout << fixed << "Target size:" << wsize / scaledImages[scale].second << " Dist:" << depth_avg << " Min/max:" << depth_min << "/" << depth_max;
		size_t thisWindowsChecked = 0;
		size_t thisWindowsPassed  = 0;

        // Start at the upper left corner.  Loop through the rows and cols until
        // the detection window falls off the edges of the scaled image
        for (int r = 0; (r + wsize) < scaledImages[scale].first.rows; r += step)
        {
            for (int c = 0; (c + wsize) < scaledImages[scale].first.cols; c += step)
            {
				thisWindowsChecked += 1;
                if (!depthIn.empty())
                {
                    Mat detectCheck = Mat(scaledDepth[scale].first(Rect(c, r, wsize, wsize)));
                    if(!depthInRange(depth_min, depth_max, detectCheck))
                    {
                        continue;
                    }
                }
                windows.push_back(Window(Rect(c, r, wsize, wsize), scale));
				thisWindowsPassed += 1;
            }
        }
		windowsChecked += thisWindowsChecked;
		cout << " Windows Passed:"<< thisWindowsPassed << "/" << thisWindowsChecked << endl;
    }
    cout << "generateInitialWindows checked " << windowsChecked << " windows and passed " << windows.size() << endl;
}


template<class MatT>
void NNDetect<MatT>::runDetection(CaffeClassifier<MatT>& classifier,
                                  const vector<pair<MatT, double> >& scaledImages,
                                  const vector<Window>& windows,
                                  float threshold,
                                  string label,
                                  vector<Window>& windowsOut,
                                  vector<float>& scores)
{
    windowsOut.clear();
    scores.clear();
    // Accumulate a number of images to test and pass them in to
    // the NN prediction as a batch
    vector<MatT> images;

    // Return value from detection. This is a list of indexes from
    // the input which have a high enough confidence score
    vector<size_t> detected;

    size_t batchSize = classifier.BatchSize();
    int    counter   = 0;
    double start     = gtod_wrapper(); // grab start time

    // For each input window, grab the correct image
    // subset from the correct scaled image.
    // Detection happens in batches, so save up a list of
    // images and submit them all at once.
    for (auto it = windows.cbegin(); it != windows.cend(); ++it)
    {
        // scaledImages[x].first is a Mat holding the image
        // scaled to the correct size for the given rect.
        // it->second is the index into scaledImages to look at
        // so scaledImages[it->second] is a Mat at the correct
        // scale for the current window. it->first is the
        // rect describing the subset of that image we need to process
        images.push_back(scaledImages[it->second].first(it->first));
        if ((images.size() == batchSize) || ((it + 1) == windows.cend()))
        {
            doBatchPrediction(classifier, images, threshold, label, detected, scores);

            // Clear out images array to start the next batch
            // of processing fresh
            images.clear();

            for (size_t j = 0; j < detected.size(); j++)
            {
                // Indexes in detected array are relative to the start of the
                // current batch just passed in to doBatchPrediction.
                // Use counter to keep track of which batch we're in
                windowsOut.push_back(windows[counter * batchSize + detected[j]]);
            }
            // Keep track of the batch number
            counter++;
        }
    }
    double end = gtod_wrapper();
    cout << "runDetection time = " << (end - start) << endl;
}


// do 1 run of the classifier. This takes up batch_size predictions
// and adds the index of anything found to the detected list
template<class MatT>
void NNDetect<MatT>::doBatchPrediction(CaffeClassifier<MatT>&   classifier,
                                       const vector<MatT>& imgs,
                                       float                    threshold,
                                       const string&       label,
                                       vector<size_t>&     detected,
                                       vector<float>&      scores)
{
    detected.clear();
    // Grab the top 2 detected classes.  Since we're doing an object /
    // not object split, that will get everything
    vector<vector<Prediction> > predictions = classifier.ClassifyBatch(imgs, 2);

    // Each outer loop is the predictions for one input image
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        // Each inner loop is the prediction for a particular label
        // for the given image, sorted by score.
        //
        // Look for object with label <label>, > threshold confidence
        for (vector<Prediction>::const_iterator it = predictions[i].begin(); it != predictions[i].end(); ++it)
        {
            if (it->first == label)
            {
                if (it->second >= threshold)
                {
                    detected.push_back(i);
                    scores.push_back(it->second);
                }
                break;
            }
        }
    }
}

// Be conservative here - if any of the depth values in the target rect
// are in the expected range, consider the rect in range.  Also 
// say that it is in range if any of the depth values are negative (i.e. no
// depth info for those pixels)
template <class MatT>
bool NNDetect<MatT>::depthInRange(float depth_min, float depth_max, const Mat &detectCheck)
{
    for (int py = 0; py < detectCheck.rows; py++)
    {
        const float *p = detectCheck.ptr<float>(py);
        for (int px = 0; px < detectCheck.cols; px++)
        {
            if ((p[px] <= 0.0) || ((p[px] < depth_max) && (p[px] > depth_min)))
            {
                return true;
            }
        }
    }
    return false;
}

// Explicitly instatiate classes used elsewhere
template class NNDetect<Mat>;
template class NNDetect<gpu::GpuMat>;
