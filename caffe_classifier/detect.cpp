#include <iostream>
#include "CaffeBatchPrediction.hpp"
#include "scalefactor.hpp"
#include "fast_nms.hpp"
#include "detect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

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
void NNDetect<MatT>::detectMultiscale(const cv::Mat&             inputImg,
                                      const cv::Mat&             depthMat,
                                      const cv::Size&            minSize,
                                      const cv::Size&            maxSize,
                                      double                     scaleFactor,
                                      const std::vector<double>& nmsThreshold,
                                      const std::vector<double>& detectThreshold,
                                      std::vector<cv::Rect>&     rectsOut)
{
    // Size of the first level classifier. Others are an integer multiple
    // of this initial size (2x and maybe 4x if we need it)
    int wsize = d12_.getInputGeometry().width;

    // scaled images which allow us to find images between min and max
    // size for the given classifier input window
    std::vector<std::pair<MatT, double> > scaledImages12;
    std::vector<std::pair<MatT, double> > scaledImages24;
    // Maybe later ? std::vector<std::pair<MatT, double> > scaledImages48;

    // list of windows to work with.
    // A window is a rectangle from a given scaled image along
    // with the index of the scaled image it corresponds with.
    std::vector<Window> windowsIn;
    std::vector<Window> windowsOut;

    // Confidence scores (0.0 - 1.0) for each detected rectangle
    std::vector<float> scores;

    // Generate a list of initial windows to search. Each window will be 12x12 from a scaled image
    // These scaled images let us search for variable sized objects using a fixed-width detector
    MatT f32Img;

    MatT(inputImg).convertTo(f32Img, CV_32FC3);     // classifier runs on float pixel data
    generateInitialWindows(f32Img, depthMat, minSize, maxSize, wsize, scaleFactor, scaledImages12, windowsIn);

    // Generate scaled images for the larger detect sizes as well. Subsequent passes will use larger
    // input sizes. These images will let us grab higher res input as the detector size goes up (as
    // opposed to just scaling up the 12x12 images to a larger size).
    scalefactor(f32Img, cv::Size(wsize * 2, wsize * 2), minSize, maxSize, scaleFactor, scaledImages24);
    // not yet - scalefactor(f32Img, cv::Size(wsize*4,wsize*4), minSize, maxSize, scaleFactor, scaledImages48);

    // Do 1st level of detection. This takes the initial list of windows
    // and returns the list which have a score for "ball" above the
    // threshold listed.
    std::cout << "d12 windows in = " << windowsIn.size() << std::endl;
    runDetection(d12_, scaledImages12, windowsIn, detectThreshold[0], "ball", windowsOut, scores);
    std::cout << "d12 windows out = " << windowsOut.size() << std::endl;
    runNMS(windowsOut, scores, scaledImages12, nmsThreshold[0], windowsIn);
    std::cout << "d12 nms windows out = " << windowsIn.size() << std::endl;

    // Double the size of the rects to get from a 12x12 to 24x24
    // detection window.  Use scaledImages24 for the detection call
    // since that has the scales appropriate for the 24x24 detector
    for (auto it = windowsIn.begin(); it != windowsIn.end(); ++it)
    {
        it->first = cv::Rect(it->first.x * 2, it->first.y * 2,
                             it->first.width * 2, it->first.height * 2);
    }

    if ((detectThreshold.size() > 1) && (detectThreshold[1] > 0.0))
    {
        std::cout << "d24 windows in = " << windowsIn.size() << std::endl;
        runDetection(d24_, scaledImages24, windowsIn, detectThreshold[1], "ball", windowsOut, scores);
        std::cout << "d24 windows out = " << windowsOut.size() << std::endl;
        runNMS(windowsOut, scores, scaledImages24, nmsThreshold[1], windowsIn);
		std::cout << "d24 nms windows out = " << windowsIn.size() << std::endl;
    }

    // Final result - scale the output rectangles back to the
    // correct scale for the original sized image
    rectsOut.clear();
    for (auto it = windowsIn.cbegin(); it != windowsIn.cend(); ++it)
    {
        double   scale = scaledImages24[it->second].second;
        cv::Rect rect(it->first);
        cv::Rect scaledRect(cv::Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale));
        rectsOut.push_back(scaledRect);
    }
}


template<class MatT>
void NNDetect<MatT>::runNMS(const std::vector<Window>& windows,
                            const std::vector<float>& scores,
                            const std::vector<std::pair<MatT, double> >& scaledImages,
                            double nmsThreshold,
                            std::vector<Window>& windowsOut)
{
    if ((nmsThreshold > 0.0) && (nmsThreshold < 1.0))
    {
        // Detected is a rect, score pair.
        std::vector<Detected> detected;

        // Need to scale each rect to the correct mapping to the
        // original image, since rectangles from multiple different
        // scales might overlap
        for (size_t i = 0; i < windows.size(); i++)
        {
            double   scale = scaledImages[windows[i].second].second;
            cv::Rect rect(windows[i].first);
            cv::Rect scaledRect(cv::Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale));
            detected.push_back(Detected(scaledRect, scores[i]));
        }

        std::vector<size_t> nmsOut;
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
    const cv::Mat& depthIn,
    const cv::Size& minSize,
    const cv::Size& maxSize,
    int wsize,
    double scaleFactor,
    std::vector<std::pair<MatT, double> >& scaledImages,
    std::vector<Window>& windows)
{
    windows.clear();

    // How many pixels to move the window for each step
    // We use 4 - the calibration step can adjust +/- 2 pixels
    // in each direction, which means they will correct for
    // anything which is actually centered in one of the
    // pixels we step over.
    const int step = 4;

    double start = gtod_wrapper(); // grab start time

    // Create array of scaled images
    std::vector<std::pair<MatT, double> > scaledDepth;
    if (!depthIn.empty())
    {
        MatT depthGpu = MatT(depthIn);
        scalefactor(depthGpu, cv::Size(wsize, wsize), minSize, maxSize, scaleFactor, scaledDepth);
    }
    scalefactor(input, cv::Size(wsize, wsize), minSize, maxSize, scaleFactor, scaledImages);
    // Main loop.  Look at each scaled image in turn
    for (size_t scale = 0; scale < scaledImages.size(); ++scale)
    {
        float frac_size = (wsize * wsize) / ((float)scaledImages[scale].first.rows * (float)scaledImages[scale].first.cols);
        float depth_min = (323.2 * pow(frac_size, -.486)) - 300.;
        float depth_max = depth_min + 600.;
        std::cout << "Min/Max: " << depth_min << " " << depth_max << std::endl;
        // Start at the upper left corner.  Loop through the rows and cols until
        // the detection window falls off the edges of the scaled image
        for (int r = 0; (r + wsize) < scaledImages[scale].first.rows; r += step)
        {
            for (int c = 0; (c + wsize) < scaledImages[scale].first.cols; c += step)
            {
                if (!depthIn.empty())
                {
                    cv::Mat detectCheck = cv::Mat(scaledDepth[scale].first(cv::Rect(c, r, wsize, wsize)));
                    if(!depthInRange(depth_min, depth_max, detectCheck))
                    {
                        break;
                    }
                }
                windows.push_back(Window(cv::Rect(c, r, wsize, wsize), scale));
            }
        }
    }
    double end = gtod_wrapper();
    std::cout << "Generate initial windows time = " << (end - start) << std::endl;
}


template<class MatT>
void NNDetect<MatT>::runDetection(CaffeClassifier<MatT>& classifier,
                                  const std::vector<std::pair<MatT, double> >& scaledImages,
                                  const std::vector<Window>& windows,
                                  float threshold,
                                  std::string label,
                                  std::vector<Window>& windowsOut,
                                  std::vector<float>& scores)
{
    windowsOut.clear();
    scores.clear();
    // Accumulate a number of images to test and pass them in to
    // the NN prediction as a batch
    std::vector<MatT> images;

    // Return value from detection. This is a list of indexes from
    // the input which have a high enough confidence score
    std::vector<size_t> detected;

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
    std::cout << "runDetection time = " << (end - start) << std::endl;
}


// do 1 run of the classifier. This takes up batch_size predictions
// and adds the index of anything found to the detected list
template<class MatT>
void NNDetect<MatT>::doBatchPrediction(CaffeClassifier<MatT>&   classifier,
                                       const std::vector<MatT>& imgs,
                                       float                    threshold,
                                       const std::string&       label,
                                       std::vector<size_t>&     detected,
                                       std::vector<float>&      scores)
{
    detected.clear();
    // Grab the top 2 detected classes.  Since we're doing an object /
    // not object split, that will get everything
    std::vector<std::vector<Prediction> > predictions = classifier.ClassifyBatch(imgs, 2);

    // Each outer loop is the predictions for one input image
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        // Each inner loop is the prediction for a particular label
        // for the given image, sorted by score.
        //
        // Look for object with label <label>, > threshold confidence
        for (std::vector<Prediction>::const_iterator it = predictions[i].begin(); it != predictions[i].end(); ++it)
        {
            if (it->first == label)
            {
                if (it->second > threshold)
                {
                    detected.push_back(i);
                    scores.push_back(it->second);
                }
                break;
            }
        }
    }
}

template <class MatT>
bool NNDetect<MatT>::depthInRange(float depth_min, float depth_max, cv::Mat &detectCheck)
{
    for (int py = 0; py < detectCheck.rows; py++)
    {
        float *p = detectCheck.ptr<float>(py);
        for (int px = 0; px < detectCheck.cols; px++)
        {
            if ((p[px] < depth_max) && (p[px] > depth_min))
            {
                return true;
            }
        }
    }
    return false;
}

// Explicitly instatiate classes used elsewhere
template class NNDetect<cv::Mat>;
template class NNDetect<cv::gpu::GpuMat>;
