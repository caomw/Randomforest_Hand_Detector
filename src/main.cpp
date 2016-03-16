#include "randomforest.hpp"

using namespace std;
using namespace cv;
using namespace handlib;

int main()
{
    CRandomForest randomforest_hand_detector;
    CTrainParam tp;
    
    randomforest_hand_detector.TrainForest(tp);
    randomforest_hand_detector.LoadForest();
    Mat img = imread("test.png");
    imshow("res", randomforest_hand_detector.Detect(img));
    waitKey();
    
    return 0;
}