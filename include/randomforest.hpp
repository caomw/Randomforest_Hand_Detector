#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "randomforest_base.hpp"

namespace handlib
{

class CRandomForest 
{
private:
    CTrainParam tp;
    CTrainingData td;
    std::vector<std::vector<CNode> > trees;
    
    void TrainTree(int tree_id);
    inline bool TestLeaf(int l, int r, int dep);
    inline int GetFeature(CPixel &p, int du, int dv);
    inline int CRandomForest::GetFeature(cv::Mat &img, int u, int v, int du, int dv);
    float GetProb(int l, int r);
    CSplitCandidate FindBestPhi(int l, int r);
    int SortData(int l, int r, CSplitCandidate &phi);
    
    float Predict(cv::Mat &img, int u, int v);
    
    void SaveNode(int tree_id, int node, ostream &fout);
    void SaveTree(int tree_id, ostream &fout);
    void LoadNode(int tree_id, string node_type, int node, ostream &fin);
    void LoadTree(int tree_id, istream &fin);
    void SaveForest(std::string file_name = "forest.model");
public:
    CRandomForest() {}
    ~CRandomForest();
    
    void TrainForest(CTrainParam &train_param);
    void LoadForest(std::string file_name = "forest.model");
    cv::Mat Detect(cv::Mat &img);
};

}


#endif