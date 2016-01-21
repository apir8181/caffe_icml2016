#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multi_task_feature_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiTaskFeatureLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  num_of_tasks_ = bottom.size();
  Theta_ = new Dtype[num_of_tasks_ * num_of_tasks_];
}

template <typename Dtype>
void MultiTaskFeatureLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MultiTaskFeatureLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultiTaskFeatureLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MultiTaskFeatureLossLayer);
#endif

INSTANTIATE_CLASS(MultiTaskFeatureLossLayer);
REGISTER_LAYER_CLASS(MultiTaskFeatureLoss);

}  // namespace caffe
