#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multi_task_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  num_of_tasks_ = bottom.size();
  Omega_.Reshape(1, 1, num_of_tasks_, num_of_tasks_);
  int max_N = 0;
  int max_K = 0;  
  total_W_num_ = 0;
  D_.Reshape(1, 1, 1, num_of_tasks_);
  for(int i = 0;i < bottom.size();++i){
      D_.mutable_cpu_data()[i] = bottom[i]->count(0, 1);
      D_.mutable_cpu_diff()[i] = bottom[i]->count(1);
      if(D_.cpu_data()[i] > max_N){
          max_N = D_.cpu_data()[i];
      }
      if(D_.cpu_diff()[i] > max_K){
          max_K = D_.cpu_diff()[i];
      }
      total_W_num_ += D_.cpu_data()[i];
  }
  temp_.Reshape(1, 1, 1, total_W_num_);
  data_.Reshape(1, total_W_num_, total_W_num_, max_K);
  kernel_.Reshape(1, 1, total_W_num_, total_W_num_);
  dimension_ = max_K;
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MultiTaskWeightLossLayer);
#endif

INSTANTIATE_CLASS(MultiTaskWeightLossLayer);
REGISTER_LAYER_CLASS(MultiTaskWeightLoss);

}  // namespace caffe
