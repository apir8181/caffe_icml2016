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
  Omega_ = new Dtype[num_of_tasks_ * num_of_tasks_];
  Omega_cache_ = new Dtype[num_of_tasks_ * num_of_tasks_];
  caffe_set(num_of_tasks_ * num_of_tasks_, Dtype(0), Omega_);
  int max_N = 0;
  int max_K = 0;  
  total_W_num_ = 0;
  for(int i = 0;i < bottom.size();++i){
      N_.push_back(bottom[i]->count(0, 1));
      K_.push_back(bottom[i]->count(1));
      if(N_[i] > max_N){
          max_N = N_[i];
      }
      if(K_[i] > max_K){
          max_K = K_[i];
      }
      total_W_num_ += N_[i];
  }
  temp_.Reshape(1, 1, 1, max_K);
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
