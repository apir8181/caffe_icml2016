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

  CHECK_GT(bottom.size(), 1) << "bottom size should be greater than 1";
  debug_info_ = this->layer_param_.weight_loss_param().debug_info();
	debug_detail_ = this->layer_param_.weight_loss_param().debug_detail();
  fronzen_omega_ = this->layer_param_.weight_loss_param().fronzen_omega();
  fronzen_theta_ = this->layer_param_.weight_loss_param().fronzen_theta();
  
  num_task_ = bottom.size();
	num_class_ = bottom[0]->shape(0);
  num_feature_ = bottom[0]->count(1);

  // Init W_by_task
  for (int i = 0; i < num_task_; ++ i) {
    Blob<int>* W = new Blob<int>();
    W->Reshape(num_class_, num_feature_, 1, 1);
    W_by_task_.push_back(W);
  }

  // Init W_by_class
  for (int i = 0; i < num_class_; ++ i) {
    Blob<int> *W = new Blob<int>();
    W->Reshape(num_task_, num_feature_, 1, 1);
    W_by_class_.push_back(W);
  }

  A_by_task_.Reshape(num_task_, num_task_, 1, 1);
  A_by_class_.Reshape(num_class_, num_class_, 1, 1);

  // Init Omega
  Omega_.Reshape(num_task_, num_task_, 1, 1);
  Dtype* Omega = Omega_.mutable_cpu_data();
  caffe_memset(Omega_.count() * sizeof(Dtype), 0, Omega);
  for (int i = 0; i < num_task_; ++ i) {
    Omega[i * (num_task_ + 1)] = 1;
  }
  
  Theta_.Reshape(num_class_, num_class_, 1, 1);
  Dtype* Theta = Theta_.mutable_cpu_data();
  caffe_memset(Theta_.count() * sizeof(Dtype), 0, Theta);
  for (int i = 0; i < num_class_; ++ i) {
    Theta[i * (num_class_ + 1)] = 1;
  }
  
  temp_D_C_.Reshape(num_feature_, num_class_, 1, 1);
  temp_D_D_.Reshape(num_feature_, num_feature_, 1, 1);
  temp_D_T_.Reshape(num_feature_, num_task_, 1, 1);
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  CHECK_EQ(bottom.size(), num_task_)
    << "number of task can't change. "
    << num_task_ << "(expected) vs " << bottom.size();
  
  for (int i = 0; i < num_tasks_; ++ i) {
    CHECK_EQ(bottom[i]->shape(0), num_class_)
      << "number of class can't change."
      << num_class_ << "(expceted) vs " << bottom[i]->shape(0);
  }

  for (int i = 0; i < num_feature_; ++ i) {
    CHECK_EQ(bottom[i]->count(1), num_feature_)
      << "number of feature can't change."
      << num_feature_ << "(expceted) vs " << bottom[i]->count(1);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiTaskWeightLossLayer);
#endif

INSTANTIATE_CLASS(MultiTaskWeightLossLayer);
REGISTER_LAYER_CLASS(MultiTaskWeightLoss);

}  // namespace caffe
