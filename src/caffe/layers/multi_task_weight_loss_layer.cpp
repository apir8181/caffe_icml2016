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

  debug_info_ = this->layer_param_.weight_loss_param().debug_info();
  num_tasks_ = bottom.size();
	
	num_classes_ = 0;
	for (int i = 0; i < num_tasks_; ++ i) {
		num_classes_ += bottom[i]->shape(0);
	}

	feature_dim_ = bottom[0]->count(1);
	for (int i = 0; i < num_tasks_; ++ i) {
		CHECK_EQ(feature_dim_, bottom[i]->count(1)) 
			<< "feature should have same dimension";
	}

	data_.Reshape(num_classes_, feature_dim_, 1, 1);

	task_start_index_.Reshape(num_tasks_, 1, 1, 1);
	task_end_index_.Reshape(num_tasks_, 1, 1, 1);

	int index_sofar = 0;
	for (int i = 0; i < num_tasks_; ++ i) {
		task_start_index_.mutable_cpu_data()[i] = index_sofar;
		index_sofar += bottom[i]->shape(0);
		task_end_index_.mutable_cpu_data()[i] = index_sofar;
	}

	index_sofar = 0;
	data2task_.Reshape(num_classes_, 1, 1, 1);
	for (int i = 0; i < num_tasks_; ++ i) {
		for (int j = 0; j < bottom[i]->shape(0); ++ j) {
			data2task_.mutable_cpu_data()[index_sofar ++] = i;
		}
	}

	pairwise_distance_.Reshape(num_classes_, num_classes_, 1, 1);
	pairwise_kernel_.Reshape(num_classes_, num_classes_, 1, 1);

	loss_.Reshape(num_tasks_, num_tasks_, 1, 1);
	Omega_.Reshape(num_tasks_, num_tasks_, 1, 1);
	A_.Reshape(num_tasks_, num_tasks_, 1, 1);
	caffe_gpu_set<Dtype>(num_tasks_ * num_tasks_, 0, Omega_.mutable_gpu_data());
	for (int i = 0; i < num_tasks_; ++ i) {
		Omega_.mutable_cpu_data()[i * (num_tasks_ + 1)] = 1;
	}

  // int max_N = 0;
  // int max_K = 0;  
  // total_W_num_ = 0;
  // D_.Reshape(1, 1, 1, num_of_tasks_);
  // for(int i = 0;i < bottom.size();++i){
  //     D_.mutable_cpu_data()[i] = bottom[i]->count(0, 1);
  //     D_.mutable_cpu_diff()[i] = bottom[i]->count(1);
  //     if(D_.cpu_data()[i] > max_N){
  //         max_N = D_.cpu_data()[i];
  //     }
  //     if(D_.cpu_diff()[i] > max_K){
  //         max_K = D_.cpu_diff()[i];
  //     }
  //     total_W_num_ += D_.cpu_data()[i];
  // }
  // temp_.Reshape(1, 1, 1, total_W_num_);
  // data_.Reshape(1, total_W_num_, total_W_num_, max_K);
  // kernel_.Reshape(1, 1, total_W_num_, total_W_num_);
  // dimension_ = max_K;
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

}

#ifdef CPU_ONLY
STUB_GPU(MultiTaskWeightLossLayer);
#endif

INSTANTIATE_CLASS(MultiTaskWeightLossLayer);
REGISTER_LAYER_CLASS(MultiTaskWeightLoss);

}  // namespace caffe
