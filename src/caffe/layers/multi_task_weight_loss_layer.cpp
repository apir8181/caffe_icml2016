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
	debug_detail_ = this->layer_param_.weight_loss_param().debug_detail();
	sigma_ = this->layer_param_.weight_loss_param().sigma();
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

	pairwise_sqr_distance_.Reshape(num_classes_, num_classes_, 1, 1);
	pairwise_kernel_.Reshape(num_classes_, num_classes_, 1, 1);

	vector<int> pair_task_shape(2, num_tasks_);
	loss_.Reshape(pair_task_shape);
	A_.Reshape(pair_task_shape);

	// Init Omega
	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob<Dtype>(pair_task_shape));
	caffe_gpu_set<Dtype>(num_tasks_ * num_tasks_, -1.0 / (num_tasks_ - 1),
											 this->blobs_[0]->mutable_gpu_data());
	for (int i = 0; i < num_tasks_; ++ i) {
		this->blobs_[0]->mutable_cpu_data()[i * (num_tasks_ + 1)] = 1.0;
	}
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
