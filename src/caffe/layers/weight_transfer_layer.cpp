#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/weight_transfer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightTransferLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.weight_transfer_param().axis());
  const int weight_output = this->layer_param_.weight_transfer_param().weight_output_num();
  const int weight_input = bottom[0]->count(axis);
  N_ = weight_output;
  K_ = weight_input;
  // Check if we need to set up the weights
  bias_term_ = this->layer_param_.weight_transfer_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.weight_transfer_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.weight_transfer_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), this->blobs_[1]->mutable_gpu_diff());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), this->blobs_[0]->mutable_gpu_diff());
}

template <typename Dtype>
void WeightTransferLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  // const int axis = bottom[0]->CanonicalAxisIndex(
  //     this->layer_param_.weight_transfer_param().axis());
  // const int new_N = bottom[0]->count(axis);
  // CHECK_EQ(N_, new_N)
  //     << "Input size incompatible with weight transfer parameters.";
  // M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape(2);
  top_shape[0] = N_;
  top_shape[1] = K_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  // if (bias_term_) {
  //   vector<int> bias_shape(1, M_);
  //   bias_multiplier_.Reshape(bias_shape);
  //   caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  // }
}

template <typename Dtype>
void WeightTransferLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void WeightTransferLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(WeightTransferLayer);
#endif

INSTANTIATE_CLASS(WeightTransferLayer);
REGISTER_LAYER_CLASS(WeightTransfer);

}  // namespace caffe
