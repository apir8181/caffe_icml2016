#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/weight_transfer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/* params
 * M_: batch_size
 * K_: input_dim
 * N_: output_dim
 */
template <typename Dtype>
void WeightTransferLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_memcpy(this->blobs_[0]->count() * sizeof(Dtype), weight, top_data);
}

template <typename Dtype>
void WeightTransferLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_add(this->blobs_[0]->count(), top_diff, weight_diff, weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightTransferLayer);

}  // namespace caffe
