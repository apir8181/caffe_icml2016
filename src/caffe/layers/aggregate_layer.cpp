#include <algorithm>
#include <vector>

#include "caffe/layers/aggregate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AggregateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AggregateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape = bottom[0]->shape();
    int num = 0;
    for(int i = 0;i < bottom.size();++i){
        num += bottom[i]->count(0, 1);
    }
    top_shape[0] = num;
    top[0]->Reshape(top_shape);
    if(bottom.size() == 1){
        top[0]->ShareData(*bottom[0]);
        top[0]->ShareDiff(*bottom[0]);
    }
}

template <typename Dtype>
void AggregateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AggregateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(AggregateLayer);
#endif

INSTANTIATE_CLASS(AggregateLayer);
REGISTER_LAYER_CLASS(Aggregate);

}  // namespace caffe
