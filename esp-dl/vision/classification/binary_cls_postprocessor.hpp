#pragma once
#include "dl_cls_postprocessor.hpp"

namespace dl {
namespace cls {
class BinaryClsPostprocessor : public ClsPostprocessor {
public:
    BinaryClsPostprocessor(
        Model *model, const int top_k, const float score_thr, bool need_softmax, const std::string &output_name = "");
};
} // namespace cls
} // namespace dl
