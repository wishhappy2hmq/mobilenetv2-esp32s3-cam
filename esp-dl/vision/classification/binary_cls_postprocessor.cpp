#include "binary_cls_postprocessor.hpp"
#include "binary_cls_category_name.hpp"

namespace dl {
namespace cls {
BinaryClsPostprocessor::BinaryClsPostprocessor(
    Model *model, const int top_k, const float score_thr, bool need_softmax, const std::string &output_name) :
    ClsPostprocessor(model, top_k, score_thr, need_softmax, output_name)
{
    m_cat_names = binary_cls_cat_names;
}
} // namespace cls
} // namespace dl
