// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MMSEG_H
#define MMDEPLOY_MMSEG_H

#include "codebase/common.h"
#include "core/device.h"
#include "core/module.h"
#include "core/tensor.h"

namespace mmdeploy::mmseg {

struct SegmentorOutput {
  Tensor mask;
  int height;
  int width;
  int classes;
  MMDEPLOY_ARCHIVE_MEMBERS(mask, height, width, classes);
};

DECLARE_CODEBASE(MMSegmentation, mmseg);

}  // namespace mmdeploy::mmseg

#endif  // MMDEPLOY_MMSEG_H
