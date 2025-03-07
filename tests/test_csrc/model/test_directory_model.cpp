// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on
#include "core/model.h"
#include "core/model_impl.h"
#include "test_resource.h"

using namespace mmdeploy;

TEST_CASE("test directory model", "[model]") {
  std::unique_ptr<ModelImpl> model_impl;
  for (auto& entry : ModelRegistry::Get().ListEntries()) {
    if (entry.name == "DirectoryModel") {
      model_impl = entry.creator();
      break;
    }
  }
  REQUIRE(model_impl);

  auto& gResource = MMDeployTestResources::Get();
  auto directory_model_list = gResource.LocateModelResources("sdk_models");
  REQUIRE(!directory_model_list.empty());
  auto model_dir = "sdk_models/good_model";
  REQUIRE(gResource.IsDir(model_dir));
  auto model_path = gResource.resource_root_path() + "/" + model_dir;
  REQUIRE(!model_impl->Init(model_path).has_error());
  REQUIRE(!model_impl->ReadFile("deploy.json").has_error());
  REQUIRE(model_impl->ReadFile("not-existing-file").has_error());

  model_dir = "sdk_models/bad_model";
  REQUIRE(gResource.IsDir(model_dir));
  model_path = gResource.resource_root_path() + "/" + model_dir;
  REQUIRE(!model_impl->Init(model_path).has_error());
  REQUIRE(model_impl->ReadMeta().has_error());
}
