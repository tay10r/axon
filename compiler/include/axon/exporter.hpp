#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace axon {

class Module;

class Exporter
{
public:
  static void addToRegistry(const char* name, std::shared_ptr<Exporter> exporter);

  [[nodiscard]] static auto create(const char* name) -> std::shared_ptr<Exporter>;

  virtual ~Exporter();

  virtual void exportFull(const Module& evalModule,
                          const Module& gradModule,
                          const std::filesystem::path& outputPath) = 0;

  virtual void exportLean(const Module& evalModule,
                          const std::vector<float>& parameters,
                          const std::filesystem::path& outputPath) = 0;
};

} // namespace axon
