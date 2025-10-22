#pragma once

#include <memory>

#include <stdint.h>

namespace axon {

class ProceduralData
{
public:
  virtual ~ProceduralData() = default;

  virtual void generate(float* row) = 0;
};

class Dataset
{
public:
  [[nodiscard]] static auto save(const char* filename, const float* data, const uint32_t rows, const uint32_t cols)
    -> bool;

  [[nodiscard]] static auto create() -> std::unique_ptr<Dataset>;

  [[nodiscard]] static auto create(uint32_t rows, uint32_t cols, ProceduralData& data) -> std::unique_ptr<Dataset>;

  virtual ~Dataset();

  [[nodiscard]] virtual auto load(const char* filename) -> bool = 0;

  [[nodiscard]] virtual auto rows() const -> uint32_t = 0;

  [[nodiscard]] virtual auto cols() const -> uint32_t = 0;

  [[nodiscard]] virtual auto data() const -> const float* = 0;
};

} // namespace axon
