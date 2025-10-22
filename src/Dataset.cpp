#include <axon/Dataset.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace axon {

namespace {

constexpr size_t headerSize{ 12 };

class DatasetImpl final : public Dataset
{
public:
  ~DatasetImpl() { free(m_data); }

  [[nodiscard]] auto load(const char* filename) -> bool override
  {
    auto* file = fopen(filename, "rb");
    if (!file) {
      return false;
    }

    char magic[4]{};

    if (fread(magic, sizeof(magic), 1, file) != 1) {
      fclose(file);
      return false;
    }

    if (memcmp(magic, "AXD\n", 4) != 0) {
      fclose(file);
      return false;
    }

    uint32_t dims[2]{};

    if (fread(dims, sizeof(dims), 1, file) != 1) {
      fclose(file);
      return false;
    }

    fseek(file, 0, SEEK_END);

    const auto fileSize = ftell(file);

    if ((fileSize == -1L) || ((headerSize + dims[0] * dims[1] * sizeof(float)) != static_cast<size_t>(fileSize))) {
      fclose(file);
      return false;
    }

    fseek(file, headerSize, SEEK_SET);

    if (!resizeData(dims[0], dims[1])) {
      fclose(file);
      return false;
    }

    if (fread(m_data, sizeof(float), m_rows * m_cols, file) != (m_rows * m_cols)) {
      fclose(file);
      return false;
    }

    fclose(file);

    return true;
  }

  [[nodiscard]] auto rows() const -> uint32_t override { return m_rows; }

  [[nodiscard]] auto cols() const -> uint32_t override { return m_cols; }

  [[nodiscard]] auto data() const -> const float* override { return m_data; }

  void generate(const uint32_t rows, const uint32_t cols, ProceduralData& procData)
  {
    if (!resizeData(rows, cols)) {
      return;
    }

    for (size_t i = 0; i < m_rows; i++) {

      procData.generate(m_data + i * m_cols);
    }
  }

protected:
  [[nodiscard]] auto resizeData(const uint32_t rows, const uint32_t cols) -> bool
  {
    auto* tmp = realloc(m_data, static_cast<size_t>(rows * cols) * sizeof(float));
    if (!tmp) {
      return false;
    }

    m_data = static_cast<float*>(tmp);
    m_dataSize = rows * cols;
    m_rows = rows;
    m_cols = cols;

    return true;
  }

private:
  float* m_data{};

  size_t m_dataSize{};

  uint32_t m_cols{};

  uint32_t m_rows{};
};

} // namespace

auto
Dataset::save(const char* filename, const float* data, const uint32_t rows, const uint32_t cols) -> bool
{
  auto* file = fopen(filename, "wb");
  if (!file) {
    return false;
  }

  if (fwrite("AXD\n", 4, 1, file) != 1) {
    fclose(file);
    return false;
  }

  uint32_t dims[2]{ rows, cols };

  if (fwrite(dims, sizeof(dims), 1, file) != 1) {
    fclose(file);
    return false;
  }

  if (fwrite(data, sizeof(float), rows * cols, file) != (rows * cols)) {
    fclose(file);
    return false;
  }

  fclose(file);

  return true;
}

auto
Dataset::create() -> std::unique_ptr<Dataset>
{
  return std::make_unique<DatasetImpl>();
}

auto
Dataset::create(const uint32_t rows, const uint32_t cols, ProceduralData& data) -> std::unique_ptr<Dataset>
{
  auto result = std::make_unique<DatasetImpl>();

  result->generate(rows, cols, data);

  return result;
}

Dataset::~Dataset() = default;

} // namespace axon
