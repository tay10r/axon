#include <axon/exporter.hpp>

#include <map>
#include <string>

namespace axon {

namespace {

std::map<std::string, std::shared_ptr<Exporter>> g_registry;

} // namespace

void
Exporter::addToRegistry(const char* name, std::shared_ptr<Exporter> exporter)
{
  g_registry.emplace(name, std::move(exporter));
}

auto
Exporter::create(const char* name) -> std::shared_ptr<Exporter>
{
  return g_registry.at(name);
}

Exporter::~Exporter() = default;

} // namespace axon
