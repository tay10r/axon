#include <axon/compiler.hpp>
#include <axon/exception.hpp>
#include <axon/exporter.hpp>
#include <axon/module.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>

#include "c_exporter.hpp"

namespace {

[[nodiscard]] auto
checkOpt(const std::string& arg, const char* shortOpt, const char* longOpt) -> bool
{
  return (arg == shortOpt) || (arg == longOpt);
}

class ArgQueue final
{
public:
  ArgQueue(int argc, char** argv)
  {
    for (int i = 1; i < argc; i++) {
      m_args.emplace_back(argv[i]);
    }
  }

  [[nodiscard]] auto pop() -> std::string
  {
    auto arg = std::move(m_args.at(0));

    m_args.erase(m_args.begin());

    return arg;
  }

  template<typename T>
  [[nodiscard]] auto popValue(const std::string& key) -> T
  {
    if (empty()) {
      std::ostringstream what;
      what << "missing value for \"" << key << "\"";
      throw axon::Exception(what.str());
    }
    std::istringstream tmp(pop());
    T value;
    tmp >> value;
    return value;
  }

  [[nodiscard]] auto empty() const -> bool { return m_args.empty(); }

private:
  std::vector<std::string> m_args;
};

void
exec(int argc, char** argv)
{
  axon::registerCExporter();

  axon::Compiler::Options options;

  ArgQueue args(argc, argv);

  while (!args.empty()) {

    const auto arg = args.pop();

    if (checkOpt(arg, "-n", "--name")) {
      options.networkName = args.popValue<std::string>(arg);
      continue;
    }

    if (checkOpt(arg, "-o", "--output-dir")) {
      options.outputFile = args.popValue<std::string>(arg);
      continue;
    }

    if (checkOpt(arg, "-r", "--release")) {
      options.release = true;
      continue;
    }

    if (checkOpt(arg, "-p", "--params")) {
      options.parametersPath = args.popValue<std::string>(arg);
      continue;
    }

    if (checkOpt(arg, "-e", "--exporter")) {
      options.exporter = args.popValue<std::string>(arg);
      continue;
    }

    std::ostringstream what;
    what << "unknown option \"" << arg << "\"";
    throw axon::Exception(what.str());
  }

  auto compiler = axon::Compiler::create(options);

  compile(*compiler);

  const auto* evalModule = compiler->getEvalModule();
  if (!evalModule) {
    throw axon::Exception("no module was defined");
  }

  const auto* gradModule = compiler->getGradModule();
  if (!gradModule) {
    throw axon::Exception("no grad module was defined");
  }

  auto exporter = axon::Exporter::create(options.exporter.c_str());

  if (options.release) {
    exporter->exportLean(*evalModule, {}, options.outputFile);
  } else {
    exporter->exportFull(*evalModule, *gradModule, options.outputFile);
  }
}

} // namespace

auto
main(int argc, char** argv) -> int
{
  try {
    exec(argc, argv);
  } catch (const axon::Exception& ex) {
    std::cerr << "exception at " << ex.location().file_name() << " line " << ex.location().line() << std::endl;
    std::cerr << "  " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
