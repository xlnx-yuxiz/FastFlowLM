#pragma once
#include "chat/chat_msg.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <functional>
#include <ostream>
#include <string>

namespace flm::chat {
class cli_tools_session {
public:
    using json = nlohmann::ordered_json;
    using generate_fn = std::function<chat_msg(const json&, const json&)>;
    cli_tools_session(const std::filesystem::path& file, const std::string& executor,
                      const std::filesystem::path& data_dir, int max_turns,
                      const std::string& choice = "required");
    const json& tools() const { return tools_; }
    bool automatic() const { return automatic_; }
    void run(const std::string& prompt, const generate_fn& generate, std::ostream& output) const;
    json execute(const std::string& name, const json& arguments) const;
private:
    json tools_;
    bool demo_;
    bool automatic_;
    int max_turns_;
    std::filesystem::path data_dir_;
    json record(const std::string& relative) const;
};
}  // namespace flm::chat
