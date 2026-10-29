#include "chat/cli_tools.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <vector>
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
template<class F> void rejects(F fn) {bool bad=false;try{fn();}catch(const std::exception&){bad=true;}require(bad,"expected rejection");}
void write(const fs::path& p,const json& j){std::ofstream(p)<<j.dump();}
json tool(const std::string& name,json properties,json required){return {{"type","function"},{"function",{{"name",name},{"description","demo"},{"parameters",{{"type","object"},{"properties",properties},{"required",required},{"additionalProperties",false}}}}}};}
chat_msg call(const std::string& id,const std::string& name,const json& args){chat_msg m;m.tool_calls.push_back({id,name,args.dump()});return m;}
int main(){
 const auto root=fs::temp_directory_path()/("flm-native-tools-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 fs::create_directories(root/"data");
 try{
  const auto add=tool("add_numbers",{{"a",{{"type","integer"}}},{"b",{{"type","integer"}}}},json::array({"a","b"}));
  const auto lookup=tool("lookup_record",{{"name",{{"type","string"}}}},json::array({"name"}));
  const auto read=tool("read_record",{{"path",{{"type","string"}}}},json::array({"path"}));
  write(root/"tools.json",json::array({add,lookup,read}));
  write(root/"data/index.json",{{"oak","oak.json"}});write(root/"data/oak.json",{{"a",12},{"b",-5},{"label","oak-demo"}});
  flm::chat::cli_tools_session session(root/"tools.json","demo",{},8);
  require(session.tools().size()==4,"finish tool added");
  require(session.execute("add_numbers",{{"a",17},{"b",28}})==json({{"result",45}}),"actual addition");
  require(session.execute("add_numbers",{{"a",-31},{"b",8}})==json({{"result",-23}}),"varied addition");
  require(session.execute("lookup_record",{{"name","oak"}})==json({{"path","oak.json"}}),"lookup");
  require(session.execute("read_record",{{"path","oak.json"}})["a"]==12,"read actual data");
  write(root/"data/oak.json",{{"a",99}});require(session.execute("read_record",{{"path","oak.json"}})["a"]==99,"updated data");
  for(const auto& p:{"../secret.json","/secret.json","C:\\secret.json","..\\secret.json"})rejects([&]{session.execute("read_record",{{"path",p}});});
  rejects([&]{session.execute("add_numbers",{{"a",true},{"b",8}});});
  rejects([&]{session.execute("add_numbers",{{"a",1},{"b",8},{"c",9}});});
  rejects([&]{session.execute("unknown",json::object());});
  int turn=0;std::ostringstream output;
  session.run("add",[&](const json& messages,const json& tools){
   require(tools.size()==4,"tools passed");
   if(turn++==0){require(messages.back()["content"]=="add","prompt passed");return call("one","add_numbers",{{"a",17},{"b",28}});}
   require(messages.back()["role"]=="tool" && messages.back()["tool_call_id"]=="one","tool ID history");
   require(json::parse(messages.back()["content"].get<std::string>())["result"]==45,"result history");
   return call("two","finish_task",{{"answer","45"}});
  },output);
  require(turn==2 && output.str().find("[answer] 45")!=std::string::npos,"loop completed");
  turn=0;rejects([&]{std::ostringstream o;session.run("x",[&](const json&,const json&){return call("dup","add_numbers",{{"a",1},{"b",2}});},o);});
  flm::chat::cli_tools_session cap(root/"tools.json","demo",{},1);
  rejects([&]{std::ostringstream o;cap.run("x",[&](const json&,const json&){return call("a","add_numbers",{{"a",1},{"b",2}});},o);});
  auto custom=tool("get_weather",{{"city",{{"type","string"}}}},json::array({"city"}));
  write(root/"weather.json",json::array({custom}));
  {
   flm::chat::cli_tools_session automatic(root/"tools.json","demo",{},8,"auto");
   require(automatic.tools().size()==3 && automatic.automatic(),"auto must not inject a finish function");
   turn=0;output.str("");
   automatic.run("add",[&](const json& messages,const json& active){
    require(active.size()==3,"auto tools changed");
    if(turn++==0)return call("auto-one","add_numbers",{{"a",17},{"b",28}});
    require(json::parse(messages.back()["content"].get<std::string>())["result"]==45,"auto lost actual result");
    chat_msg m;m.content="45";return m;
   },output);
   require(turn==2 && output.str().find("[answer] 45")!=std::string::npos &&
       output.str().find("finish_task")==std::string::npos,"auto did not finish on ordinary answer");
   output.str("");automatic.run("greet",[](const json&,const json&){chat_msg m;m.content="Hello";return m;},output);
   require(output.str()=="[answer] Hello\n","no-tool auto answer");
   flm::chat::cli_tools_session auto_generic(root/"weather.json","",{},8,"auto");
   output.str("");auto_generic.run("greet",[](const json&,const json&){chat_msg m;m.content="Hello";return m;},output);
   require(json::parse(output.str())["content"]=="Hello","generic auto answer JSON");
   rejects([&]{flm::chat::cli_tools_session bad(root/"tools.json","demo",{},8,"none");});
  }
  flm::chat::cli_tools_session generic(root/"weather.json","",{},8);
  output.str(""); generic.run("weather",[&](const json&,const json&){return call("w","get_weather",{{"city","Shanghai"}});},output);
  require(json::parse(output.str())["tool_calls"][0]["function"]["name"]=="get_weather","generic CLI calls");
  rejects([&]{flm::chat::cli_tools_session x(root/"weather.json","demo",{},8);});
  rejects([&]{std::ostringstream o;generic.run("x",[](const json&,const json&){return chat_msg{};},o);});
  rejects([&]{std::ostringstream o;generic.run("x",[](const json&,const json&){return call("a","wrong",json::object());},o);});
  for(const auto& name:std::vector<std::string>{"bad\"name","bad\nname","weather.city",std::string(65,'a'),"天气"}){
   auto malformed=custom;malformed["function"]["name"]=name;
   write(root/"invalid-name.json",json::array({malformed}));
   rejects([&]{flm::chat::cli_tools_session x(root/"invalid-name.json","",{},8);});
  }
  for(const auto& name:std::vector<std::string>{"weather_v2-test",std::string(64,'a')}){
   auto valid=custom;valid["function"]["name"]=name;
   write(root/"valid-name.json",json::array({valid}));
   flm::chat::cli_tools_session x(root/"valid-name.json","",{},8);
   require(x.tools()[0]["function"]["name"]==name,"valid OpenAI function name");
  }
  write(root/"bad.json",json::array());rejects([&]{flm::chat::cli_tools_session x(root/"bad.json","",{},8);});
  write(root/"bad.json",json::array({add,add}));rejects([&]{flm::chat::cli_tools_session x(root/"bad.json","",{},8);});
  std::ofstream(root/"big.json")<<std::string(65537,' ');rejects([&]{flm::chat::cli_tools_session x(root/"big.json","",{},8);});
  std::cout<<"PASS native CLI definitions, real executors, history, boundaries and failure cases\n";
  fs::remove_all(root);return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";fs::remove_all(root);return 1;}
}
