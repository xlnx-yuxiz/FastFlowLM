// Test-only fault adapter around real OGA operations. The production recovery,
// accounting and checkpoint commit statements are compiled from a pinned .inc.
#include "AutoModel/hip_oga_prefix_cache.hpp"
#include "hip_oga_policy.hpp"
#include "hip_oga_config_directory.hpp"
#include <ort_genai.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using nlohmann::json;
using namespace flm::hip_oga;
using Clock = std::chrono::steady_clock;

void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void header_print(const char*, const std::string& message) { std::cout << message << '\n'; }
namespace time_utils {
auto duration_ns(Clock::time_point a, Clock::time_point b) {
    return std::make_pair(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count(),0);
}
}
struct Timer { Clock::time_point start(){return Clock::now();} Clock::time_point stop(std::size_t){return Clock::now();} };
enum { PREFILL_TIME };

class Provider {
public:
    OgaModel& model;
    std::unique_ptr<OgaGeneratorParams> params;
    std::unique_ptr<OgaGenerator> generator;
    std::size_t checkpoint=0;
    std::string fault;
    bool fired=false, rebuild_failed=false;
    json events=json::array();
    explicit Provider(OgaModel& value):model(value){}
    void event(const char* stage) {
        events.push_back({{"stage",stage},{"sequenceCount",generator?generator->GetSequenceCount(0):0}});
    }
    void fail(const char* stage) {
        if (!fired && fault==stage) {fired=true;event(stage);throw std::runtime_error(std::string("injected:")+stage);}
    }
    void begin_turn(const std::vector<int>& tokens) {
        generator.reset();
        params=OgaGeneratorParams::Create(model);
        params->SetSearchOption("max_length",1024);
        params->SetSearchOption("batch_size",1);
        params->SetSearchOptionBool("do_sample",false);
        generator=OgaGenerator::Create(model,*params);
        std::vector<int32_t> ids(tokens.begin(),tokens.end());
        if (fault=="rebuild-fails" && fired && !rebuild_failed) {
            generator->AppendTokens(ids.data(),1);
            rebuild_failed=true;event("partial-rebuild");
            throw std::runtime_error("injected:rebuild-fails");
        }
        generator->AppendTokens(ids.data(),ids.size());
        checkpoint=tokens.size();event("fresh");
        require(generator->GetSequenceCount(0)==tokens.size(),"fresh sequence count differs");
    }
    void restore_text_checkpoint() {
        fail("before-rewind");
        generator->RewindTo(checkpoint);event("rewound");
        require(generator->GetSequenceCount(0)==checkpoint,"rewind sequence count differs");
        fail("after-rewind");
    }
    void append_text_suffix(const std::vector<int>& suffix) {
        if(suffix.empty()){event("empty-suffix");return;}
        std::vector<int32_t> ids(suffix.begin(),suffix.end());
        auto before=generator->GetSequenceCount(0);
        if(!fired && fault=="partial-append") {
            require(ids.size()>1,"partial append fixture too short");
            generator->AppendTokens(ids.data(),1);event("partial-append");
            fired=true;throw std::runtime_error("injected:partial-append");
        }
        generator->AppendTokens(ids.data(),ids.size());event("appended");
        require(generator->GetSequenceCount(0)==before+ids.size(),"append sequence count differs");
        fail("after-append");
        if(!fired && fault=="rebuild-fails") {fired=true;throw std::runtime_error("injected:after-append");}
    }
    int checkpoint_text_prompt(){checkpoint=generator->GetSequenceCount(0);return static_cast<int>(checkpoint);}
    std::size_t generator_capacity() const {return generator?1024:0;}
    void log_sampling_settings(bool){}
    std::vector<int> generate(int limit=8) {
        std::vector<int> result;
        while(static_cast<int>(result.size())<limit && !generator->IsDone()) {
            const auto before=generator->GetSequenceCount(0);
            generator->GenerateNextToken();
            auto n=generator->GetSequenceCount(0);
            require(n==before+1,"generation advanced by more than one token");
            result.push_back(generator->GetSequenceData(0)[n-1]);
        }
        event("generated");
        require(!result.empty(),"generator produced no tokens");return result;
    }
};

struct Harness {
    Provider provider;
    PrefixCacheState prefix_cache_;
    Timer profiler_list[1];
    std::vector<int> token_history;
    explicit Harness(OgaModel& m):provider(m){}
    json insert(const std::vector<int>& tokens,bool allowed=true) {
        auto* hip_engine=&provider;
        const std::string signature="test-model;ctx=1024;greedy";
        auto cache_decision=prefix_cache_.evaluate(true,allowed,true,signature,
            tokens.size(),provider.generator_capacity(),tokens);
        std::size_t prompt_tokens=0;
        struct {int cached_prompt_tokens=-1;uint64_t prefill_duration=0;} meta_info;
#include "production_prefill.inc"
        require(prefix_cache_.checkpoint_size()==tokens.size(),"checkpoint commit count differs");
        require(token_history==tokens,"checkpoint committed wrong tokens");
        return {{"reason",cache_decision.reason},{"hit",cache_decision.reuse},
                {"cached",meta_info.cached_prompt_tokens},{"prompt",prompt_tokens},
                {"new",cache_decision.newly_prefilled_tokens},
                {"disabled",prefix_cache_.disabled_for_session()}};
    }
};

int main(int argc,char** argv) {
    if(argc!=3)return 2;
    OgaHandle handle;
    json report={{"cases",json::array()},{"faults","test-injected; real OGA rewind/append/rebuild and generation"}};
    try {
        std::ifstream file(std::string(argv[1])+"/genai_config.json");
        const auto original=json::parse(file);
        const auto overlay=build_session_overlay(original,{{"offload_policy","strict_gpu"}},false,"");
        const auto effective_config=build_effective_config(original,{{"offload_policy","strict_gpu"}});
        ScopedConfigDirectory config_directory(argv[1],effective_config);
        auto config=OgaConfig::Create(config_directory.path().string().c_str());
        report["overlay"]=overlay;
        auto model=OgaModel::Create(*config);
        auto tokenizer=OgaTokenizer::Create(*model);
        auto encode=[&](const char* text){auto seq=OgaSequences::Create();tokenizer->Encode(text,*seq);return std::vector<int>(seq->SequenceData(0),seq->SequenceData(0)+seq->SequenceCount(0));};
        const auto a=encode("The first five prime numbers are");
        auto b=a;const auto suffix=encode(" 2, 3, 5. Continue the sequence:");b.insert(b.end(),suffix.begin(),suffix.end());
        std::vector<int> expected;
        { Provider reference(*model);reference.begin_turn(b);expected=reference.generate(); }
        report["referenceTokens"]=expected;
        report["promptA"]=a;report["promptB"]=b;
        { Provider fresh_repeat(*model);fresh_repeat.begin_turn(b);
          report["freshRepeatTokens"]=fresh_repeat.generate(); }
        require(report["freshRepeatTokens"]==expected,"fresh reference is not repeatable");
        for(const std::string fault:{"none","before-rewind","after-rewind","partial-append","after-append","rebuild-fails"}) {
            report["activeFault"]=fault;
            Harness test(*model);test.insert(a,false);test.provider.generate();
            test.provider.fault=fault;
            json item={{"fault",fault}};
            bool failed=false;
            try {item["result"]=test.insert(b);} catch(const std::exception& e){failed=true;item["error"]=e.what();}
            require(failed==(fault=="rebuild-fails"),"unexpected recovery success/failure");
            if(failed) {
                require(!test.prefix_cache_.has_checkpoint(),"failed rebuild retained checkpoint");
                require(test.token_history==a,"failed rebuild committed new history");
                require(test.prefix_cache_.disabled_for_session(),"failed rebuild allowed reuse");
            } else {
                item["tokens"]=test.provider.generate();
                item["events"]=test.provider.events;report["lastCase"]=item;
                require(item["tokens"]==expected,"recovered output differs from fresh reference");
                const auto& r=item["result"];
                require(r["cached"].get<std::size_t>()+r["new"].get<std::size_t>()==b.size(),"accounting sum differs");
                require(r["hit"]==(fault=="none"),"fallback incorrectly reported a hit");
                if(fault!="none")require(r["cached"]==0 && r["reason"]=="provider_exception","fallback accounting not fresh");
            }
            auto next=test.insert(b);
            item["next"]=next;
            item["nextTokens"]=test.provider.generate();
            item["events"]=test.provider.events;report["lastCase"]=item;
            if(fault!="none")require(next["reason"]=="session_disabled" && next["cached"]==0,"next request reused after provider failure");
            else require(next["hit"]==false && next["cached"]==0 &&
                         next["new"]==b.size() && next["reason"]=="empty_suffix" &&
                         next["disabled"]==false,"identical prompt did not rebuild truthfully");
            require(item["nextTokens"]==expected,"next request differs from fresh reference");
            require(fault=="none" || test.provider.fired,"fault was never injected");
            item["next"]=next;item["events"]=test.provider.events;item["status"]="PASS";
            report["cases"].push_back(item);
        }
        // Exhaust the generator's actual capacity, then restore and append into
        // the final free position. This catches a stale IsDone state on reuse.
        report["activeFault"]="capacity-exhaustion";
        const auto repeated=encode(" x");
        require(!repeated.empty(),"capacity fixture has no tokens");
        std::vector<int> near(1022,repeated.back());
        auto extended_prompt=near;extended_prompt.push_back(repeated.back());
        std::vector<int> near_expected;
        { Provider near_reference(*model);near_reference.begin_turn(extended_prompt);
          near_expected=near_reference.generate(); }
        require(near_expected.size()==1,"boundary reference must generate one token");
        Harness exhausted(*model);exhausted.insert(near,false);
        const auto first=exhausted.provider.generate();
        require(first.size()==2 && exhausted.provider.generator->IsDone(),
                "fixture did not exhaust the real generator capacity");
        auto extended=exhausted.insert(extended_prompt);
        require(extended["hit"]==true && extended["cached"]==1022 && extended["new"]==1,
                "exhausted generator extension did not reuse the correct prefix");
        require(exhausted.provider.generate()==near_expected,"exhausted generator extension differs from fresh");
        auto repeated_result=exhausted.insert(extended_prompt);
        require(repeated_result["hit"]==false && repeated_result["cached"]==0 &&
                repeated_result["new"]==extended_prompt.size() &&
                repeated_result["reason"]=="empty_suffix" && repeated_result["disabled"]==false,
                "exhausted identical prompt did not rebuild truthfully");
        require(exhausted.provider.generate()==near_expected,"exhausted fully cached output differs from fresh");
        report["cases"].push_back({{"fault","capacity-exhaustion"},{"status","PASS"},
            {"extended",extended},{"repeated",repeated_result},{"events",exhausted.provider.events}});
        report["status"]="PASS";
    }catch(const std::exception& e){report["status"]="FAIL";report["error"]=e.what();}
    std::ofstream(argv[2])<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<std::endl;return report["status"]=="PASS"?0:1;
}
