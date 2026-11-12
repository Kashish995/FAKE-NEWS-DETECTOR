import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from src.scraper import scrape_article
from src.fact_checker import query_fact_checking_pipeline
from src.classifier import analyze_four_components

test_cases = [
    {
        "id": "TEST-1",
        "name": "Genuine The Hindu Article (User's URL)",
        "type": "URL",
        "input": "https://www.thehindu.com/incoming/delhi-hc-directs-mcd-cm-to-set-up-panel-to-monitor-unsafe-buildings-during-grap/article71683027.ece"
    },
    {
        "id": "TEST-2",
        "name": "Genuine BBC News Headline & Body",
        "type": "TEXT",
        "domain": "bbc.com",
        "headline": "UK inflation falls to 1.7% in September, official figures show",
        "input": "UK inflation fell sharply to 1.7% in September, driven down by cheaper air fares and petrol prices, official figures show. It means the rate of price rises is now below the Bank of England's 2% target for the first time in more than three years. The lower-than-expected figure reinforces expectations that the Bank will cut interest rates at its next meeting in November. Economists said the slowdown in the pace of price increases would be welcomed by households, although higher energy bills from October are expected to nudge inflation back up slightly."
    },
    {
        "id": "TEST-3",
        "name": "Genuine Reuters News Headline & Body",
        "type": "TEXT",
        "domain": "reuters.com",
        "headline": "US Federal Reserve cuts interest rates by half percentage point",
        "input": "The Federal Reserve kicked off an expected series of interest rate cuts on Wednesday with a larger-than-usual half-percentage-point reduction that Fed Chair Jerome Powell said was meant to show policymakers' commitment to sustaining a low unemployment rate now that inflation has eased. Policymakers see the Fed's benchmark interest rate falling by another half percentage point by the end of this year, another full percentage point in 2025, and by a final half percentage point in 2026 to end in a 2.75%-3.00% range."
    },
    {
        "id": "TEST-4",
        "name": "Fabricated News (Leaked Documents / Aging Cure)",
        "type": "TEXT",
        "domain": "unknown-blog.com",
        "headline": "Secret documents prove pharmaceutical companies hide cure for aging",
        "input": "Secret documents leaked from government archives prove that scientists have developed a cure for cellular aging, but major pharmaceutical corporations are colluding with politicians to hide it from the public to protect their trillion-dollar healthcare profits. Whistleblowers have confirmed human trials were suppressed."
    },
    {
        "id": "TEST-5",
        "name": "Sensational Headline & Claim (5G Microchips)",
        "type": "TEXT",
        "domain": "social-forward.xyz",
        "headline": "COVID-19 vaccines contain 5G tracking microchips to control the population",
        "input": "Breaking bombshell: COVID-19 vaccines contain 5G tracking microchips designed to monitor and control human neural patterns worldwide. Spread this emergency alert before big tech scrubs the video proof!"
    },
    {
        "id": "TEST-6",
        "name": "Incomplete Extraction / Paywalled Blocked Link",
        "type": "URL",
        "input": "https://httpstat.us/403"
    },
    {
        "id": "TEST-7",
        "name": "Unrelated Regional Article (Testing News Relevance)",
        "type": "TEXT",
        "domain": "local-herald.in",
        "headline": "Gram panchayat in Koraput inaugurates minor irrigation canal for paddy cultivation",
        "input": "A local gram panchayat in Koraput district inaugurated a minor gravity flow irrigation canal on Tuesday morning. The community-funded channel is expected to irrigate twenty hectares of standing paddy crop across two neighboring villages. Local farmers participated in the ceremonial canal opening along with panchayat ward members."
    },
    {
        "id": "TEST-8",
        "name": "Very Short Text (10 words)",
        "type": "TEXT",
        "domain": "",
        "headline": "The city council postponed the municipal budget vote until Monday",
        "input": "The city council postponed the municipal budget vote until Monday."
    },
    {
        "id": "TEST-9",
        "name": "Empty Input Buffer",
        "type": "TEXT",
        "domain": "",
        "headline": "",
        "input": "   "
    }
]

print("=" * 80)
print("TRUTHLENS EMPIRICAL TEST SUITE (10 SCENARIOS)")
print("=" * 80)

results_summary = []

for tc in test_cases:
    print(f"\n[{tc['id']}] {tc['name']}")
    
    # 1. Extraction phase
    if tc["type"] == "URL":
        scraped = scrape_article(tc["input"])
        extraction_status = "Successful" if scraped["success"] else "Unsuccessful"
        extracted_length = scraped["word_count"]
        text_to_eval = scraped["full_text"]
        eval_domain = scraped["domain"]
        eval_headline = scraped["title"]
        extract_error = scraped.get("error")
    else:
        if not tc["input"].strip():
            extraction_status = "Blocked (Empty Input)"
            extracted_length = 0
            text_to_eval = ""
            eval_domain = ""
            eval_headline = ""
            extract_error = "Input buffer is empty."
        else:
            extraction_status = "Manual Text Input"
            extracted_length = len(tc["input"].split())
            text_to_eval = tc["input"]
            eval_domain = tc.get("domain", "")
            eval_headline = tc.get("headline", "")
            extract_error = None
            
    print(f"  • Extraction Status: {extraction_status}")
    print(f"  • Extracted Text Length: {extracted_length} words")
    
    # If extraction unsuccessful or empty, ML is NOT run!
    if extraction_status in ["Unsuccessful", "Blocked (Empty Input)"]:
        print(f"  • ML Prediction: BLOCKED (Not evaluated on invalid/incomplete input)")
        print(f"  • ML Confidence: N/A")
        print(f"  • Source Credibility: N/A")
        print(f"  • External Verification: N/A")
        print(f"  • Final Assessment: BLOCKED ({extract_error})")
        results_summary.append({
            "id": tc["id"],
            "name": tc["name"],
            "extraction_status": extraction_status,
            "length": extracted_length,
            "ml_pred": "BLOCKED",
            "ml_conf": "N/A",
            "source_cred": "N/A",
            "ext_verif": "N/A",
            "final_verdict": f"BLOCKED ({extract_error})"
        })
        continue
        
    # 2. Fact Checking & News Matching
    fc_res = query_fact_checking_pipeline(eval_headline, text_to_eval)
    
    # 3. Four-Component Analysis
    analysis = analyze_four_components(text_to_eval, eval_domain, fc_res)
    
    comp_a = analysis["component_a_ml"]
    comp_b = analysis["component_b_source"]
    comp_c = analysis["component_c_external"]
    comp_d = analysis["component_d_assessment"]
    
    ext_matches_desc = f"{len(comp_c['relevant_news'])} relevant articles" if comp_c['has_relevant_news'] else comp_c['status']
    if comp_c['has_debunk']:
        ext_matches_desc = f"ClaimReview Debunk ({len(comp_c['verified_claims'])})"
        
    print(f"  • ML Prediction: {comp_a['prediction_label']} ({comp_a['raw_prediction']})")
    print(f"  • ML Confidence: {comp_a['confidence_pct']}% (P(Real): {comp_a['probability_real']*100:.1f}%, P(Fake): {comp_a['probability_fake']*100:.1f}%)")
    print(f"  • Source Credibility: {comp_b['credibility_score']}/100 ({comp_b['source_name']})")
    print(f"  • External Verification: {ext_matches_desc}")
    print(f"  • Conflict Detected: {comp_d['has_conflict']}")
    print(f"  • Final Assessment: {comp_d['verdict'].upper()} ({comp_d['confidence_pct']}%)")
    print(f"  • Explanation: {comp_d['explanation'][:110]}...")
    
    results_summary.append({
        "id": tc["id"],
        "name": tc["name"],
        "extraction_status": extraction_status,
        "length": extracted_length,
        "ml_pred": comp_a['prediction_label'],
        "ml_conf": f"{comp_a['confidence_pct']}%",
        "source_cred": f"{comp_b['credibility_score']}/100",
        "ext_verif": ext_matches_desc,
        "final_verdict": f"{comp_d['verdict']} ({comp_d['confidence_pct']}%)"
    })

print("\n" + "=" * 80)
print("TEST SUMMARY COMPLETE")
print("=" * 80)
