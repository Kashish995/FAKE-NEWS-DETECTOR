import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import analyze_four_components
from src.fact_checker import query_fact_checking_pipeline

TEST_CASES = [
    # -------------------------------------------------------------
    # CATEGORY 1: 10 GENUINE NEWS ARTICLES
    # -------------------------------------------------------------
    {
        "id": "GEN-01",
        "category": "Genuine News",
        "publisher": "The Hindu",
        "headline": "Delhi HC directs MCD, CM to set up panel to monitor unsafe buildings during GRAP",
        "text": "A Bench of Chief Justice Manmohan and Justice Tushar Rao Gedela directed the Municipal Corporation of Delhi commissioner to set up a committee to monitor unsafe buildings during GRAP. The bench noted that unauthorized constructions pose serious safety risks to citizens in the capital.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-02",
        "category": "Genuine News",
        "publisher": "BBC News",
        "headline": "UK inflation falls to 1.7% in September, official figures show",
        "text": "UK inflation fell sharply to 1.7% in September, driven down by cheaper air fares and petrol prices, official figures show. It means the rate of price rises is now below the Bank of England's 2% target for the first time in more than three years, reinforcing expectations of an interest rate reduction.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-03",
        "category": "Genuine News",
        "publisher": "Reuters",
        "headline": "Federal Reserve cuts interest rates by half percentage point",
        "text": "The Federal Reserve kicked off an expected series of interest rate cuts on Wednesday with a larger-than-usual half-percentage-point reduction that Fed Chair Jerome Powell said was meant to show policymakers' commitment to sustaining low unemployment as inflation recedes.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-04",
        "category": "Genuine News",
        "publisher": "Associated Press",
        "headline": "Midwest winter blizzard closes schools and cancels hundreds of flights",
        "text": "A massive winter storm brought heavy snow and frigid temperatures across parts of the Midwest on Friday, closing schools, grounding hundreds of commercial flights, and cutting electrical power to thousands of homes.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-05",
        "category": "Genuine News",
        "publisher": "The Guardian",
        "headline": "UK electricity grid runs entirely without coal for record duration",
        "text": "Britain's electricity network has operated without coal power generation for a record consecutive period as renewable wind and solar energy supplies accounted for over half of total domestic generation.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-06",
        "category": "Genuine News",
        "publisher": "Indian Express",
        "headline": "ISRO launches Earth Observation Satellite EOS-08 from Sriharikota",
        "text": "The Indian Space Research Organisation successfully launched its Earth Observation Satellite EOS-08 aboard the Small Satellite Launch Vehicle from the Satish Dhawan Space Centre in Sriharikota, achieving intended orbital insertion.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-07",
        "category": "Genuine News",
        "publisher": "NPR",
        "headline": "James Webb Space Telescope detects water vapor on distant rocky exoplanet",
        "text": "Astronomers analyzing spectroscopic transmission data from NASA's James Webb Space Telescope have discovered definitive signatures of atmospheric water vapor on a rocky exoplanet orbiting an M-dwarf star.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-08",
        "category": "Genuine News",
        "publisher": "Bloomberg",
        "headline": "Semiconductor manufacturing firms expand global fabrication facilities",
        "text": "Major semiconductor manufacturing corporations have announced substantial capital investment commitments to construct advanced semiconductor fabrication facilities across Europe and North America to secure supply resilience.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-09",
        "category": "Genuine News",
        "publisher": "NDTV",
        "headline": "Supreme Court of India issues notice on environmental compliance guidelines",
        "text": "The Supreme Court of India on Monday issued formal notice to central and state pollution control boards regarding the strict implementation of environmental protection guidelines across major industrial zones.",
        "expected_label": "REAL NEWS"
    },
    {
        "id": "GEN-10",
        "category": "Genuine News",
        "publisher": "Nature",
        "headline": "Clinical trials indicate promising efficacy for next-generation malaria vaccine",
        "text": "Phase three randomized controlled clinical trial results published in peer-reviewed journals indicate that the novel candidate malaria vaccine provides sustained efficacy of up to seventy-eight percent against clinical episodes.",
        "expected_label": "REAL NEWS"
    },

    # -------------------------------------------------------------
    # CATEGORY 2: 10 FABRICATED NEWS STORIES
    # -------------------------------------------------------------
    {
        "id": "FAB-01",
        "category": "Fabricated News",
        "publisher": "Conspiracy Blog",
        "headline": "Secret documents prove pharmaceutical companies hide cure for aging",
        "text": "Secret documents leaked from government archives prove that scientists have developed a cure for cellular aging, but major pharmaceutical corporations are colluding with politicians to hide it from the public to protect their healthcare profits.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-02",
        "category": "Fabricated News",
        "publisher": "WhatsApp Forward",
        "headline": "COVID-19 vaccines contain 5G tracking microchips to control the population",
        "text": "Breaking bombshell: COVID-19 vaccines contain 5G tracking microchips designed to monitor and control human neural patterns worldwide. Spread this emergency alert before big tech deletes this proof!",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-03",
        "category": "Fabricated News",
        "publisher": "Fake Wire",
        "headline": "UNESCO declares Indian National Anthem the best anthem in the world",
        "text": "Great pride for India! UNESCO has officially announced and declared Jana Gana Mana as the best national anthem in the entire world. No other country has achieved this honor. Share with every Indian!",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-04",
        "category": "Fabricated News",
        "publisher": "Viral Blog",
        "headline": "Drinking boiled garlic water cures cancer and all tumors overnight",
        "text": "Medical doctors don't want you to know this simple miracle trick! Drinking freshly boiled garlic water completely dissolves malignant tumors and cures all forms of cancer in just 24 hours without chemotherapy.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-05",
        "category": "Fabricated News",
        "publisher": "Social Chain",
        "headline": "RBI to discontinue all 500 rupee currency notes starting midnight",
        "text": "Urgent alert: Reserve Bank of India has decided to ban and demonetize all five hundred rupee denomination banknotes starting midnight tonight. Exchange your cash immediately before it becomes worthless paper!",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-06",
        "category": "Fabricated News",
        "publisher": "Hoax Forum",
        "headline": "NASA admits moon landing was staged in Hollywood desert studio",
        "text": "In a shocking leaked confession, senior NASA officials have admitted that the 1969 Apollo moon landings were completely faked inside a Nevada film studio to win the cold war space race against the Soviet Union.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-07",
        "category": "Fabricated News",
        "publisher": "Clickbait Hub",
        "headline": "Living dinosaur discovered alive and thriving in dense Amazon rainforest",
        "text": "Explorers on an expedition into unexplored regions of the Amazon basin have captured photographic evidence of a living prehistoric triceratops grazing in a secluded jungle clearing.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-08",
        "category": "Fabricated News",
        "publisher": "Conspiracy XYZ",
        "headline": "Government secretly installs biometric tracking chips in paper currency",
        "text": "Whistleblowers reveal that central banks have embedded microscopic RFID and biometric tracking filaments in all paper currency bills to track the physical location of citizens everywhere.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-09",
        "category": "Fabricated News",
        "publisher": "Sensational Wire",
        "headline": "World Bank cancels all global personal debt and student loans",
        "text": "Unbelievable news: The World Bank and IMF have announced an executive jubilee order completely wiping out all credit card debt, mortgages, and student loans for all citizens worldwide effective immediately.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "FAB-10",
        "category": "Fabricated News",
        "publisher": "Rumor Daily",
        "headline": "Secret underground alien base discovered beneath Central Park",
        "text": "City excavation workers drilling near Central Park have accidentally breached an ancient underground extraterrestrial command base containing extraterrestrial technology and glowing crystalline computers.",
        "expected_label": "FAKE NEWS"
    },

    # -------------------------------------------------------------
    # CATEGORY 3: 5 MISLEADING / PARTIALLY TRUE CLAIMS
    # -------------------------------------------------------------
    {
        "id": "MIS-01",
        "category": "Misleading / Ambiguous",
        "publisher": "Health Forward",
        "headline": "Lemon and warm water completely prevents viral respiratory infections",
        "text": "Daily consumption of hot lemon water creates an alkaline biological environment that renders all respiratory viruses incapable of entering cellular membranes.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "MIS-02",
        "category": "Misleading / Ambiguous",
        "publisher": "Sky Watchers",
        "headline": "Mars will look as large as the full moon in the night sky tonight",
        "text": "A rare planetary alignment will bring Mars closer to Earth than ever before in human history. To the naked eye, the red planet will appear identical in diameter and brightness to the full moon.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "MIS-03",
        "category": "Misleading / Ambiguous",
        "publisher": "Eco Skeptics",
        "headline": "Electric vehicles generate more net lifecycle emissions than diesel trucks",
        "text": "Studies supposedly demonstrate that manufacturing lithium batteries produces far more carbon emissions than driving a diesel vehicle for twenty years, making electric vehicles worse for the planet.",
        "expected_label": "UNABLE TO VERIFY"
    },
    {
        "id": "MIS-04",
        "category": "Misleading / Ambiguous",
        "publisher": "Tradition Post",
        "headline": "Standing outdoors during solar eclipse causes permanent cellular damage",
        "text": "Traditional authorities warn that cosmic radiation peaks during solar eclipses, causing chromosomal damage to anyone who steps outdoors during the astrological event.",
        "expected_label": "FAKE NEWS"
    },
    {
        "id": "MIS-05",
        "category": "Misleading / Ambiguous",
        "publisher": "Economic Forward",
        "headline": "Gold prices predicted to reach one million dollars per ounce next month",
        "text": "Financial analysts warn that hyperinflation will crash all fiat currency systems within thirty days, pushing gold prices to over one million dollars per ounce overnight.",
        "expected_label": "UNABLE TO VERIFY"
    },

    # -------------------------------------------------------------
    # CATEGORY 4: 5 UNFAMILIAR / LOCAL PUBLISHERS
    # -------------------------------------------------------------
    {
        "id": "LOC-01",
        "category": "Unfamiliar / Local",
        "publisher": "Koraput Herald",
        "headline": "Gram panchayat in Koraput district inaugurates minor irrigation canal for paddy",
        "text": "A local gram panchayat in Koraput district inaugurated a minor gravity flow irrigation canal on Tuesday morning. The community-funded channel is expected to irrigate twenty hectares of standing paddy crop across two neighboring villages. Local farmers participated in the ceremonial canal opening along with panchayat ward members.",
        "expected_label": "UNABLE TO VERIFY"
    },
    {
        "id": "LOC-02",
        "category": "Unfamiliar / Local",
        "publisher": "Berhampur Tribune",
        "headline": "District sports meet inaugurated by collector at municipal stadium",
        "text": "The annual inter-school district athletic championship was inaugurated by the district collector at the municipal stadium on Thursday. Over four hundred student athletes from thirty secondary schools will compete across track and field events over the next three days.",
        "expected_label": "UNABLE TO VERIFY"
    },
    {
        "id": "LOC-03",
        "category": "Unfamiliar / Local",
        "publisher": "Western Gazette",
        "headline": "Town council approves funding allocation for public library refurbishment",
        "text": "The municipal town council voted unanimously on Wednesday evening to allocate capital reserve funding toward repairing the roof and upgrading digital computer facilities at the town central library.",
        "expected_label": "UNABLE TO VERIFY"
    },
    {
        "id": "LOC-04",
        "category": "Unfamiliar / Local",
        "publisher": "Kandhamal Voice",
        "headline": "Organic turmeric cooperative registers record harvest yields in Phulbani",
        "text": "Smallholder farmer collectives affiliated with the district organic spices cooperative reported an increase in organic turmeric procurement volumes following favorable monsoon distribution.",
        "expected_label": "UNABLE TO VERIFY"
    },
    {
        "id": "LOC-05",
        "category": "Unfamiliar / Local",
        "publisher": "Coastal Bulletin",
        "headline": "Harbor seal population shows modest recovery along protected bay sanctuary",
        "text": "Wildlife conservation biologists conducting seasonal aerial census surveys recorded a modest increase in the local harbor seal breeding colony within the protected coastal estuary marine reserve.",
        "expected_label": "UNABLE TO VERIFY"
    }
]

def run_generalization_benchmark():
    print("=" * 80)
    print("TRUTHLENS 30-CASE GENERALIZATION & REAL-WORLD BENCHMARK")
    print("=" * 80)
    
    summary_rows = []
    
    for tc in TEST_CASES:
        fc_res = query_fact_checking_pipeline(
            headline=tc["headline"],
            full_text=tc["text"],
            exclude_domain_or_publisher=tc["publisher"]
        )
        res = analyze_four_components(
            text=tc["text"],
            domain_or_url=tc["publisher"],
            fact_check_result=fc_res
        )
        
        comp_d = res["component_d_assessment"]
        comp_a = res["component_a_ml"]
        comp_c = res["component_c_external"]
        
        final_label = comp_d["final_label"]
        p_fake = comp_d["fake_probability_pct"]
        has_fc = len(comp_c["verified_claims"]) > 0
        has_news = len(comp_c["relevant_news"]) > 0
        
        ext_verdict = "Debunk Found" if has_fc else "Corroborated" if has_news else "No External Matches"
        
        summary_rows.append({
            "id": tc["id"],
            "category": tc["category"],
            "publisher": tc["publisher"],
            "expected": tc["expected_label"],
            "actual": final_label,
            "fake_prob": f"{p_fake}%",
            "ext_verif": ext_verdict,
            "explanation": comp_d["explanation"][:75] + "..."
        })
        
        print(f"[{tc['id']}] {tc['publisher']:<18} | Exp: {tc['expected_label']:<16} | Act: {final_label:<16} | P(Fake): {p_fake:5.1f}% | Ext: {ext_verdict}")

    print("\n" + "=" * 80)
    print("CATEGORY ACCURACY BREAKDOWN")
    print("=" * 80)
    
    cat_stats = {}
    for r in summary_rows:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        if r["actual"] == r["expected"]:
            cat_stats[cat]["correct"] += 1
            
    for cat, stats in cat_stats.items():
        pct = (stats["correct"] / stats["total"]) * 100
        print(f"  {cat:<25}: {stats['correct']}/{stats['total']} ({pct:.1f}% alignment with ground truth)")

if __name__ == "__main__":
    run_generalization_benchmark()
