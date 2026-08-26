import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# ----------------------------
# Sampling utilities
# ----------------------------

_EXAMPLE_PATH="/scratch/project_2017000/emotion-mining-pilot/data/example_notes_short.jsonl"


def choice_weighted(rng: np.random.Generator, values, probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return values[rng.choice(len(values), p=probs)]

def bernoulli(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)

def clipped_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    x = rng.normal(mean, sd)
    return float(np.clip(x, low, high))

def sample_age(rng: np.random.Generator) -> int:
    """
    Age distribution:
      18–64: 20%
      65–84: 70%
      85+: 10%
    Within-bin: uniform integer (simple, transparent).
    """
    bin_name = choice_weighted(
        rng,
        values=np.array(["18-64", "65-84", "85+"]),
        probs=np.array([0.20, 0.70, 0.1]),
    )
    if bin_name == "18–64":
        return int(rng.integers(18, 64))
    if bin_name == "65-84":
        return int(rng.integers(65, 84))
    return int(rng.integers(85, 101))  # cap at 100 for plausibility

def sample_sex(rng: np.random.Generator) -> str:
    # Sex: 49.5% men, 50.5% women
    return choice_weighted(rng, np.array(["male", "female"]), np.array([0.495, 0.505]))

def sample_mother_tongue(rng: np.random.Generator) -> str:
    # Finnish 84.1%, Swedish 5.1%, Sámi 0.04%, other 10.8%
    return choice_weighted(
        rng,
        np.array(["Finnish", "Swedish", "Sámi", "Other"]),
        np.array([0.841, 0.051, 0.0004, 0.1086]),
    )

def sample_citizenship(rng: np.random.Generator) -> str:
    # Finnish 92.7%, foreign 7.3%
    return choice_weighted(rng, np.array(["Finnish", "Foreign"]), np.array([0.927, 0.073]))

def sample_urban(rng: np.random.Generator) -> bool:
    # Urbanization: 85% urban
    return bernoulli(rng, 0.85)

def sample_household_size(rng: np.random.Generator) -> int:
    # Households: one-person 47%, two-person 32%, three or more persons 21%
    cat = choice_weighted(rng, np.array(["1", "2", "3+"]), np.array([0.47, 0.32, 0.21]))
    if cat == "1":
        return 1
    if cat == "2":
        return 2
    # for 3+, sample 3-6 with a mild decay
    return int(choice_weighted(rng, np.array([3, 4, 5, 6]), np.array([0.55, 0.25, 0.13, 0.07])))

def sample_education(rng: np.random.Generator, age: int) -> Optional[str]:
    # Education distribution for >=15y
    if age < 15:
        return None
    return choice_weighted(
        rng,
        np.array([
            "Basic education only",
            "Upper secondary",
            "Short-cycle tertiary",
            "Bachelor's degree",
            "Master's degree",
            "Doctorate",
        ]),
        np.array([0.249, 0.401, 0.085, 0.130, 0.112, 0.012]),
    )

def sample_employment(rng: np.random.Generator, age: int, sex: str) -> Optional[bool]:
    # Employment rate (ages 20–64): men 76.8%, women 76.6%
    if age > 64:
        return None
    p = 0.768 if sex == "male" else 0.766
    return bernoulli(rng, p)

def sample_smoking(rng: np.random.Generator, age: int, sex: str) -> bool:
    # Daily smoking: 10% overall (men 11%, women 9%)
    p = 0.11 if sex == "male" else 0.09
    # slightly lower in >=75 as a pragmatic cohort effect; keep simple
    if age >= 75:
        p *= 0.7
    return bernoulli(rng, p)

def sample_bmi(rng: np.random.Generator, age: int, sex: str) -> float:
    """
    Target: obesity (BMI>=30) ~24.5% of adults.
    Use a simple two-component mixture for adults; pediatric uses lower mean.
    """
    obese = bernoulli(rng, 0.245)
    if obese:
        bmi = clipped_normal(rng, mean=33.5, sd=3.2, low=28.0, high=50.0)
    else:
        bmi = clipped_normal(rng, mean=24.0, sd=3.4, low=18.0, high=28.0)
    return float(np.round(bmi, 1))

def sample_height_cm(rng: np.random.Generator, age: int, sex: str) -> float:
    mean = 179.0 if sex == "male" else 165.0
    return float(np.round(clipped_normal(rng, mean=mean, sd=7.0, low=145.0, high=205.0), 1))

def bmi_to_weight_kg(bmi: float, height_cm: float) -> float:
    h_m = height_cm / 100.0
    w = bmi * (h_m ** 2)
    return float(np.round(w, 1))

def sample_diabetes(rng: np.random.Generator, age: int, bmi: float) -> bool:
    # Calibrated loosely: higher with age and obesity; kept simple and plausible
    base = 0.03 + 0.0025 * max(age - 35, 0)  # increases with age
    if bmi >= 30:
        base *= 2.2
    return bernoulli(rng, float(np.clip(base, 0.01, 0.35)))

def sample_vaccination_status(rng: np.random.Generator, age: int) -> Dict[str, Any]:
    # Simple plausible flags (not intended as official coverage stats)
    return {
        "influenza_last_season": bernoulli(rng, 0.55 if age >= 65 else 0.25),
        "covid_primary_series": bernoulli(rng, 0.92 if age >= 50 else 0.85),
        "covid_booster_last_12m": bernoulli(rng, 0.70 if age >= 65 else 0.35),
        "pneumococcal": bernoulli(rng, 0.55 if age >= 65 else 0.10),
    }

def sample_allergies(rng: np.random.Generator) -> List[str]:
    # Keep sparse; many patients have none
    if bernoulli(rng, 0.70):
        return []
    options = np.array([
        "No known drug allergies (NKDA)",  # occasionally documented explicitly
        "Penicillin",
        "ACE inhibitor",
        "Cephalosporins",
        "Latex",
        "Iodinated contrast (urticaria)",
        "NSAIDs (angioedema)",
    ])
    k = int(choice_weighted(rng, np.array([1, 2]), np.array([0.85, 0.15])))
    picks = rng.choice(options[1:], size=k, replace=False)  # avoid NKDA mixed with true allergy
    return list(picks)

def sample_chronic_conditions(rng: np.random.Generator, age: int, sex: str, bmi: float, diabetes: bool) -> List[str]:
    conds = []
    # Hypertension more common with age and BMI
    p_htn = 0.18 + 0.006 * (age - 40 if age > 40 else 0)
    if bmi >= 30:
        p_htn *= 1.4
    if bernoulli(rng, float(np.clip(p_htn, 0.05, 0.70))):
        conds.append("Essential hypertension (I10)")
    if diabetes:
        conds.append("Type 2 diabetes mellitus (E11)" if age >= 18 else "Type 1 diabetes mellitus (E10)")
    # Dyslipidemia
    p_dys = 0.10 if age < 30 else 0.25
    if bernoulli(rng, p_dys):
        conds.append("Hypercholesterolemia (E78.0)")
    # CKD mild in older
    p_ckd = 0.01 + 0.002 * max(age - 60, 0)
    if bernoulli(rng, float(np.clip(p_ckd, 0.0, 0.18))):
        conds.append("Chronic kidney disease, stage 2–3 (N18)")
    #Schizophrenia
    p_sch = 0.015
    if bernoulli(rng, p_sch):
        conds.append("schizophrenia")
    return conds

def sample_family_status(rng: np.random.Generator, household_size: int, age: int) -> Dict[str, Any]:
    # "70% belongs to a family; 37% of families have minor children"
    belongs_to_family = bernoulli(rng, 0.70)
    has_minor_children = False
    if belongs_to_family and household_size >= 2:
        has_minor_children = bernoulli(rng, 0.37)
    return {"belongs_to_family": belongs_to_family, "has_minor_children": has_minor_children}

def sample_example(seed: int,example_path) -> str:
    rng = np.random.default_rng(seed)
    examples = []
    with open(example_path) as f:
        for line in f:
            d = json.loads(line)
            examples.append(d['text'])

    return rng.choice(examples)


def sample_adverse_event(rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    # use 50%
    preventable_adverse_events = [
        "Delayed medication administration",
        "Wrong dose",
        "Wrong medication",
        "Failure in equipment usage",
        "Equipment malfunction",
        "Drug interactions",
        "Documentation error",
        "Breakdown in communication",
        "Pressure ulcers",
        "Fall",
    ]
    random_adverse_events = [
        "Hospital-acquired infection",
        "IV infiltration",
        "Hospital aquired pneumonia",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to pain, withdrawal or anxiety",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to delirium",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to sleep distruption",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to close relative or visitor actions",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to trust",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to feeling not being heard",
        "Agitation (kiihtymys or levottomuus in Finnish) or aggression related to feeling not being not being valued",
        "Patient dissatisfaction based in perceived competency of health professional",
        "Patient dissatisfaction based in that they felt they were treated incorrectly",
        "Patient dissatisfaction based in perceived amount of caring",
        "Patient dissatisfaction based in perceived long waiting time",
        "Patient dissatisfaction based in perceived amount of caring",
        "Patient dissatisfaction based in perceived long discharge process",
        "Patient dissatisfaction based in feeling not being heard",
        "Patient dissatisfaction based in feeling not being valued"
    ]

    all_adverse_events = preventable_adverse_events + random_adverse_events
    if not bernoulli(rng, 0.75):
        return None

    # Sample number of adverse outcome combinations uniformly from the list
    n_events = rng.choice([1,2,3])
    all_events = []
    for i in range(n_events):
        e = rng.choice(all_adverse_events)
        all_events.append(e)
        all_adverse_events.remove(e)

    if len(all_events)>1:
        all_preventable = []
        for e in all_events:
            p = bernoulli(rng, 0.75 if e in preventable_adverse_events else 0.25)
            all_preventable.append(str(p))
        preventable = ", ".join(all_preventable)
        event = " and ".join(all_events)
    else:
        preventable = bernoulli(rng, 0.75 if all_events[0] in preventable_adverse_events else 0.25)
        event = all_events[0]


    # Sample severity uniformly from the list
    severities = ["mild", "moderate", "severe"]
    severity = rng.choice(severities)
    
    return {"event": event, "severity": severity, "preventable": preventable}

# ----------------------------
# EHR prompt templating with sampled fields
# ----------------------------

@dataclass
class PatientProfile:
    age: int
    sex: str
    mother_tongue: str
    citizenship: str
    urban: bool
    household_size: int
    education: Optional[str]
    employed: Optional[bool]
    smoking_daily: bool
    bmi: float
    height_cm: float
    weight_kg: float
    diabetes: bool
    vaccinations: Dict[str, Any]
    allergies: List[str]
    chronic_conditions: List[str]
    family_status: Dict[str, Any]
    adverse_event: Optional[Dict[str, Any]]

def sample_patient_profile(seed: Optional[int] = None) -> PatientProfile:
    rng = np.random.default_rng(seed)

    age = sample_age(rng)
    sex = sample_sex(rng)
    mother_tongue = sample_mother_tongue(rng)
    citizenship = sample_citizenship(rng)
    urban = sample_urban(rng)
    household_size = sample_household_size(rng)
    education = sample_education(rng, age)
    employed = sample_employment(rng, age, sex)

    bmi = sample_bmi(rng, age, sex)
    height_cm = sample_height_cm(rng, age, sex)
    weight_kg = bmi_to_weight_kg(bmi, height_cm)

    smoking_daily = sample_smoking(rng, age, sex)
    diabetes = sample_diabetes(rng, age, bmi)

    vaccinations = sample_vaccination_status(rng, age)
    allergies = sample_allergies(rng)
    chronic_conditions = sample_chronic_conditions(rng, age, sex, bmi, diabetes)
    family_status = sample_family_status(rng, household_size, age)

    adverse_event = sample_adverse_event(rng)

    return PatientProfile(
        age=age,
        sex=sex,
        mother_tongue=mother_tongue,
        citizenship=citizenship,
        urban=urban,
        household_size=household_size,
        education=education,
        employed=employed,
        smoking_daily=smoking_daily,
        bmi=bmi,
        height_cm=height_cm,
        weight_kg=weight_kg,
        diabetes=diabetes,
        vaccinations=vaccinations,
        allergies=allergies,
        chronic_conditions=chronic_conditions,
        family_status=family_status,
        adverse_event=adverse_event,
    )

def build_prompt_with_sampling(seed: Optional[int] = None, example_path = None) -> str:
    if example_path is None:
        example_path = _EXAMPLE_PATH
    example = sample_example(seed=seed,example_path=example_path)
    p = sample_patient_profile(seed=seed)

    # Apply sampling principle to "all possible fields that can be sampled" from the given text:
    # demographics, social profile, lifestyle risk factors, vaccination status, allergies, chronic diseases, height/weight.
    # (Clinical course, diagnoses, labs etc. are not specified in the user's prompt as distributions, so not sampled here.)

    employed_str = "employed" if p.employed is True else ("unemployed" if p.employed is False else "not in labor force (age-related)")
    urban_str = "urban" if p.urban else "rural"
    allergies_str = ", ".join(p.allergies) if p.allergies else "None reported"
    chronic_str = "; ".join(p.chronic_conditions) if p.chronic_conditions else "None documented"
    edu_str = p.education if p.education is not None else "N/A (pediatric)"
    family_str = f"belongs_to_family={p.family_status['belongs_to_family']}, has_minor_children={p.family_status['has_minor_children']}"

    vacc = p.vaccinations
    vacc_str = (
        f"Influenza last season: {vacc['influenza_last_season']}; "
        f"COVID primary series: {vacc['covid_primary_series']}; "
        f"COVID booster last 12m: {vacc['covid_booster_last_12m']}; "
        f"Pneumococcal: {vacc['pneumococcal']}"
    )

    adverse_str = "None" if p.adverse_event is None else (
        f"{p.adverse_event['event']} (severity={p.adverse_event['severity']}, preventable={p.adverse_event['preventable']})"
    )

    prompt = f"""
# Introduction

You are a medical language model whose task is to create three realistic progress notes in Finnish for cardiac patients treated in Finnish university hospitals.

Your task is to generate three progress notes that is consistent with:
- Current Finnish and European clinical guidelines
- Evidence-based practice
- The progess notes must be fully in Finnish
- Include the sampled adverse event to the progress notes
- Follow strictly the text format given in the note example (example given in section)
- Do not include any term-mentions from these insturctions such as "Käypä-hoito" or "Adverse event"
- Add mentions only from occured events e.g if patient do not have diabetes the note should not write "no diabetes".
- Use the Patient profile as context for the progress notes

The generated progress notes will be used to build and test AI systems, and they must be fully simulated, de-identified, and epidemiologically and clinically plausible.
## Patient profile

### Demographic and social profile (sampled)
- Age: {p.age} years
- Sex: {p.sex}
- Mother tongue: {p.mother_tongue}
- Citizenship: {p.citizenship}
- Area: {urban_str}
- Household size: {p.household_size}
- Family situation: {family_str}
- Education: {edu_str}
- Employment status (18–64 only): {employed_str}

### Lifestyle and cardiovascular risk factors (sampled)

- Daily smoking: {p.smoking_daily}
- BMI: {p.bmi} kg/m^2 (obesity if >=30)
- Height: {p.height_cm} cm
- Weight: {p.weight_kg} kg
- Diabetes: {p.diabetes}

### Other sampled background

- Vaccination status: {vacc_str}
- Allergies: {allergies_str}
- Chronic diseases: {chronic_str}

## Progress note formulation

The generated Progress notes should contain:
- Be in Finnish
- Daily assessments, clinical course, treatment updates
- Diagnostics
- Nursing notes
- Integrate nursing notes as part of the clinical foundation
- Include assessments, interventions, medication administration, fluid balance, cardiac monitoring, care planning
- Adverse event(s): {adverse_str}
Link the generated records to Finland’s epidemiological data:
    - Hospitalization rates: for adults, hospitalizations due to heart disease reflect the high prevalence of cardiovascular disease (32% of mortality)
    - Common cardiac conditions to simulate:
        - Ischemic heart disease / acute coronary syndrome
        - Heart failure
        - Arrhythmias (atrial fibrillation, supraventricular tachycardia)
        - Congenital heart disease (in pediatric cases)
        - Hypertension and related complications
All care must follow:
    - Finnish Käypä hoito guidelines
    - European Society of Cardiology (ESC) guidelines
    - WHO and NICE recommendations, where relevant
Examples:
    - STEMI: immediate reperfusion (preferably PCI), dual antiplatelet therapy, statins, ACE inhibitors/ARBs, beta-blockers
    - Heart failure: ACE inhibitors/ARNI, beta-blockers, MRAs, SGLT2 inhibitors, device therapy as needed
    - Arrhythmias: guideline-based anticoagulation in AF treatment, rate vs. rhythm control

## Style and formatting
All Progress notes must:
    - Mimic Finnish progress note formatting (short section headers, SOAP notes, standardized lab formats)
    - Ensure internal consistency: symptoms &rarr; physical exam &rarr; labs &rarr; diagnoses &rarr; treatments
    - Use a concise, clinically precise style
    - All data must be simulated
    - Contain exactly three progress notes
    - Follow strictly the text format given in the note example (date,time,heading,free text)
    - Insert the adverse event to note if specified &rarr; the adverse event may effect the mood of the patient, the content of progress notes, assessments, clinical course and treatment updates
### Example
<<<
{example}
>>>
### Output format

Output the Progress notes in the following format:

# Generated Progress notes

<the generated progress notes in Finnish>

""".strip()

    return prompt


generic_system_prompt="""You are a helpful and focused AI assistant.

    Always follow the user’s instructions carefully and complete the requested tasks to the best of your ability.
    Provide clear, accurate, and relevant responses that stay on topic.
    Do not include extra information or content that was not requested by the user.

    """

emotion_prompt_template = """
    # Introduction
    Input is a clinical note that can contain 0 or up to 34 distinct emotions.

    The emotions are classified as follows:
    1. Admiration: Admiration is a feeling of great liking and respect for a person or thing. 
    2. Adoration: Deep respect or affection; fervent admiration or love.
    3. Aesthetic Appreciation: The experience of beauty.
    4. Amusement: Humour excited by something comical or funny; entertainment or enjoyment derived from this.
    5. Anger: A strong feeling of displeasure, dissatisfaction, or annoyance, generally combined with antagonism or hostility towards a particular cause or object; the state of experiencing such feelings; wrath, rage, fury.
    6. Anxiety: Worry over the future or about something with an uncertain outcome; uneasy concern about a person, situation, etc.; a troubled state of mind arising from such worry or concern.
    7. Awe: The feeling of respect and amazement that you have when you are faced with something wonderful and often rather frightening.
    8. Awkwardness: Lack of skill or dexterity; clumsiness.
    9. Boredom: The state of being bored.
    10. Calmness: The state or quality of being calm; stillness, tranquillity, quietness. Freedom from agitation or disturbance.
    11. Confusion: The confounding or mistaking of one for another; failure to distinguish. Const. of (things), of one with another, between (things). It is not clear what the true situation is, especially because people believe different things.
    12. Contempt: A feeling of dislike or hostility towards a person or thing one regards as inferior, worthless, or despicable; an attitude expressive of such a feeling; (later) a complete lack of consideration or respect for a person or thing.
    13. Craving: An intense desire or longing.
    14. Disappointment: Deprivation or denial of something required, desired, or expected; spec. failure in the proper equipping of a store, or in the expected provision of goods, supplies, etc.
    15. Disgust: Strong repugnance, aversion, or repulsion excited by that which is loathsome or offensive, as a foul smell, disagreeable person or action, disappointed ambition, etc.; profound instinctive dislike or dissatisfaction.
    16. Empathic Pain: Have an experience of another's pain. Being able to understand what others feel, be it an emotion or a sensory state.
    17. Entrancement: The state of being filled with wonder and delight; enchantment. The condition of being put into a trance; hypnotization.
    18. Envy: The feeling you have when you wish you could have the same thing or quality that someone else has.
    19. Excitement: A person or thing that excites; stimulation or thrill
    20. Fear: The emotion of pain or uneasiness caused by the sense of impending danger, or by the prospect of some possible evil.
    21. Guilt: An unpleasant feeling of having committed wrong or failed in an obligation; a guilty feeling.
    22. Horror: A feeling of great shock, fear, and worry caused by something extremely unpleasant
    23. Interest: A feeling of particular concern for or curiosity about a person or thing; attention or consideration devoted to a person or thing; engagement with a subject, topic, etc. 
    24. Joy: A vivid emotion of pleasure arising from a sense of well-being or satisfaction; the feeling or state of being highly pleased or delighted; exultation of spirit; gladness, delight.
    25. Nostalgia: Is an affectionate feeling you have for the past, especially for a particularly happy time.
    26. Pride: A feeling of satisfaction that you have because you or people close to you have done something good or possess something good.
    27. Relief: Alleviation of or deliverance from distress, anxiety, or some other emotional burden; the feeling accompanying this; mental relaxation, release, or reassurance.
    28. Romance: Ardour or warmth of feeling in a love affair; love, esp. of an idealized or sentimental kind.
    29. Sadness: Feel unhappy, usually because something has happened that you do not like.
    30. Satisfaction: The state or quality of feeling satisfied or contented; (in later use chiefly) gratification, pleasure, or contentment caused by a fact, event, or state of things.
    31. Sexual Desire: A wish, need, or drive to seek out sexual objects or to engage in sexual activities.
    32. Surprise: To affect with the characteristic emotion caused by something unexpected; to excite to wonder by being unlooked-for.
    33. Sympathy: The sharing of another's emotions, esp of sorrow or anguish; pity; compassion.
    34. Triumph: The feeling of exultation and happiness derived from a victory or major achievement.

    # Instructions
    Extract the patient's emotions that the healthcare professional may have documented in the note. Use the categories given above.
    The note is given after the header "# Note".
    Be careful to extract only the patient's emotions, not those of the professional or a close relative.
    Verify your decision by applying logical principles. Justify your decision with a maximum of three sentences.
    Sometimes, a note may not contain any emotions. Answer then "No emotions". Notes may contain long lists of measurements, and they do not change the instruction.
    After the extraction, output a confidence score from 0.0 to 1.0 for the answer. Confidence 0.0 means the answer is unreliable, 0.5 means the answer is somewhat confident, and 1 means there is no possible error.
    ## Output format
    Output both justification and answer in the following format:

    **Justification:** justification for the answer

    **Confidence:** a score from 0.0 to 1.0

    **Answer:** emotion_1, emotion_2


    DO NOT PROVIDE REASONING OR ANY ADDITIONAL INFORMATION IN THE ANSWER OR CONFIDENCE SECTIONS!
    # Note
    {note}
    """


if __name__ == "__main__":
    # Example usage: deterministic prompt
    for i in range(10):
        print(build_prompt_with_sampling(seed=i+2))
