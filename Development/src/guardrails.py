import math
import re

import numpy as np
import torch

from config import DEBUG, GROUNDING_CHECK_MAX_DOCS
from src.latency_tracker import track_latency

_WORD_RE = re.compile(r"[a-zA-Zऀ-ॿ஀-௿]+")

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Reference set for the off-topic gate (see check_off_topic). The corpus
# (MSMARCO-XI) is open-domain general-knowledge QA, not a narrow business
# topic - there's no bounded "list of in-scope subjects" to enumerate. These
# span the actual breadth observed across this project's own benchmark runs
# (government/legal process, business/tax, geography/travel, food/culture,
# science/medicine, history, pop culture) rather than one narrow theme, plus
# a few Hindi/Gujarati anchors since the multilingual embedding space isn't
# perfectly language-symmetric (a real, measured gap - see the Aug 2026
# percentile-batch hedge-rate investigation). New reference queries can be
# added here if a genuine in-scope query starts getting misflagged.
_OFF_TOPIC_REFERENCE_QUERIES = [
    "How to apply for a passport",
    "What is a corporation",
    "income tax filing deadline",
    "voter ID card requirements",
    "how does GST registration work",
    "what is Goa famous for",
    "what is kaju katli",
    "what is barter trade",
    "tell me about forts in India",
    "historical sites near Chichen Itza",
    "describe the Udaipur lake palace",
    "what are New Orleans beignets",
    "what is trigger finger condition",
    "diesel fuel cost per gallon",
    "standard molar entropy in chemistry",
    "what is cornflour made from",
    "average high school teacher salary",
    "who created the Iron Man comic character",
    "what degree do you need to become a teacher",
    "death's head insignia military history",
    "what is a DBA business registration",
    "how is an S-Corp taxed",
    "beaches in Goa",
    "what is feni liquor made from",
    "GST tax credit claims process",
    "documents needed for a passport card",
    "Portuguese architecture in Goa",
    "bilateral trade vs multilateral trade",
    "significance of Isthmia in Greek mythology",
    "who was Harriet Tubman",
    # General-science anchor added after corpus-coverage testing confirmed
    # the corpus (post Aug 2026 MSMARCO-XI expansion) has genuinely correct
    # content for this topic specifically - NOT a blanket "let general trivia
    # through" change. Queries like "what causes rainbows" or "what is
    # compound interest" deliberately stay excluded: coverage testing showed
    # the corpus only has homonym-adjacent wrong content for those (e.g.
    # "compound interest" retrieves "equity interest" passages), so opening
    # the gate for them would trade a decline for a confidently wrong answer.
    # "why do leaves change color in autumn" was tried and reverted: static
    # coverage testing showed a genuinely relevant top-3 passage, but live
    # generation actually grounded on an unrelated "indicators of chemical
    # changes" passage instead and answered confidently wrong - the doc
    # ranking retrieval sees isn't always the doc generation ends up using.
    "how does the human heart pump blood",
    "पासपोर्ट के लिए आवेदन कैसे करें",
    "गोवा किस लिए प्रसिद्ध है",
    "મતદાર ID માટે શું જરૂરી છે",
    "કંપની એટલે શું",
    "ફેની શું છે",
    "કાજુ કતલી શું છે",
]

# Calibrated (see scripts/calibrate_off_topic_threshold.py) against the
# project's own real benchmark queries (must NOT trigger) vs deliberately
# off-topic ones - creative writing, pure computation, casual chat, meta
# questions about the assistant itself (must trigger). Sits between the two
# observed similarity distributions, not chosen by eye.
OFF_TOPIC_SIMILARITY_THRESHOLD = 0.499  # midpoint between max(off-topic)=0.4123 and
                                         # min(in-scope)=0.5857, per scripts/calibrate_off_topic_threshold.py

# Minimum-viable unsafe-input gate: obvious-intent phrase patterns, not a
# full classifier. This is a different failure mode than "ungrounded
# answer" (check_grounding) - it's "shouldn't have been processed as a real
# query at all," so it runs before retrieval/generation even starts.
_UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhow (?:do|can|to) i? ?(?:make|build|create) (?:a )?(?:bomb|explosive|weapon)",
        r"\bhow (?:do|can|to) i? ?(?:hurt|kill|harm) (?:myself|someone|others)",
        r"\b(?:suicide|self.?harm) (?:methods|instructions|how.?to)",
        r"\bhow (?:do|can|to) i? ?(?:synthesize|make) (?:illegal drugs|meth|cocaine)",
        r"\bchild (?:sexual|porn|abuse)",
    ]
]

# en/hi/gu were hand-verified early in the project (see _LOW_GROUNDING_MESSAGES
# comment below). The remaining 11 supported language codes were added later,
# machine-translated (not native-speaker-verified) rather than left falling
# back to English - a translated refusal is more useful to a non-English
# speaker than a correct-but-unreadable English one, but flag this if a
# native speaker is available to spot-check before the deadline.
_UNSAFE_MESSAGES = {
    "en": "I can't help with that request.",
    "hi": "मैं इस अनुरोध में मदद नहीं कर सकता।",
    "gu": "હું આ વિનંતીમાં મદદ કરી શકતો નથી.",
    "mr": "मी या विनंतीत मदत करू शकत नाही.",
    "ta": "இந்த கோரிக்கைக்கு நான் உதவ முடியாது.",
    "te": "ఈ అభ్యర్థనలో నేను సహాయం చేయలేను.",
    "kn": "ಈ ವಿನಂತಿಯಲ್ಲಿ ನಾನು ಸಹಾಯ ಮಾಡಲಾಗುವುದಿಲ್ಲ.",
    "ml": "ഈ അഭ്യർത്ഥനയിൽ എനിക്ക് സഹായിക്കാൻ കഴിയില്ല.",
    "bn": "আমি এই অনুরোধে সাহায্য করতে পারছি না।",
    "pa": "ਮੈਂ ਇਸ ਬੇਨਤੀ ਵਿੱਚ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ।",
    "or": "ମୁଁ ଏହି ଅନୁରୋଧରେ ସାହାଯ୍ୟ କରିପାରିବି ନାହିଁ।",
    "ur": "میں اس درخواست میں مدد نہیں کر سکتا۔",
    "ne": "म यो अनुरोधमा मद्दत गर्न सक्दिन।",
    "as": "মই এই অনুৰোধত সহায় কৰিব নোৱাৰোঁ।",
}

_OFF_TOPIC_MESSAGES = {
    "en": "That looks outside what this system can help with - it's built to answer questions grounded in its own document corpus. Try rephrasing as a factual question, or ask about a related topic.",
    "hi": "यह इस सिस्टम की सहायता क्षेत्र से बाहर लगता है - यह अपने दस्तावेज़ों पर आधारित प्रश्नों के उत्तर देने के लिए बनाया गया है। कृपया इसे एक तथ्यात्मक प्रश्न के रूप में दोबारा पूछें।",
    "gu": "આ આ સિસ્ટમની મદદના ક્ષેત્રની બહાર લાગે છે - તે તેના પોતાના દસ્તાવેજો પર આધારિત પ્રશ્નોના જવાબ આપવા માટે બનાવવામાં આવ્યું છે. કૃપા કરીને તેને તથ્યાત્મક પ્રશ્ન તરીકે ફરીથી પૂછો.",
    "mr": "हे या प्रणालीच्या मदतीच्या कक्षेबाहेर दिसते - ही प्रणाली स्वतःच्या दस्तऐवजांवर आधारित प्रश्नांची उत्तरे देण्यासाठी तयार केली आहे. कृपया हा प्रश्न वस्तुस्थितीवर आधारित प्रश्न म्हणून पुन्हा विचारा, किंवा संबंधित विषयाबद्दल विचारा.",
    "ta": "இது இந்த அமைப்பு உதவக்கூடிய வரம்புக்கு வெளியே உள்ளது போல் தெரிகிறது - இது அதன் சொந்த ஆவணத் தொகுப்பை அடிப்படையாகக் கொண்ட கேள்விகளுக்கு பதிலளிக்க வடிவமைக்கப்பட்டுள்ளது. இதை ஒரு உண்மைசார் கேள்வியாக மீண்டும் கேளுங்கள், அல்லது தொடர்புடைய தலைப்பைப் பற்றி கேளுங்கள்.",
    "te": "ఇది ఈ వ్యవస్థ సహాయం చేయగల పరిధికి బయట ఉన్నట్లు కనిపిస్తోంది - ఇది దాని స్వంత పత్రాల సమాహారంపై ఆధారపడిన ప్రశ్నలకు సమాధానం ఇవ్వడానికి రూపొందించబడింది. దయచేసి దీన్ని వాస్తవిక ప్రశ్నగా మళ్లీ అడగండి, లేదా సంబంధిత అంశం గురించి అడగండి.",
    "kn": "ಇದು ಈ ವ್ಯವಸ್ಥೆ ಸಹಾಯ ಮಾಡಬಹುದಾದ ವ್ಯಾಪ್ತಿಯ ಹೊರಗಿದೆ ಎಂದು ತೋರುತ್ತದೆ - ಇದು ತನ್ನದೇ ಆದ ದಾಖಲೆಗಳ ಸಂಗ್ರಹವನ್ನು ಆಧರಿಸಿ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಲು ವಿನ್ಯಾಸಗೊಳಿಸಲಾಗಿದೆ. ದಯವಿಟ್ಟು ಇದನ್ನು ವಾಸ್ತವಿಕ ಪ್ರಶ್ನೆಯಾಗಿ ಮರುರೂಪಿಸಿ ಕೇಳಿ, ಅಥವಾ ಸಂಬಂಧಿತ ವಿಷಯದ ಬಗ್ಗೆ ಕೇಳಿ.",
    "ml": "ഇത് ഈ സിസ്റ്റത്തിന് സഹായിക്കാൻ കഴിയുന്ന പരിധിക്ക് പുറത്താണെന്ന് തോന്നുന്നു - ഇത് സ്വന്തം രേഖാ ശേഖരത്തെ അടിസ്ഥാനമാക്കിയുള്ള ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകാൻ രൂപകൽപ്പന ചെയ്തതാണ്. ദയവായി ഇത് ഒരു വസ്തുതാപരമായ ചോദ്യമായി വീണ്ടും ചോദിക്കുക, അല്ലെങ്കിൽ ബന്ധപ്പെട്ട വിഷയത്തെക്കുറിച്ച് ചോദിക്കുക.",
    "bn": "এটি এই সিস্টেম যে বিষয়ে সাহায্য করতে পারে তার বাইরে বলে মনে হচ্ছে - এটি নিজস্ব নথির সংগ্রহের উপর ভিত্তি করে প্রশ্নের উত্তর দেওয়ার জন্য তৈরি করা হয়েছে। অনুগ্রহ করে এটিকে একটি বাস্তবভিত্তিক প্রশ্ন হিসেবে আবার জিজ্ঞাসা করুন, অথবা সম্পর্কিত বিষয় সম্পর্কে জিজ্ঞাসা করুন।",
    "pa": "ਇਹ ਇਸ ਸਿਸਟਮ ਦੀ ਮਦਦ ਦੇ ਦਾਇਰੇ ਤੋਂ ਬਾਹਰ ਜਾਪਦਾ ਹੈ - ਇਹ ਆਪਣੇ ਦਸਤਾਵੇਜ਼ਾਂ ਦੇ ਸੰਗ੍ਰਹਿ 'ਤੇ ਆਧਾਰਿਤ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਦੇਣ ਲਈ ਬਣਾਇਆ ਗਿਆ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ ਇੱਕ ਤੱਥਾਤਮਕ ਸਵਾਲ ਵਜੋਂ ਦੁਬਾਰਾ ਪੁੱਛੋ, ਜਾਂ ਸੰਬੰਧਿਤ ਵਿਸ਼ੇ ਬਾਰੇ ਪੁੱਛੋ।",
    "or": "ଏହା ଏହି ସିଷ୍ଟମ ସାହାଯ୍ୟ କରିପାରୁଥିବା ପରିସର ବାହାରେ ଥିବା ପରି ମନେହୁଏ - ଏହା ନିଜର ଦଲିଲ ସଂଗ୍ରହ ଉପରେ ଆଧାରିତ ପ୍ରଶ୍ନଗୁଡ଼ିକର ଉତ୍ତର ଦେବା ପାଇଁ ତିଆରି ହୋଇଛି। ଦୟାକରି ଏହାକୁ ଏକ ତଥ୍ୟଭିତ୍ତିକ ପ୍ରଶ୍ନ ଭାବରେ ପୁଣି ପଚାରନ୍ତୁ, କିମ୍ବା ସମ୍ବନ୍ଧିତ ବିଷୟ ବିଷୟରେ ପଚାରନ୍ତୁ।",
    "ur": "یہ اس نظام کی مدد کے دائرے سے باہر لگتا ہے - یہ اپنے دستاویزات کے مجموعے پر مبنی سوالات کے جواب دینے کے لیے بنایا گیا ہے۔ براہ کرم اسے ایک حقیقت پر مبنی سوال کے طور پر دوبارہ پوچھیں، یا کسی متعلقہ موضوع کے بارے میں پوچھیں۔",
    "ne": "यो यस प्रणालीले मद्दत गर्न सक्ने दायराभन्दा बाहिर देखिन्छ - यो आफ्नै कागजातहरूको संग्रहमा आधारित प्रश्नहरूको जवाफ दिन बनाइएको हो। कृपया यसलाई तथ्यपरक प्रश्नको रूपमा फेरि सोध्नुहोस्, वा सम्बन्धित विषयको बारेमा सोध्नुहोस्।",
    "as": "এইটো এই প্ৰণালীয়ে সহায় কৰিব পৰা পৰিসৰৰ বাহিৰত থকা যেন লাগিছে - এইটো নিজৰ নথিৰ সংগ্ৰহৰ ওপৰত ভিত্তি কৰি প্ৰশ্নৰ উত্তৰ দিবলৈ তৈয়াৰ কৰা হৈছে। অনুগ্ৰহ কৰি ইয়াক তথ্যমূলক প্ৰশ্ন হিচাপে পুনৰ সোধক, বা সম্পৰ্কিত বিষয়ৰ বিষয়ে সোধক।",
}


def build_unsafe_response(language: str = "en") -> str:
    return _UNSAFE_MESSAGES.get(language, _UNSAFE_MESSAGES["en"])


def build_off_topic_response(language: str = "en") -> str:
    return _OFF_TOPIC_MESSAGES.get(language, _OFF_TOPIC_MESSAGES["en"])


def check_unsafe(query_text: str) -> bool:
    """True if the query matches an obvious-intent unsafe pattern. Runs
    ahead of everything else - retrieval/generation cost shouldn't be spent
    on something that should never have been processed as a real query."""
    if not query_text:
        return False
    return any(p.search(query_text) for p in _UNSAFE_PATTERNS)

# Refusal/non-answer phrases that indicate the model didn't actually answer.
_REFUSAL_PHRASES = [
    "i don't know", "i do not know", "unknown", "not available",
    "i could not find", "i cannot find", "no information",
]

# Hedge message for the confident-hallucination case: the answer passed
# validate_answer() (not empty, not a refusal phrase) but scored below
# ANSWER_CACHE_MIN_GROUNDING - the retrieved context doesn't actually
# support it. Distinct from validate_answer's own fallback, which covers
# the model explicitly saying it doesn't know. en/hi/gu were hand-verified
# early in the project; the remaining 11 language codes are machine-
# translated (not native-speaker-verified) - see _UNSAFE_MESSAGES comment.
_LOW_GROUNDING_MESSAGES = {
    "en": "I couldn't find information in the available sources that confidently answers this question. Could you rephrase it, or ask about a related topic?",
    "hi": "मुझे उपलब्ध जानकारी में इस प्रश्न का विश्वसनीय उत्तर नहीं मिला। कृपया प्रश्न को दोबारा पूछें या किसी संबंधित विषय के बारे में पूछें।",
    "gu": "મને ઉપલબ્ધ માહિતીમાં આ પ્રશ્નનો વિશ્વસનીય જવાબ મળ્યો નથી. કૃપા કરીને પ્રશ્નને ફરીથી પૂછો અથવા સંબંધિત વિષય વિશે પૂછો.",
    "mr": "मला उपलब्ध माहितीमध्ये या प्रश्नाचे विश्वासार्ह उत्तर सापडले नाही. कृपया प्रश्न पुन्हा विचारा किंवा संबंधित विषयाबद्दल विचारा.",
    "ta": "இந்த கேள்விக்கு நம்பிக்கையுடன் பதிலளிக்கும் தகவலை கிடைக்கக்கூடிய ஆதாரங்களில் என்னால் கண்டுபிடிக்க முடியவில்லை. தயவுசெய்து கேள்வியை மீண்டும் கேளுங்கள் அல்லது தொடர்புடைய தலைப்பைப் பற்றி கேளுங்கள்.",
    "te": "అందుబాటులో ఉన్న మూలాల్లో ఈ ప్రశ్నకు నమ్మకంగా సమాధానం ఇచ్చే సమాచారం నాకు కనుగొనబడలేదు. దయచేసి ప్రశ్నను మళ్లీ అడగండి లేదా సంబంధిత అంశం గురించి అడగండి.",
    "kn": "ಲಭ್ಯವಿರುವ ಮೂಲಗಳಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ವಿಶ್ವಾಸಾರ್ಹವಾಗಿ ಉತ್ತರಿಸುವ ಮಾಹಿತಿ ನನಗೆ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಪ್ರಶ್ನೆಯನ್ನು ಮರುರೂಪಿಸಿ ಅಥವಾ ಸಂಬಂಧಿತ ವಿಷಯದ ಬಗ್ಗೆ ಕೇಳಿ.",
    "ml": "ലഭ്യമായ ഉറവിടങ്ങളിൽ ഈ ചോദ്യത്തിന് ആത്മവിശ്വാസത്തോടെ ഉത്തരം നൽകുന്ന വിവരങ്ങൾ എനിക്ക് കണ്ടെത്താനായില്ല. ദയവായി ചോദ്യം വീണ്ടും ചോദിക്കുക അല്ലെങ്കിൽ ബന്ധപ്പെട്ട വിഷയത്തെക്കുറിച്ച് ചോദിക്കുക.",
    "bn": "উপলব্ধ উৎসগুলিতে আমি এই প্রশ্নের আত্মবিশ্বাসের সাথে উত্তর দেওয়ার মতো তথ্য খুঁজে পাইনি। অনুগ্রহ করে প্রশ্নটি আবার জিজ্ঞাসা করুন অথবা সম্পর্কিত বিষয় সম্পর্কে জিজ্ঞাসা করুন।",
    "pa": "ਉਪਲਬਧ ਸਰੋਤਾਂ ਵਿੱਚ ਮੈਨੂੰ ਇਸ ਸਵਾਲ ਦਾ ਭਰੋਸੇਯੋਗ ਜਵਾਬ ਦੇਣ ਵਾਲੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਸਵਾਲ ਦੁਬਾਰਾ ਪੁੱਛੋ ਜਾਂ ਸੰਬੰਧਿਤ ਵਿਸ਼ੇ ਬਾਰੇ ਪੁੱਛੋ।",
    "or": "ଉପଲବ୍ଧ ଉତ୍ସଗୁଡ଼ିକରେ ମୁଁ ଏହି ପ୍ରଶ୍ନର ଭରସାଯୋଗ୍ୟ ଉତ୍ତର ଦେଉଥିବା ସୂଚନା ପାଇ ପାରିଲି ନାହିଁ। ଦୟାକରି ପ୍ରଶ୍ନକୁ ପୁଣି ପଚାରନ୍ତୁ କିମ୍ବା ସମ୍ବନ୍ଧିତ ବିଷୟ ବିଷୟରେ ପଚାରନ୍ତୁ।",
    "ur": "دستیاب ذرائع میں مجھے اس سوال کا اعتماد کے ساتھ جواب دینے والی معلومات نہیں ملیں۔ براہ کرم سوال دوبارہ پوچھیں یا کسی متعلقہ موضوع کے بارے میں پوچھیں۔",
    "ne": "उपलब्ध स्रोतहरूमा मैले यो प्रश्नको भरपर्दो जवाफ दिने जानकारी फेला पारिन। कृपया प्रश्न फेरि सोध्नुहोस् वा सम्बन्धित विषयको बारेमा सोध्नुहोस्।",
    "as": "উপলব্ধ উৎসত মই এই প্ৰশ্নৰ বিশ্বাসযোগ্য উত্তৰ দিয়া তথ্য বিচাৰি পোৱা নাই। অনুগ্ৰহ কৰি প্ৰশ্নটো পুনৰ সোধক বা সম্পৰ্কিত বিষয়ৰ বিষয়ে সোধক।",
}


def build_low_grounding_response(language: str = "en") -> str:
    """Language-aware hedge message - reuses the language code already
    threaded through the pipeline (target_language/request.language)
    rather than re-detecting it from the query text."""
    return _LOW_GROUNDING_MESSAGES.get(language, _LOW_GROUNDING_MESSAGES["en"])


def _tokenize(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Guardrails:
    """Grounding and answer-quality checks.

    Grounding uses a multilingual cross-encoder (trained on mMARCO - the same
    corpus family as our MSMARCO-XI index) to score how well the answer is
    supported by each retrieved passage. Word overlap breaks down badly for
    morphologically rich languages (Hindi case suffixes, etc.) and for
    paraphrased LLM output, so it's kept only as a fallback if the
    cross-encoder can't be loaded.
    """

    def __init__(self, embedding_service=None):
        self.cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
            # Same lazy-device quirk as SentenceTransformer: __init__ only
            # records _target_device, the actual .to(device) transfer
            # happens on first predict() otherwise - see embedding_service.py.
            self.cross_encoder.model.to(device)
            if DEBUG:
                print(f"[Guardrails] Loaded cross-encoder: {CROSS_ENCODER_MODEL} on {device}")
        except Exception as e:
            if DEBUG:
                print(f"[Guardrails] Cross-encoder unavailable ({e}), falling back to word overlap")

        # Off-topic gate needs the same embedding model/space the rest of the
        # pipeline already uses (so it can score the caller's precomputed
        # query_embedding directly, no re-embedding). Reuses the shared
        # EmbeddingService instance rather than loading a second copy of the
        # model - optional so this class stays usable standalone (see
        # __main__ below) without the full service graph.
        self._reference_embeddings = None
        if embedding_service is not None:
            self._reference_embeddings = np.stack([
                embedding_service.embed_query(q) for q in _OFF_TOPIC_REFERENCE_QUERIES
            ])

    def check_off_topic(self, query_embedding) -> tuple:
        """Returns (is_off_topic, max_similarity). Compares the query's
        embedding against a fixed reference set via cosine similarity - if
        it isn't close to anything the corpus is actually about, decline
        before spending retrieval/generation cost on it. Falls back to
        never flagging (False, 1.0) if no embedding_service was provided at
        construction time, rather than blocking every query."""
        if self._reference_embeddings is None:
            return False, 1.0
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        refs = self._reference_embeddings / (
            np.linalg.norm(self._reference_embeddings, axis=1, keepdims=True) + 1e-9
        )
        similarities = refs @ q
        max_sim = float(np.max(similarities))
        return max_sim < OFF_TOPIC_SIMILARITY_THRESHOLD, max_sim

    @track_latency("grounding_check")
    def check_grounding(self, answer: str, retrieved_docs: list) -> float:
        """Score how well the answer is supported by the retrieved documents.

        Only scores the top GROUNDING_CHECK_MAX_DOCS of retrieved_docs (which
        arrives already ranked, most-relevant first, from merge_and_rank) -
        the score below is always max(per-doc scores), so anything past the
        top few is CPU time spent without changing the result on the
        overwhelming majority of queries. See config.py's GROUNDING_CHECK_MAX_DOCS
        docstring for the real (if occasional) accuracy tradeoff this carries.
        """
        if not answer or not answer.strip() or not retrieved_docs:
            return 0.0

        docs_to_score = retrieved_docs[:GROUNDING_CHECK_MAX_DOCS]

        if self.cross_encoder is not None:
            score = self._check_grounding_cross_encoder(answer, docs_to_score)
        else:
            score = self._check_grounding_word_overlap(answer, docs_to_score)

        if DEBUG:
            print(f"[check_grounding] Score: {score:.4f}")

        return round(score, 4)

    def _check_grounding_cross_encoder(self, answer: str, retrieved_docs: list) -> float:
        pairs = [(answer, doc) for doc in retrieved_docs if doc]
        if not pairs:
            return 0.0
        raw_scores = self.cross_encoder.predict(pairs)
        best = max(raw_scores)
        return min(_sigmoid(float(best)), 1.0)

    def _check_grounding_word_overlap(self, answer: str, retrieved_docs: list) -> float:
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0

        context_tokens = set()
        for doc in retrieved_docs:
            context_tokens |= _tokenize(doc)

        if not context_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        return min(len(overlap) / len(answer_tokens), 1.0)

    def validate_answer(self, answer: str) -> bool:
        """Reject empty, too-short, or refusal-style non-answers."""
        if not answer or len(answer.strip()) == 0:
            return False
        if len(answer.strip()) < 3:
            return False
        lowered = answer.strip().lower()
        if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
            return False
        return True


if __name__ == "__main__":
    guardrails = Guardrails()
    score = guardrails.check_grounding(
        answer="A corporation is a business entity chartered by a state.",
        retrieved_docs=["A corporation is the most common form of business organization, "
                         "chartered by a state and given legal rights separate from its owners."]
    )
    print(f"Grounding score: {score}")
    print(f"Valid: {guardrails.validate_answer('A corporation is a business entity.')}")
    refusal_text = "I don't know the answer."
    print(f"Valid (refusal): {guardrails.validate_answer(refusal_text)}")
