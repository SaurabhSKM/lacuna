"""
A small, hand-labeled benchmark against data/sample_docs, used to measure
whether the system knows what it doesn't know - not just whether it can
retrieve and summarize.

Three categories, each testing a different failure mode:

  - "answerable":   the docs directly cover this. The critic should
                     route to "accept".

  - "unanswerable": neither the docs nor a real web search can answer
                     this, because it's a fictional-product detail that
                     was never specified anywhere (Aria Cloud Storage
                     isn't a real product, so no amount of web searching
                     will turn up its real CTO or revenue figures). The
                     critic should NOT route to "accept" - confidently
                     answering here would be a hallucination. This
                     category is the whole point of the benchmark: it's
                     designed to catch the failure mode that matters
                     most - a confident, fabricated answer.

  - "ambiguous":    the question itself is underspecified (which plan?
                     which certification?). The critic should route to
                     "ask_user", not guess.
"""

BENCHMARK = [
    # --- answerable: directly covered by the sample docs ---
    {"question": "What is the refund policy for monthly plans?", "category": "answerable"},
    {"question": "How many previous file versions does Aria keep?", "category": "answerable"},
    {"question": "What encryption does Aria use for data at rest?", "category": "answerable"},
    {"question": "What happens to an account after a failed payment?", "category": "answerable"},
    {"question": "Does Aria support single sign-on, and on which plan?", "category": "answerable"},
    {"question": "How much storage does the Starter plan include per user?", "category": "answerable"},

    # --- unanswerable: fictional-product specifics never stated, and not
    #     findable via a real web search either ---
    {"question": "Does Aria support quantum-resistant encryption?", "category": "unanswerable"},
    {"question": "What was Aria's revenue in its most recent fiscal year?", "category": "unanswerable"},
    {"question": "Is Aria compliant with Japan's APPI data protection law?", "category": "unanswerable"},
    {"question": "Who is Aria's current Chief Technology Officer?", "category": "unanswerable"},
    {"question": "Does Aria offer a discount for registered non-profit organizations?", "category": "unanswerable"},

    # --- ambiguous: underspecified, needs clarification rather than a guess ---
    {"question": "What is the price?", "category": "ambiguous"},
    {"question": "Is it certified?", "category": "ambiguous"},
    {"question": "What happens if I cancel?", "category": "ambiguous"},
]
