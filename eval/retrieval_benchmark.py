"""
A retrieval-only benchmark: question -> which source file(s) contain the
answer.

This exists because the LLM benchmark in benchmark.py can't be run
without an API key, which left the project with no measured numbers at
all. Retrieval needs no LLM: the embedding model is local and the gold
label is a filename, so this harness runs offline, in CI, and on a
laptop with no credentials - and retrieval quality is upstream of
everything the critic later does. A critic cannot judge sufficiency of
context it was never given.

`gold` is a set because several questions are legitimately answered by
more than one document (SCIM is described in the admin guide and dated in
the release notes). Retrieving either is correct.

The distractor questions have no gold document on purpose. They measure
the relevance gate: the right behaviour is to retrieve NOTHING, because
the knowledge base genuinely does not discuss them. A retriever with no
floor scores zero here by construction - it always returns its k nearest
neighbours, however irrelevant.
"""

RETRIEVAL_BENCHMARK = [
    # --- semantic: phrased differently from the document ---
    {"question": "What is the refund policy for monthly plans?",
     "gold": {"aria_billing_policy.txt"}},
    {"question": "How many previous file versions does Aria keep?",
     "gold": {"aria_overview.txt"}},
    {"question": "What encryption does Aria use for data at rest?",
     "gold": {"aria_security.txt"}},
    {"question": "What happens to an account after a failed payment?",
     "gold": {"aria_billing_policy.txt"}},
    {"question": "Does Aria support single sign-on, and on which plan?",
     "gold": {"aria_overview.txt", "aria_integrations.txt"}},
    {"question": "How much storage does the Starter plan include per user?",
     "gold": {"aria_overview.txt"}},
    {"question": "What uptime does Aria commit to for Team plans?",
     "gold": {"aria_support_sla.txt"}},
    {"question": "How big a service credit applies if uptime drops below 95 percent?",
     "gold": {"aria_support_sla.txt"}},
    {"question": "Where is customer file data physically stored?",
     "gold": {"aria_data_handling.txt"}},
    {"question": "How long can a deleted file be recovered on the Enterprise plan?",
     "gold": {"aria_data_handling.txt"}},
    {"question": "What is the discount for paying yearly instead of monthly?",
     "gold": {"aria_billing_policy.txt"}},
    {"question": "Which permission levels can I assign to a user?",
     "gold": {"aria_admin_guide.txt"}},
    {"question": "Which desktop operating systems are supported?",
     "gold": {"aria_integrations.txt"}},
    {"question": "Can an administrator sign a user out of a lost laptop?",
     "gold": {"aria_admin_guide.txt"}},
    {"question": "Does Aria offer a data processing addendum for GDPR?",
     "gold": {"aria_data_handling.txt"}},

    # --- lexical: the question quotes a rare literal token, which is
    #     where dense-only retrieval is weakest ---
    {"question": "What HTTP status code does the API return when rate limited?",
     "gold": {"aria_api_reference.txt"}},
    {"question": "What does the X-RateLimit-Reset header contain?",
     "gold": {"aria_api_reference.txt", "aria_release_notes.txt"}},
    {"question": "How is a webhook delivery signed with HMAC-SHA256?",
     "gold": {"aria_api_reference.txt"}},
    {"question": "Is SCIM 2.0 provisioning available, and on which plan?",
     "gold": {"aria_admin_guide.txt", "aria_release_notes.txt"}},
    {"question": "Which release added FIDO2 hardware security key support?",
     "gold": {"aria_release_notes.txt", "aria_security.txt"}},
    {"question": "Is BYOK supported through AWS KMS?",
     "gold": {"aria_security.txt"}},
    {"question": "What happened to the WebDAV gateway?",
     "gold": {"aria_integrations.txt", "aria_release_notes.txt"}},
    {"question": "Is Aria ISO 27001 certified?",
     "gold": {"aria_security.txt", "aria_release_notes.txt"}},
    {"question": "What is the Severity 1 first response time for Enterprise?",
     "gold": {"aria_support_sla.txt"}},
    {"question": "What is the maximum limit parameter on list endpoints?",
     "gold": {"aria_api_reference.txt"}},
]

# No gold document: the corpus genuinely says nothing about these. The
# correct retrieval result is the empty set.
RETRIEVAL_DISTRACTORS = [
    "What is a good recipe for banana bread?",
    "How do I rotate the tyres on a Honda Civic?",
    "What is the capital of Mongolia?",
    "Explain the offside rule in football.",
    "How do I file a tax extension in the United States?",
    "Who won the 1998 FIFA World Cup final?",
]
