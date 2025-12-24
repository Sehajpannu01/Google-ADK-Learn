from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import litellm
from google.adk.tools import google_search

import os
load_dotenv()
 
litellm.use_litellm_proxy = True
 
lite_llm_model = LiteLlm(
    model="gemini-2.5-flash",
    api_base=os.getenv("LITELLM_PROXY_API_BASE"),
    api_key=os.getenv("LITELLM_PROXY_GEMINI_API_KEY")    
)


def _best_effort_summary(prompt: str) -> str:
    """Try to collect a short summary for the given prompt/URL via google_search."""
    try:
        search_results = google_search(prompt)
    except Exception:
        return ""

    if isinstance(search_results, str):
        return search_results[:300]

    if isinstance(search_results, list) and search_results:
        top = search_results[0]
        if isinstance(top, dict):
            return (
                top.get("description")
                or top.get("snippet")
                or top.get("content", "")
            )[:300]
        return str(top)[:300]

    if isinstance(search_results, dict):
        return str(search_results.get("description") or search_results)[:300]

    return ""


def _build_base_cases(prompt: str, summary: str):
    sanitized_prompt = prompt.strip() or "the target experience"
    context = summary or sanitized_prompt

    cases = [
        {
            "name": "Smoke_Loads_And_Renders",
            "goal": f"Confirm {sanitized_prompt} loads and key UI renders using real content.",
            "steps": [
                f"Navigate to {sanitized_prompt}.",
                "Wait for above-the-fold content to load.",
                "Validate no console/network errors.",
            ],
            "expected": [
                "HTTP 200 (or appropriate success status).",
                "Hero/primary CTA visible and readable.",
                "Critical scripts/styles load without errors.",
            ],
            "notes": context,
        },
        {
            "name": "Navigation_Primary_CTA",
            "goal": "Ensure the primary call-to-action transitions the user to the right flow.",
            "steps": [
                "Identify primary CTA (e.g., Get Started / Sign Up).",
                "Click CTA and wait for navigation.",
                "Verify destination content and URL.",
            ],
            "expected": [
                "CTA is visible and enabled.",
                "Navigation occurs within 3 seconds.",
                "Destination content matches CTA promise.",
            ],
            "notes": context,
        },
        {
            "name": "Content_Accuracy",
            "goal": "Spot-check that critical copy, imagery, and links match product intent.",
            "steps": [
                "Review hero text, secondary sections, and footer links.",
                "Validate contact/about links resolve.",
                "Compare copy against product positioning in summary/context.",
            ],
            "expected": [
                "No broken links or placeholder text.",
                "Contact/about links reachable.",
                "Copy aligns with prompt/context with no major errors.",
            ],
            "notes": context,
        },
    ]

    lowered_prompt = sanitized_prompt.lower()

    if any(k in lowered_prompt for k in ("login", "sign in", "signin", "auth")):
        cases.append(
            {
                "name": "Auth_Positive_Login",
                "goal": "Validate happy-path authentication works.",
                "steps": [
                    "Navigate to login.",
                    "Enter valid credentials.",
                    "Submit and verify dashboard/home loads.",
                ],
                "expected": [
                    "Credentials accepted.",
                    "User session/token created.",
                    "Dashboard personalized state visible.",
                ],
                "notes": "Focus on authentication experience.",
            }
        )
        cases.append(
            {
                "name": "Auth_Negative_Invalid_Password",
                "goal": "Ensure invalid credentials are rejected securely.",
                "steps": [
                    "Navigate to login.",
                    "Enter valid username + invalid password.",
                    "Submit.",
                ],
                "expected": [
                    "Friendly error surfaced without leaking details.",
                    "Account remains locked.",
                    "Rate limiting / captcha triggered after repeated attempts.",
                ],
                "notes": "Covers negative auth scenario.",
            }
        )

    if any(k in lowered_prompt for k in ("checkout", "cart", "purchase", "payment")):
        cases.append(
            {
                "name": "Checkout_AddToCart_Flow",
                "goal": "Validate add-to-cart and checkout funnel end-to-end.",
                "steps": [
                    "Add representative item to cart.",
                    "Review cart totals, shipping, taxes.",
                    "Proceed to checkout and select payment method.",
                ],
                "expected": [
                    "Cart reflects quantity/price accurately.",
                    "Checkout page shows shipping/billing forms.",
                    "Order summary matches cart before payment.",
                ],
            }
        )

    if "search" in lowered_prompt:
        cases.append(
            {
                "name": "Search_Relevant_Results",
                "goal": "Confirm search returns relevant results and supports filters.",
                "steps": [
                    "Open search input.",
                    "Submit domain-specific query.",
                    "Apply one filter (if available).",
                ],
                "expected": [
                    "Results load quickly (<2s).",
                    "Top results relate to query.",
                    "Filters update results without page reload or errors.",
                ],
            }
        )

    if "form" in lowered_prompt or "contact" in lowered_prompt:
        cases.append(
            {
                "name": "Form_Validation",
                "goal": "Ensure required form fields enforce validation and submission works.",
                "steps": [
                    "Open form (contact/onboarding).",
                    "Attempt submit without required fields.",
                    "Fill valid data and resubmit.",
                ],
                "expected": [
                    "Inline validation for missing/invalid inputs.",
                    "Successful submission returns confirmation message/email.",
                ],
            }
        )

    return cases


def generate_test_cases(prompt: str):
    summary = _best_effort_summary(prompt)
    cases = _build_base_cases(prompt, summary)

    return {
        "status": "success",
        "source": summary or "prompt",
        "test_case_count": len(cases),
        "test_cases": cases,
    }


root_agent = Agent(
    name="qa_test_cases_gen",
    model=lite_llm_model,
    description="QA Test Cases Generator",
    instruction= '''You are a helpful assistant that can generate test cases for a given prompt.
    You will be given a prompt and you will need to generate test cases for it.
    I will provide you a website url you need to navigate and generate test cases for it.''',
    tools=[generate_test_cases],
)