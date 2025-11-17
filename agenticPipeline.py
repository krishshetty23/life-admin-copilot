import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from semanticsearch import SemanticSearch


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SIMILARITY_THRESHOLD_AGENTIC: float = 0.3


# planning function
def plan_actions(email_data):
    """
    Use OpenAI to decide what information should be retrieved from the
    user's profile before replying to this email.

    Returns a dict with:
        - needed_info: list of things to search for
        - reasoning: why it needs these things
        - search_queries: actual queries to run
    """
    systemPrompt = (
        "You are a planning assistant for an email-reply AI system. "
        "Given structured information about an incoming email, you decide "
        "what information should be retrieved from the user's personal profile "
        "before writing a reply. You must respond ONLY with valid JSON "
        "matching the requested schema."
    )

    userPrompt =  f"""
Here is the parsed email data (as JSON):

{json.dumps(email_data, indent=2)}

Given this email, what information would you need from the user's profile
to write a good reply? List 2–3 specific pieces of information.

For example, you might request:
- current home address
- preferred payment method
- usual appointment availability
- relevant account or policy numbers, etc.

For each needed piece of information, suggest one or more short search
queries that could be used to look it up in the user's profile.

Return ONLY valid JSON in this exact format:

{{
  "needed_info": [
    "first piece of information to retrieve",
    "second piece of information to retrieve"
  ],
  "reasoning": "Short explanation of why these items are needed.",
  "search_queries": [
    "first search query",
    "second search query",
    "optional third search query"
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,  # small creativity, but mostly deterministic
            response_format={"type": "json_object"},  # force JSON
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt},
            ],
        )

        content = response.choices[0].message.content or "{}"
        plan = json.loads(content)

        return {
            "needed_info": plan.get("needed_info", []),
            "reasoning": plan.get("reasoning", ""),
            "search_queries": plan.get("search_queries", []),
        }
    
    except Exception as e:
        print(f"[plan_actions] Error while planning actions: {e}")
        return {
            "needed_info": [],
            "reasoning": "Failed to plan actions due to an error.",
            "search_queries": [],
        }


# connecting the brain to the search engine
def executePlan(plan):
    """
    Takes the plan from plan_actions() and executes the searches.
    Returns context needed for reply generation.
    """
    if not plan:
        return {
            "found_context": [],
            "missing_info": [],
            "confidence_scores": {},
        }
    
    seaerchQueries = plan.get("search_queries", []) or []
    foundContext = []
    missingInfo = []
    confidenceScores = {}

    for query in seaerchQueries:
        if not query or not query.strip():      # skipping empty queries just in case
            continue

        try:
            result = SemanticSearch(query)
        except Exception as e:
            print(f"[executePlan] Error during SemanticSearch for query '{query}': {e}")
            confidenceScores[query] = 0.0
            missingInfo.append(query)
            continue

        score = float(result.get("score", 0.0))
        bestMatch = result.get("best_match")
        confidenceScores[query] = score

        if score >= SIMILARITY_THRESHOLD_AGENTIC and bestMatch:
            foundContext.append(
                {
                    "query": query,
                    "best_match": bestMatch,
                    "score": score,
                }
            )
        else:
            missingInfo.append(query)

    return {
        "found_context": foundContext,
        "missing_info": missingInfo,
        "confidence_scores": confidenceScores,
    }


# generating the replies
def replyAI(email_data, context):
    """
    Generate a reply using the context found by the agent.
    
    Args:
        email_data: Original parsed email
        context: Output from executePlan() with found_context and missing_info
    
    Returns:
        dict with 'reply' text and 'metadata' about what was used
    """
    foundContext = context.get("found_context", []) if context else []
    missingInfo = context.get("missing_info", []) if context else []
    confidenceScores = context.get("confidence_scores", {}) if context else {}

    if foundContext:
        contextLines = []
        for item in foundContext:
            q = item.get("query", "")
            best = item.get("best_match", "")
            score = item.get("score", 0.0)
            contextLines.append(
                f"- For query '{q}' (score {score:.2f}): {best}"
            )
        contextText = "\n".join(contextLines)
    else:
        contextText = "None found."

    if missingInfo:
        missingText = "\n".join(f"- {q}" for q in missingInfo)
    else:
        missingText = "None."

    systemPrompt = (
        "You are a helpful email assistant that writes polite, professional replies "
        "on behalf of the user. You should use any profile context provided to make "
        "the reply more accurate and personalized."
    )

    userPrompt = f"""
Here is the structured information about the incoming email:
{json.dumps(email_data, indent=2)}

Profile search results (context you may use in the reply):
{contextText}

Information that could not be found in the profile (you may need to ask for this politely):
{missingText}

Instructions:
- Write a polite, professional reply to the original email.
- Use the profile context naturally if it helps (do NOT mention that it came from a profile).
- If some needed information is missing, politely ask the sender to provide it.
- Keep the reply concise (2–3 sentences max).
- Write the reply as if you are the user responding directly to the sender.
- Return ONLY the email body text (no JSON, no explanation).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.35,
            max_tokens=200,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt},
            ],
        )

        reply_text = (response.choices[0].message.content or "").strip()

    except Exception as e:
        print(f"[generate_reply] Error calling OpenAI: {e}")
        reply_text = (
            "I'm sorry, but I’m currently unable to generate a full reply. "
            "Please let me know your updated details so I can assist you further."
        )

    # --- Build metadata ---

    metadata = {
        "email_type": email_data.get("type"),
        "used_context": foundContext,
        "missing_info": missingInfo,
        "confidence_scores": confidenceScores,
    }

    return {
        "reply": reply_text,
        "metadata": metadata,
    }


if __name__ == "__main__":
    test_email = {
        "type": "address_change",
        "content": "Please update your address in our system..."
    }
    
    print("="*60)
    print("STEP 1: PLANNING")
    print("="*60)
    plan = plan_actions(test_email)
    print(json.dumps(plan, indent=2))
    
    print("\n" + "="*60)
    print("STEP 2: EXECUTING SEARCHES")
    print("="*60)
    context = executePlan(plan)
    print(json.dumps(context, indent=2, default=str))
    
    print("\n" + "="*60)
    print("STEP 3: GENERATING REPLY")
    print("="*60)
    reply = replyAI(test_email, context)
    print(reply)