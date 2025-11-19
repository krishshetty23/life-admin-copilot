import streamlit as st
from CompleteCopilot import handle_email_end_to_end

# Page configuration
st.set_page_config(
    page_title="Life Admin Copilot - Email Assistant Demo",
    page_icon="📧",
    layout="wide"
)

# Header
st.title("📧 Life Admin Copilot - Email Assistant Demo")
st.markdown("**AI-powered assistant that automatically parses bureaucratic emails, searches your profile, and generates intelligent replies.**")

st.divider()

# Sample emails showcasing different capabilities
SAMPLE_EMAILS = {
    "1. Address Update - University Records (Semantic Search)": """Subject: URGENT - Verify Your Educational Institution

Dear Student,

We are updating our alumni database and need to verify your current educational institution on file.

Please confirm your university name by December 15, 2024. There is no processing fee for this update.

Current address: Illinois, 60192

For questions, contact our office at 1-800-555-0123.

Best regards,
National Student Clearinghouse""",

    "2. Payment Received - Tuition (Structured Extraction)": """Subject: Payment Confirmation - Spring 2024 Tuition

Dear Student,

This confirms receipt of your tuition payment for Spring 2024 semester.

Amount Paid: $8,750.00
Payment Date: November 15, 2024
Invoice Number: INV-2024-11-089
Transaction ID: TXN-447821-EDU
Payment Method: Bank Transfer - Account ending in 9012

Your enrollment is now confirmed for Spring 2024.

Bursar's Office
Student Financial Services""",

    "3. Academic Advising Appointment (Intelligent Reply)": """Subject: Schedule Your Graduation Planning Meeting

Hello,

You are required to schedule a graduation planning meeting for all students graduating in 2025.

Appointment Details:
Date: January 15, 2025
Time: 10:00 AM
Location: Student Services Building, Room 204
With: Dr. Sarah Johnson, Academic Advisor
Purpose: Graduation Requirements Review

Please confirm your expected graduation year. If you cannot attend, call us at least 48 hours in advance at (607) 777-4000.

Academic Advising Office""",

    "4. DMV Address Verification (Profile Phone Search)": """Subject: Action Required - Driver's License Address Update

Dear Illinois Resident,

Our records indicate your driver's license address may need updating.

Deadline to update: January 30, 2025
Processing fee: $25.00
Current address on file: Illinois, 60192

Please reply with your current phone number and confirm your mailing address, or call 1-800-252-8980.

Illinois Department of Motor Vehicles""",

    "5. Tech Company Interview Appointment (Complex Semantic)": """Subject: Interview Confirmation - Software Engineer Position

Dear Candidate,

We are pleased to schedule your technical interview for our Software Engineer role.

Interview Date: December 10, 2024
Interview Time: 14:00
Location: Virtual - Zoom Meeting
Interviewer: Alex Chen, Engineering Manager
Focus: Technical Skills Assessment - Python, AI/ML

Please confirm your relevant technical skills and degree program. To reschedule, contact us at least 48 hours before at (415) 555-7890.

Looking forward to speaking with you!
Tech Innovations Inc.""",

    "6. Conference Payment - Academic Registration (Multi-field)": """Subject: Payment Received - AI Research Conference 2024

Dear Researcher,

Thank you for registering for the International AI Research Conference 2024.

Payment Amount: $350.00
Registration Date: November 18, 2024
Confirmation Number: AIRC-2024-1847
Transaction Reference: PAY-AI-192847
Payment Method: Credit Card ending in 5521

Your academic affiliation (university name) and email are recorded for badge printing.

Questions? Contact conference@airesearch.org

Conference Committee"""
}

# Initialize session state for tracking active input
if 'use_sample' not in st.session_state:
    st.session_state.use_sample = True

# Create two tabs for input
tab1, tab2 = st.tabs(["📝 Write Email", "📚 Sample Emails"])

with tab1:
    st.subheader("Paste or Write Your Own Email")
    user_email = st.text_area(
        "Enter the email content below:",
        height=300,
        placeholder="Paste an email here to test the AI assistant...",
        help="The assistant will parse this email, search your profile, and generate an intelligent reply.",
        on_change=lambda: setattr(st.session_state, 'use_sample', False)
    )

with tab2:
    st.subheader("Try Pre-Built Examples")

    # Explanation of what each email showcases
    with st.expander("ℹ️ What does each email showcase?"):
        st.markdown("""
        **1. University Records** → Semantic search for institution name
        **2. Tuition Payment** → Structured extraction (amount, dates, IDs)
        **3. Academic Advising** → Intelligent matching of graduation year
        **4. DMV Verification** → Profile phone number retrieval
        **5. Tech Interview** → Complex semantic search (skills + degree)
        **6. Conference Registration** → Multi-field extraction + affiliation search
        """)

    selected_sample = st.selectbox(
        "Choose a sample email:",
        options=list(SAMPLE_EMAILS.keys()),
        help="Each sample demonstrates different AI capabilities",
        on_change=lambda: setattr(st.session_state, 'use_sample', True)
    )

    st.text_area(
        "Email Preview:",
        value=SAMPLE_EMAILS[selected_sample],
        height=300,
        disabled=True
    )

st.divider()

# Determine which email to process
email_to_process = None
if st.button("🚀 Process Email", type="primary", use_container_width=True):
    # Use session state to determine which input to use
    # Priority: if user has typed in custom email and it's not empty, use that
    # Otherwise, use the selected sample
    if not st.session_state.use_sample and user_email and user_email.strip():
        email_to_process = user_email
    else:
        email_to_process = SAMPLE_EMAILS[selected_sample]

    if email_to_process:
        # Display processing indicator
        with st.spinner("🔄 Processing email... This may take a few seconds."):
            try:
                # Call the backend function
                result = handle_email_end_to_end(email_to_process)

                st.success("✅ Email processed successfully!")

                # Pipeline Status Badge with visual flow
                st.markdown("### 🔄 Pipeline Status")
                pipeline_cols = st.columns([1, 0.2, 1, 0.2, 1])
                with pipeline_cols[0]:
                    st.success("**📋 Parsing**")
                with pipeline_cols[1]:
                    st.markdown("<h3 style='text-align: center;'>→</h3>", unsafe_allow_html=True)
                with pipeline_cols[2]:
                    st.success("**🔍 Searching**")
                with pipeline_cols[3]:
                    st.markdown("<h3 style='text-align: center;'>→</h3>", unsafe_allow_html=True)
                with pipeline_cols[4]:
                    st.success("**✉️ Generating**")

                st.divider()

                # Display results in 3 columns
                st.subheader("📊 Results")
                st.caption("See how the AI processes your email in three stages")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 📋 Parsed Email")
                    st.caption("✨ Structured data extracted from the email")

                    parsed = result["parsed_email"]

                    # Show email type prominently
                    email_type = parsed.get("type", "unknown")
                    st.info(f"**Type:** {email_type.replace('_', ' ').title()}")

                    # Display key fields (exclude internal metadata)
                    fields_to_hide = ["original_email", "parsing_status", "error", "raw_response"]
                    display_fields = {k: v for k, v in parsed.items() if k not in fields_to_hide}

                    # Format the display
                    if display_fields:
                        for key, value in display_fields.items():
                            if key != "type":  # Already displayed above
                                formatted_key = key.replace('_', ' ').title()
                                st.text(f"{formatted_key}:")
                                st.code(str(value), language=None)

                    # Show error if parsing failed
                    if "error" in parsed:
                        st.error(f"⚠️ {parsed['error']}")

                with col2:
                    st.markdown("### 🔍 Retrieved Context")
                    st.caption("🎯 Information found in your profile")

                    # Extract agent's tool calls and searches from full_result
                    full_result = result.get("full_result", {})
                    messages = full_result.get("messages", [])

                    # Map tool names to friendly labels
                    tool_display_names = {
                        "search_profile": "🔍 Searched: User Profile",
                        "calculator": "🧮 Calculated: Math Expression",
                        "check_missing_info": "✓ Verified: Required Information"
                    }

                    # Look for tool calls and their responses
                    tool_calls_found = False
                    tool_responses = {}

                    # First pass: collect all tool responses
                    for i, msg in enumerate(messages):
                        if hasattr(msg, "type") and msg.type == "tool":
                            # Get the tool call ID to match with its response
                            tool_id = getattr(msg, "tool_call_id", None)
                            if tool_id:
                                tool_responses[tool_id] = msg.content

                    # Second pass: display tool calls with their responses
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            tool_calls_found = True
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get("name", "unknown")
                                tool_args = tool_call.get("args", {})
                                tool_id = tool_call.get("id", None)

                                # Display friendly tool name
                                friendly_name = tool_display_names.get(tool_name, f"🔧 Tool: {tool_name}")
                                st.markdown(f"**{friendly_name}**")

                                # Display what was searched for
                                if "question" in tool_args:
                                    st.text("🔎 Looking for:")
                                    st.code(tool_args["question"], language=None)
                                elif "expression" in tool_args:
                                    st.text("🔢 Expression:")
                                    st.code(tool_args["expression"], language=None)

                                # Display what was FOUND
                                if tool_id and tool_id in tool_responses:
                                    response_content = tool_responses[tool_id]
                                    st.text("✨ Found:")
                                    # Highlight key information
                                    if len(response_content) > 150:
                                        st.success(response_content[:150] + "...")
                                    else:
                                        st.success(response_content)

                                st.markdown("---")

                    if not tool_calls_found:
                        st.info("✓ Agent generated reply using only the parsed email data.")

                with col3:
                    st.markdown("### ✉️ Generated Reply")
                    st.caption("💼 AI-generated professional response")

                    agent_response = result["agent_response"]

                    # Display the reply in a nice formatted box
                    st.markdown("---")
                    st.markdown(agent_response)
                    st.markdown("---")

                    # Add a copy button
                    if st.button("📋 Copy to Clipboard", key="copy_reply"):
                        st.code(agent_response, language=None)
                        st.caption("Copy the text above ☝️")

            except Exception as e:
                st.error(f"❌ Error processing email: {str(e)}")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter an email or select a sample email first.")

st.divider()

# Footer with instructions
with st.expander("💡 How to use this demo"):
    st.markdown("""
    1. **Choose your input method:**
       - **Write Email tab**: Paste or type your own email
       - **Sample Emails tab**: Select from pre-built examples

    2. **Click "Process Email"** to run the AI pipeline

    3. **View the results:**
       - **Parsed Email**: See how AI extracts structured data
       - **Retrieved Context**: See what information was found in your profile
       - **Generated Reply**: Get an intelligent, professional email response

    **Tip**: Try different sample emails to see how the AI handles various scenarios!
    """)
