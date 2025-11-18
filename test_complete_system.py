"""
Test suite for Life Admin Copilot
Demonstrates the system handling various email scenarios
"""

from complete_copilot import handle_email_end_to_end


def test_all_scenarios():
    """Run all test scenarios"""
    
    test_cases = [
        {
            "name": "Appointment Reminder",
            "email": """
Subject: Upcoming Appointment Reminder

Dear Patient,

This is a reminder that you have an appointment scheduled. 
Please confirm your email address and phone number for our records.

If you need to reschedule, please let us know at least 24 hours in advance.

Regards,
Medical Office
"""
        },
        {
            "name": "Account Verification",
            "email": """
Subject: Account Verification Required

Hello,

We need to verify your account information. Please provide:
- Full name
- Email address
- Phone number
- Current address

This is for security purposes.

Thank you,
Security Team
"""
        },
        {
            "name": "Shipping Update",
            "email": """
Subject: Shipping Address Confirmation

Hi there,

Your order #12345 is ready to ship. Please confirm your current shipping address 
to ensure timely delivery.

Best,
Logistics Team
"""
        },
        {
            "name": "Contact Info Update",
            "email": """
Subject: Update Your Contact Information

Dear Member,

We're updating our records. Please confirm if the following information is still current:
- Email
- Phone number
- Mailing address

Reply with any changes or confirm if everything is correct.

Thanks,
Membership Services
"""
        },
        {
            "name": "Subscription Renewal",
            "email": """
Subject: Subscription Renewal Notice

Hello,

Your annual subscription is due for renewal. We need to verify your payment information 
and contact details before processing.

Please provide your current email and phone number.

Sincerely,
Billing Department
"""
        }
    ]
    
    print("="*70)
    print(" " * 20 + "COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"\nTesting {len(test_cases)} different email scenarios...\n")
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_cases)}: {test['name']}")
        print(f"{'='*70}")
        
        try:
            result = handle_email_end_to_end(test['email'])
            results.append({
                "test_name": test['name'],
                "status": "PASS",
                "response_length": len(result['agent_response'])
            })
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append({
                "test_name": test['name'],
                "status": "FAIL",
                "error": str(e)
            })
    
    # Print summary
    print("\n\n" + "="*70)
    print(" " * 25 + "TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    
    for r in results:
        status_symbol = "✅" if r['status'] == 'PASS' else "❌"
        print(f"{status_symbol} {r['test_name']}: {r['status']}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    test_all_scenarios()