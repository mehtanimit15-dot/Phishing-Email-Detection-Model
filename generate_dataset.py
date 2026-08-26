"""
generate_dataset.py
--------------------
Builds a synthetic-but-realistic labeled dataset of phishing and legitimate
emails, since we don't have internet access to pull a real public dataset
(e.g. the Kaggle "Phishing Email Detection" or Nazario phishing corpus).

The templates below are deliberately built from *patterns* commonly cited in
phishing-awareness training (urgency, fake account alerts, prize scams,
lookalike domains, IP-address links, "@" trick URLs) rather than copied from
any real dataset or real organization. Swap `USE_REAL_DATASET = True` and
point `REAL_DATASET_PATH` at a CSV with `text,label` columns to use real data
instead — the rest of the pipeline works identically either way.

Output: sample_emails.csv with columns [text, label] where label is
"phishing" or "safe".
"""

import random
import pandas as pd

random.seed(42)

NAMES = ["Alex", "Jordan", "Taylor", "Sam", "Morgan", "Priya", "Wei", "Carlos",
         "Fatima", "Liam", "Nina", "Omar", "Grace", "Daniel", "Aisha"]

COMPANIES = ["Northwind Bank", "Cascade Cloud", "BrightPay", "Meridian Health",
             "Orbit Telecom", "Summit Retail", "Harbor Insurance", "Vertex Software"]

REAL_DOMAINS = ["northwindbank.com", "cascadecloud.io", "brightpay.com",
                "meridianhealth.org", "orbittelecom.com", "summitretail.com",
                "harborinsurance.com", "vertexsoftware.dev", "yourcompany.com"]

FAKE_TLDS = [".tk", ".ml", ".ga", ".cf", ".info", ".xyz", ".top"]

# ---------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------

def legit_url():
    domain = random.choice(REAL_DOMAINS)
    path = random.choice(["account", "billing", "newsletter", "support", "dashboard"])
    return f"https://www.{domain}/{path}"


def phishing_url():
    style = random.choice(["ip", "lookalike", "at_trick", "fake_tld", "long_random"])
    if style == "ip":
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/login/verify"
    if style == "lookalike":
        base = random.choice(REAL_DOMAINS).split(".")[0]
        return f"http://{base}-secure-login{random.choice(FAKE_TLDS)}/verify"
    if style == "at_trick":
        real = random.choice(REAL_DOMAINS)
        fake = "".join(random.choices("abcdefghijklmnop", k=8))
        return f"http://{real}@{fake}.ru/session"
    if style == "fake_tld":
        word = random.choice(["account-verify", "secure-update", "confirm-now", "billing-alert"])
        return f"http://{word}{random.choice(FAKE_TLDS)}/login"
    rand = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=14))
    return f"http://{rand}.info/index.php?verify=1"


# ---------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------

PHISHING_TEMPLATES = [
    "Dear {name}, your {company} account has been suspended due to unusual "
    "activity. Click here to verify your account immediately or it will be "
    "permanently closed within 24 hours: {url}",

    "URGENT SECURITY ALERT: We detected a login attempt from an unrecognized "
    "device. Verify your identity now to avoid losing access to your account: {url}",

    "Congratulations {name}! You have been selected to win a $500 gift card. "
    "Claim your prize now before it expires: {url}",

    "Your payment of $249.99 could not be processed. Update your billing "
    "information immediately to avoid service interruption: {url}",

    "Dear valued customer, our records show your {company} password will "
    "expire today. Click below to reset it now and keep your account active: {url}",

    "ACTION REQUIRED: Unusual sign-in activity was detected on your account. "
    "Confirm it was you within 24 hours or your account will be locked: {url}",

    "Hi {name}, this is {company} Support. We noticed a failed delivery "
    "attempt for your package. Confirm your address here to reschedule: {url}",

    "Final notice: Your {company} subscription payment failed. Please update "
    "your card details immediately to avoid cancellation: {url}",

    "Your mailbox has exceeded its storage limit and new emails will be "
    "rejected. Verify your account now to restore full access: {url}",

    "We have processed a refund of $89.00 to your account by mistake. Please "
    "click here to return the funds immediately to avoid legal action: {url}",
]

LEGIT_TEMPLATES = [
    "Hi {name}, just a reminder that our meeting is scheduled for {day} at "
    "{time}. Let me know if you have any questions before then.",

    "Hello {name}, please find attached the invoice for last month's "
    "services from {company}. Let us know if anything looks off.",

    "Hi team, this week's newsletter is out — highlights and updates from "
    "{company} are here: {url}",

    "Dear {name}, thank you for your recent purchase from {company}. Your "
    "order #{order} has shipped and should arrive by {day}.",

    "Hi {name}, following up on our call earlier — I've attached the notes "
    "and next steps we discussed. Talk soon.",

    "Hello, this is a reminder that your appointment with {company} is "
    "confirmed for {day} at {time}. Reply to this email if you need to reschedule.",

    "Hi {name}, the quarterly report from {company} is ready for review. "
    "You can access it here: {url}",

    "Hey, are we still on for lunch on {day}? Let me know what time works "
    "for you.",

    "Dear {name}, your recent statement from {company} is now available in "
    "your online account: {url}",

    "Hi all, just sharing the slides from today's presentation for anyone "
    "who missed the meeting. Thanks for joining!",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["9:00 AM", "10:30 AM", "1:00 PM", "3:15 PM", "4:00 PM"]


def fill(template, is_phishing):
    return template.format(
        name=random.choice(NAMES),
        company=random.choice(COMPANIES),
        day=random.choice(DAYS),
        time=random.choice(TIMES),
        order=random.randint(10000, 99999),
        url=phishing_url() if is_phishing else legit_url(),
    )


def generate_dataset(n_per_class=300):
    rows = []
    for _ in range(n_per_class):
        template = random.choice(PHISHING_TEMPLATES)
        rows.append({"text": fill(template, is_phishing=True), "label": "phishing"})
    for _ in range(n_per_class):
        template = random.choice(LEGIT_TEMPLATES)
        rows.append({"text": fill(template, is_phishing=False), "label": "safe"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_per_class=300)
    df.to_csv("sample_emails.csv", index=False)
    print(f"Generated {len(df)} emails -> sample_emails.csv")
    print(df["label"].value_counts())
