import pandas as pd
from collections import defaultdict
from sf_client import SalesforceClient


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ACCOUNTS_FILE = "data/accounts_raw.csv"
CASES_FILE = "data/cases_raw.csv"


# --------------------------------------------------
# Load CSV data
# --------------------------------------------------

print("Loading CSV data...")

accounts = pd.read_csv(ACCOUNTS_FILE)
cases = pd.read_csv(CASES_FILE)

print(f"Accounts in CSV: {len(accounts)}")
print(f"Cases in CSV: {len(cases)}")


# --------------------------------------------------
# Connect to Salesforce
# --------------------------------------------------

print("\nConnecting to Salesforce...")

sf = SalesforceClient()

print("Salesforce connection successful.")


# --------------------------------------------------
# Get existing project Accounts
# --------------------------------------------------

print("\nChecking existing Salesforce Accounts...")

existing_accounts_query = """
SELECT Id, Account_Ref__c
FROM Account
WHERE Account_Ref__c != NULL
"""

existing_records = sf.query(existing_accounts_query)

account_id_map = {}

if not existing_records.empty:

    for _, row in existing_records.iterrows():

        account_ref = str(row["Account_Ref__c"]).strip()
        salesforce_id = str(row["Id"]).strip()

        account_id_map[account_ref] = salesforce_id


print(
    f"Project Accounts already in Salesforce: "
    f"{len(account_id_map)}"
)


# --------------------------------------------------
# Create / reuse Accounts
# --------------------------------------------------

created_accounts = 0
skipped_accounts = 0
failed_accounts = 0

print("\nProcessing Accounts...\n")


for _, account in accounts.iterrows():

    account_ref = str(account["Account_Ref"]).strip()


    # Existing Account
    if account_ref in account_id_map:

        print(
            f"[SKIP] {account_ref} already exists "
            f"({account['Name']})"
        )

        skipped_accounts += 1
        continue


    # Account payload
    account_payload = {
        "Name": str(account["Name"]),
        "Industry": str(account["Industry"]),
        "Account_Ref__c": account_ref,
        "Tenure_Months__c": int(account["Tenure_Months"]),
        "Monthly_Charges__c": float(account["Monthly_Charges"]),
        "Total_Charges__c": float(account["Total_Charges"]),
        "Contract_Type__c": str(account["Contract_Type"])
    }


    try:

        result = sf.sf.Account.create(account_payload)

        if result.get("success"):

            salesforce_id = result["id"]

            account_id_map[account_ref] = salesforce_id

            created_accounts += 1

            print(
                f"[CREATED] {account_ref} - "
                f"{account['Name']}"
            )

        else:

            failed_accounts += 1

            print(
                f"[FAILED] {account_ref}: {result}"
            )

    except Exception as e:

        failed_accounts += 1

        print(
            f"[ERROR] {account_ref}: {e}"
        )


# --------------------------------------------------
# Get existing Cases in ONE query
# --------------------------------------------------

print("\n----------------------------------------")
print("Checking existing Salesforce Cases...")
print("----------------------------------------\n")


existing_cases_query = """
SELECT Id,
       AccountId,
       Type,
       Priority,
       IsEscalated,
       Days_To_Resolve__c,
       Status
FROM Case
WHERE AccountId != NULL
"""

existing_cases = sf.query(existing_cases_query)


# --------------------------------------------------
# Build Salesforce Account ID -> Account Ref
# --------------------------------------------------

account_ref_by_id = {
    salesforce_id: account_ref
    for account_ref, salesforce_id in account_id_map.items()
}


# --------------------------------------------------
# Build lookup of existing Cases
# --------------------------------------------------

existing_case_lookup = defaultdict(list)

if not existing_cases.empty:

    for _, existing_case in existing_cases.iterrows():

        salesforce_account_id = str(
            existing_case["AccountId"]
        ).strip()

        # Ignore unrelated Salesforce Accounts
        if salesforce_account_id not in account_ref_by_id:
            continue

        account_ref = account_ref_by_id[
            salesforce_account_id
        ]

        existing_case_lookup[account_ref].append({
            "Type": str(existing_case["Type"]).strip(),
            "Priority": str(existing_case["Priority"]).strip(),
            "IsEscalated": bool(
                existing_case["IsEscalated"]
            ),
            "Days_To_Resolve__c": int(
                existing_case["Days_To_Resolve__c"]
            ),
            "Status": str(existing_case["Status"]).strip()
        })


total_existing_project_cases = sum(
    len(case_list)
    for case_list in existing_case_lookup.values()
)


print(
    f"Existing project Cases found: "
    f"{total_existing_project_cases}"
)


# --------------------------------------------------
# Create Cases
# --------------------------------------------------

print("\n----------------------------------------")
print("Processing Cases...")
print("----------------------------------------\n")


created_cases = 0
skipped_cases = 0
failed_cases = 0

total_cases = len(cases)


for position, (_, case) in enumerate(
    cases.iterrows(),
    start=1
):

    account_ref = str(
        case["Account_Ref"]
    ).strip()


    # --------------------------------------------------
    # Find Salesforce Account
    # --------------------------------------------------

    if account_ref not in account_id_map:

        print(
            f"[{position}/{total_cases}] "
            f"[SKIP] Account not found: "
            f"{account_ref}"
        )

        failed_cases += 1
        continue


    salesforce_account_id = account_id_map[
        account_ref
    ]


    # --------------------------------------------------
    # Prepare Case values
    # --------------------------------------------------

    case_type = str(
        case["Case_Type"]
    ).strip()

    priority = str(
        case["Priority"]
    ).strip()

    is_escalated = bool(
        int(case["Is_Escalated"])
    )

    days_to_resolve = int(
        case["Days_To_Resolve"]
    )

    status = str(
        case["Status"]
    ).strip()


    # --------------------------------------------------
    # Check for existing matching Case
    # --------------------------------------------------

    matching_index = None

    existing_cases_for_account = (
        existing_case_lookup.get(
            account_ref,
            []
        )
    )


    for index, existing_case in enumerate(
        existing_cases_for_account
    ):

        if (
            existing_case["Type"] == case_type
            and
            existing_case["Priority"] == priority
            and
            existing_case["IsEscalated"] == is_escalated
            and
            existing_case["Days_To_Resolve__c"]
                == days_to_resolve
            and
            existing_case["Status"] == status
        ):

            matching_index = index
            break


    # --------------------------------------------------
    # Skip existing Case
    # --------------------------------------------------

    if matching_index is not None:

        existing_cases_for_account.pop(
            matching_index
        )

        skipped_cases += 1

        print(
            f"[{position}/{total_cases}] "
            f"[SKIP] Existing Case - "
            f"{account_ref}"
        )

        continue


    # --------------------------------------------------
    # Prepare Salesforce Case payload
    # --------------------------------------------------

    case_payload = {
        "AccountId": salesforce_account_id,
        "Type": case_type,
        "Priority": priority,
        "IsEscalated": is_escalated,
        "Days_To_Resolve__c": days_to_resolve,
        "Status": status
    }


    # --------------------------------------------------
    # Create Case
    # --------------------------------------------------

    try:

        result = sf.sf.Case.create(
            case_payload
        )

        if result.get("success"):

            created_cases += 1

            print(
                f"[{position}/{total_cases}] "
                f"[CREATED] Case - "
                f"{account_ref}"
            )

        else:

            failed_cases += 1

            print(
                f"[{position}/{total_cases}] "
                f"[FAILED] Case - "
                f"{account_ref}: "
                f"{result}"
            )

    except Exception as e:

        failed_cases += 1

        print(
            f"[{position}/{total_cases}] "
            f"[ERROR] Case - "
            f"{account_ref}: {e}"
        )


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n========================================")
print("SALESFORCE IMPORT COMPLETE")
print("========================================")

print(
    f"Accounts created : {created_accounts}"
)

print(
    f"Accounts skipped : {skipped_accounts}"
)

print(
    f"Accounts failed  : {failed_accounts}"
)

print(
    f"Cases created    : {created_cases}"
)

print(
    f"Cases skipped    : {skipped_cases}"
)

print(
    f"Cases failed     : {failed_cases}"
)

print("========================================")

