import requests
import streamlit as st
import re

#============================================================================================================================================
# PULL PTO DATA FROM CW

@st.cache_data(ttl=3600*24) # cache for 1 day
def fetch_pto_tickets(extra_params=None):
    # Credentials (already base64 if you got them from Postman)
    clientid = st.secrets['cw_pto']['clientid']
    auth_header = st.secrets['cw_pto']['auth_header']

    # API base
    base_url = "https://cw.lincolncomputers.com/v4_6_release/apis/3.0/service/tickets"

    # Fixed parameters
    page_size = 1000
    page = 1
    all_tickets = []

    base_params = {
        "conditions": "board/id=42",
        "orderby": "id desc",
        "pageSize": str(page_size),
        "page": str(page)
    }

    if extra_params:
        base_params.update(extra_params)

    headers = {
        "clientid": clientid,
        "Authorization": auth_header,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate"
    }

    # print(f"Fetching page {page}...")
    response = requests.get(base_url, headers=headers, params=base_params)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        # break

    tickets = response.json()

    # if not tickets:
        # print("No more tickets.")
        # break

    all_tickets.extend(tickets)
    page += 1

    return all_tickets

# print(f"Fetched {len(all_tickets)} tickets in total.")

# # Optional: write to file
# with open("all_tickets.json", "w", encoding="utf-8") as f:
#     json.dump(all_tickets, f, ensure_ascii=False, indent=2)

# def clear_pto_cache():
#     fetch_pto_tickets.clear

def clear_pto_cache():
    fetch_pto_tickets.clear()


#============================================================================================================================================
# PARSE DATA INTO DICTS WITH SUMMARY AND STATUS

def clean_api_response(all_tickets):
    # extract pto summaries from api response
    pto_data = []
    for ticket in all_tickets:
        if "summary" in ticket and "status" in ticket and isinstance(ticket["status"], dict) and ticket["status"]["name"] != ">Closed":
            if "name" in ticket["status"]:
                pto_data.append({
                    "summary": ticket["summary"],
                    "status": ticket["status"]["name"]
                })


    # st.write(f'PTO Data: {pto_data}')

#============================================================================================================================================
# PARSE OUT NAME, LOCATION, TEAM, DATE(S), TIME FROM PTO_DATA



    pattern = re.compile(r"""
        ^                             # start of string
        (?P<Name>[A-Za-z]+\s[A-Za-z]) # Name: First Last (2 words)
        \s\(
            (?P<Location>[^-]+?)      # Location: everything before " -"
            \s*-\s*
            (?P<Team>[^)]+)           # Team: everything before ")"
        \)\s*
        (?P<Type>[A-Za-z]+)?          # Optional Type (e.g., "FMLA", "PTO", etc.)
        (?:\s*\((?P<TimeRange>\d{1,2}:\d{2}[ap] - \d{1,2}:\d{2}[ap])\))? # Optional time range
        \s*
        (?P<Days>(?:\d{1,2}/\d{1,2})(?:,\s*\d{1,2}/\d{1,2})*) # One or more dates
        $                             # end of string
    """, re.VERBOSE | re.IGNORECASE)


    parsed_pto_data = []

    for entry in pto_data:
        summary = entry.get('summary', '')
        match = pattern.match(summary)

        if match:
            parsed = match.groupdict()
            parsed['status'] = entry.get('status')
            parsed_pto_data.append(parsed)

    # print(parsed_pto_data)

#============================================================================================================================================
# SPLIT DATA INTO PTO HAPPENING AND PTO REQUESTED

    pto_happening = []
    pto_requests = []

    for entry in parsed_pto_data:
        if entry['status'] in ["Approved", "Added to LCS Calendar", "Time Checked"]:
            pto_happening.append(entry)
        elif entry['status'] == 'Needs Approval':
            pto_requests.append(entry)

    return {
        'all': parsed_pto_data, 
        'happening': pto_happening, 
        'requests': pto_requests
        }
