import requests
import streamlit as st

def fetch_pto_tickets():
    # Credentials (already base64 if you got them from Postman)
    clientid = st.secrets['cw_pto']['clientid']
    auth_header = st.secrets['cw_pto']['auth_header']

    # API base
    base_url = "https://cw.lincolncomputers.com/v4_6_release/apis/3.0/service/tickets"

    # Fixed parameters
    page_size = 1000
    page = 1
    all_tickets = []

    while True:
        params = {
            "conditions": "board/id=42",
            "orderby": "id desc",
            "pageSize": str(page_size),
            "page": str(page)
        }

        headers = {
            "clientid": clientid,
            "Authorization": auth_header,
            "Accept": "application/json"
        }

        # print(f"Fetching page {page}...")
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        tickets = response.json()

        if not tickets:
            # print("No more tickets.")
            break

        all_tickets.extend(tickets)
        page += 1
    
    return all_tickets

# print(f"Fetched {len(all_tickets)} tickets in total.")

# # Optional: write to file
# with open("all_tickets.json", "w", encoding="utf-8") as f:
#     json.dump(all_tickets, f, ensure_ascii=False, indent=2)

