# IMPORTS/CONFIGS
import streamlit as st
import pandas as pd
from datetime import datetime, date, time
from schedule_tracker import df, df_oncall
from streamlit_dynamic_filters import DynamicFilters
import re
import os, time
import requests


st.set_page_config(
    layout="wide",
    page_title="Schedule Tracker",
    page_icon="📅",
)

# force the whole process to treat “local” as America/New_York
os.environ['TZ'] = 'America/New_York'
if hasattr(time, "tzset"):
    time.tzset()

# ============================================================================
# CONTAINERS / LAYOUT
header = st.container()
filters = st.container()
data, counter, buttons = st.columns([3.5,.9,1.2])
count = st.container()

with header:
    st.title('LIT Schedule Tracker')
    today = date.today()
    if today.month == 6 and today.day == 3:
        st.subheader('✨ Happy Birthday, Judy!  🎉 🎂 🎈')

# ============================================================================
# PASSWORD PROTECTION

# Persistent login
from streamlit_cookies_controller import CookieController, RemoveEmptyElementContainer

controller = CookieController()
RemoveEmptyElementContainer()

login_cookie = st.secrets["cookie_auth"]["password"]
token = controller.get(login_cookie)

if token and not st.session_state.get("authenticated", False):
    st.session_state["authenticated"] = True

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "password_tried" not in st.session_state:
        st.session_state["password_tried"] = False

    with st.sidebar.form(key="login_form"):
        st.text('Enter the password!')
        password = st.text_input("Password:", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if password == st.secrets["auth"]["password"]:
            st.session_state["authenticated"] = True
        else:
            st.session_state["password_tried"] = True
            st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        # persist a cookie for 14 days
        controller.set(login_cookie, "yes", max_age=14*24*60*60)
        st.sidebar.success("Access Granted")
        return True
    elif st.session_state["password_tried"]:
        st.sidebar.error("Incorrect password. Try again.")
        return False
    else:
        return False

if not check_password():
    st.stop()


# ============================================================================
# HELPER FUNCTIONS

# Normalize category values
def clean_category(x):
    clean_names = (
        x.dropna()
         .astype(str)
         .str.strip()
    )
    return sorted(clean_names.unique())

# Clean up user-inputted times (supports formats like 'HH:MM', 'HHa', 'hham', 'HH:MMAM', etc.)
def parse_time(s: str):
    """Try parsing a variety of time formats; returns a datetime.time or raises ValueError."""
    s_clean = s.strip().lower().replace('.', '').replace(' ', '')
    # List of strptime formats to try (am/pm variants first)
    formats = [
        "%I:%M:%S%p",  # '4:30:15pm'
        "%I:%M%p",     # '4:30pm'
        "%I%p",        # '4pm' or '4p'
        "%H:%M:%S",    # '16:30:15'
        "%H:%M",       # '16:30'
    ]
    for fmt in formats:
        try:
            # strptime expects uppercase AM/PM when using %p
            return datetime.strptime(s_clean.upper(), fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Time '{s}' not in a supported format (e.g. '4p', '4pm', '04:00', '16:00', '4:30pm')")

def on_duty(start: time, end: time, t: time) -> bool:
    # regular shift (e.g. 9am–5pm)
    if start <= end:
        return start <= t < end
    # overnight shift (e.g. 10pm–6am)
    return t >= start or t < end

def interval_overlaps(start1: time, end1: time, start2: time, end2: time) -> bool:
    """Return True if [start1→end1) overlaps [start2→end2), 
       where either interval may wrap past midnight."""
    def to_minutes(t):
        return t.hour*60 + t.minute
    s1, e1 = to_minutes(start1), to_minutes(end1)
    s2, e2 = to_minutes(start2), to_minutes(end2)

    # break a wrap-around interval into two linear segments
    segs1 = [(s1, e1)] if s1 < e1 else [(s1, 24*60), (0, e1)]
    segs2 = [(s2, e2)] if s2 < e2 else [(s2, 24*60), (0, e2)]

    # standard overlap test on any pair of segments
    for a0, a1 in segs1:
        for b0, b1 in segs2:
            if a0 < b1 and b0 < a1:
                return True
    return False

# ============================================================================
# SET UP DYNAMIC FILTERS

dynamic = DynamicFilters(
    df,
    filters=["Name", "Team", "Location", "Weekday"],
    filters_name="filters"
)


# ============================================================================
# UI FILTERS (Five-column layout)
with filters:
    names_col, depts_col, locs_col, days_col, times_col = st.columns(5, gap="small")

    # NAME FILTER
    with names_col:
        sel = st.multiselect(
            "Name(s)",
            options=clean_category(df["Name"]),
            default=None,
            key="name_sel",
            placeholder="Name(s)",
            label_visibility="collapsed"
        )
        st.session_state['filters']['Name'] = [] if "All" in sel else sel

    # TEAM FILTER
    with depts_col:
        sel = st.multiselect(
            "Team(s)",
            options=clean_category(df["Team"]),
            default=None,
            key="team_sel",
            placeholder="Team(s)",
            label_visibility="collapsed"
        )
        st.session_state['filters']['Team'] = sel

    # LOCATION FILTER
    with locs_col:
        sel = st.multiselect(
            "Location(s)",
            options=clean_category(df["Location"]),
            default=None,
            key="loc_sel",
            placeholder="Location(s)",
            label_visibility="collapsed"
        )
        st.session_state['filters']['Location'] = sel

    # WEEKDAY FILTER
    with days_col:
        # include 'Today' option
        weekdays = ['Today','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        sel = st.multiselect(
            "Day(s)",
            options=weekdays,
            default=None,
            key="day_sel",
            placeholder="Day(s)",
            label_visibility="collapsed"
        )
        # after your multiselect (key="day_sel"), grab the selections:
        sel = st.session_state['day_sel']  # e.g. ["Monday", "Wednesday"]

        # build a list of all matching "Work Days" strings
         # or logic (includes this day OR that day)
    #     matches = [
    #     wd
    #     for wd in df['Work Days'].dropna().astype(str)
    #     if all(day in wd for day in sel)
    # ]
        # and logic (includes this day AND that day)
        matches = [
        wd
        for wd in df['Work Days'].dropna().astype(str)
        if all(day in wd for day in sel)
    ]

    # store it back in your filters
    st.session_state['filters']['Work Days'] = matches


    # TIME FILTER (handled separately)
    with times_col:
        mode = st.selectbox(
            "",
            ["All", "Now", "Custom Time", "Custom Time Range"],
            key="time_mode",
            label_visibility="collapsed"
        )
        if mode == "All":
            st.write("Showing **all** times.")
        elif mode == "Now":
            now = datetime.now().time()
            st.write(f"Current time: **{now.strftime('%I:%M %p').lstrip('0')}**")
        elif mode == "Custom Time":
            txt = st.text_input(
                "Enter time (e.g. '4p', '4pm', '16:00')", 
                placeholder="Enter Time", 
                key="custom_time_txt",
                label_visibility="collapsed"
            )
            if txt:
                try:
                    t = parse_time(txt)
                    st.success(f"✔ Parsed time: **{t.strftime('%I:%M %p').lstrip('0')}**")
                except ValueError as e:
                    st.error(str(e))
        else:  # Custom Time Range
            c1, c2 = st.columns(2)
            start_txt = c1.text_input("Start (e.g. '4p' or '09:00')", value="09:00", key="start_txt")
            end_txt   = c2.text_input("End   (e.g. '5pm' or '17:00')", value="17:00", key="end_txt")
            valid = True
            try:
                start_t = parse_time(start_txt)
            except ValueError as e:
                c1.error(str(e)); valid = False
            try:
                end_t = parse_time(end_txt)
            except ValueError as e:
                c2.error(str(e)); valid = False
            if valid and end_t < start_t:
                st.error("End time must be ≥ start time.")
            elif valid:
                st.success(f"Range: **{start_t.strftime('%H:%M')}** – **{end_t.strftime('%H:%M')}**")

    # COMPANIES FILTER
    # Normalize the 'Companies' column and split into a list
    df['Companies'] = df['Companies'].fillna('').astype(str)
    # Aggressive normalization: strip whitespace around each company name
    df['Company List'] = (
        df['Companies']
        .str.split(',')
        .apply(lambda lst: [c.strip() for c in lst if c.strip()])
    )
        
    # extract unique companies
    all_companies = sorted({c for sublist in df['Company List'] for c in sublist})

    with names_col:
        # Use comp_sel as the key
        st.multiselect(
            "Companies",
            options=clean_category(df['Companies'].str.split(',').explode()),
            key="comp_sel",
            placeholder="Companies",
            label_visibility="collapsed"
        )

        # Grab the selected companies from session_state
        sel = st.session_state.get('comp_sel', [])

        # Match any row where at least one selected company is present
        matches = [
            row for row in df['Companies'].dropna().astype(str)
            if any(company.strip() in row for company in sel)
        ]

        st.session_state['filters']['Companies'] = matches

    # VIP FILTER
    with depts_col:
        sel = st.multiselect(
        "VIP",
        options=clean_category(df["Handles VIP"]),
        default=None,
        key="vip_sel",
        placeholder="Handles VIP",
        label_visibility="collapsed"
    )
    st.session_state['filters']['Handles VIP'] = sel



# ============================================================================
# APPLY TIME FILTERS
filtered_df = dynamic.filter_df()

# for col, vals in self.active_filters.items():
#     if col == "Companies":
#         df = df[df["Companies"].isin(vals)]


mode = st.session_state.get('time_mode', 'All')
if mode == "Now":
    now = datetime.now().time()

    df_base = dynamic.filter_df()

    filtered_df = df_base[
        df_base.apply(
            lambda row: on_duty(
                parse_time(row["Start Time"]),
                parse_time(row["End Time"]),
                now
            ),
            axis=1
        )
    ]
    st.write(f"Current Time: {now.strftime('%I:%M %p').lstrip('0')}")

elif mode == "Custom Time":
    txt = st.session_state.get("custom_time_txt", "")
    if txt:
        t = parse_time(txt)
        df_base = dynamic.filter_df(except_filter="Time of Day")
        filtered_df = df_base[
            df_base.apply(lambda row: on_duty(
                parse_time(row["Start Time"]),
                parse_time(row["End Time"]),
                t
            ), axis=1)
        ]
        st.success(f"✔ Parsed time: **{t.strftime('%I:%M %p').lstrip('0')}**")

elif mode == "Custom Time Range":
    start_txt = st.session_state.get("start_txt", "")
    end_txt   = st.session_state.get("end_txt", "")
    if start_txt and end_txt:
        s = parse_time(start_txt)
        e = parse_time(end_txt)
        if e < s and not True:
            st.error("End time must be ≥ start time.")
        else:
            st.success(f"Range: **{s.strftime('%I:%M %p').lstrip('0')}** – **{e.strftime('%I:%M %p').lstrip('0')}**")

            df_base = dynamic.filter_df()

            # keep rows whose shift_start ≤ filter_end  AND  shift_end ≥ filter_start
            filtered_df = df_base[
                df_base.apply(
                    lambda row: interval_overlaps(
                        parse_time(row["Start Time"]),
                        parse_time(row["End Time"]),
                        s, e
                    ),
                    axis=1
                )
            ]


# ============================================================================
# RENDER DATA & VISUALS

# formatter: adds a space before am/pm and uppercases it
def format_lunch(s: str) -> str:
    return re.sub(
        r'(?i)(\d{1,2}:\d{2})(am|pm)$',               # find “H:MMam” or “HH:MMpm” at end
        lambda m: f"{m.group(1)} {m.group(2).upper()}",  # rebuild as “H:MM AM” / “HH:MM PM”
        s.strip()
    )

df_display = filtered_df.drop(columns=['Company List'])

with data:
    st.dataframe(
        df_display.style.format({
            'Start Time': lambda s: parse_time(s).strftime('%I:%M %p').lstrip('0'),
            'End Time': lambda s: parse_time(s).strftime('%I:%M %p').lstrip('0'),
            'Lunch': format_lunch,
        }), use_container_width=True)


with count:
    st.text(f"Total rows: {len(filtered_df)}")

#----------------------------------------------------------------------------
# EXTRACT COUNTS AND RENDER DATA

from datetime import time

def interval_overlaps(start1: time, end1: time, start2: time, end2: time) -> bool:
    """Return True if [start1→end1) overlaps [start2→end2), 
       where either interval may wrap past midnight."""
    def to_minutes(t):
        return t.hour * 60 + t.minute

    s1, e1 = to_minutes(start1), to_minutes(end1)
    s2, e2 = to_minutes(start2), to_minutes(end2)

    # break-wrap intervals into linear segments
    segs1 = [(s1, e1)] if s1 < e1 else [(s1, 24*60), (0, e1)]
    segs2 = [(s2, e2)] if s2 < e2 else [(s2, 24*60), (0, e2)]

    # test any pair for overlap
    for a0, a1 in segs1:
        for b0, b1 in segs2:
            if a0 < b1 and b0 < a1:
                return True
    return False


# TIME RANGE COUNTER
with counter:
    # ① base DataFrame (all filters except time)
    df_base = dynamic.filter_df()
    mode = st.session_state.get("time_mode", "All")
    hours_to_plot = []

    if mode == "Now":
        t = datetime.now().time()
        hours_to_plot = [t.hour]
    
    elif mode == "Custom Time":
        txt = st.session_state.get("custom_time_txt", "")
        if txt:
            try:
                t = parse_time(txt)
                hours_to_plot = [t.hour]
            except ValueError:
                hours_to_plot = []
    
    elif mode == "Custom Time Range":
        start_txt = st.session_state.get("start_txt", "")
        end_txt = st.session_state.get("end_txt", "")
        if start_txt and end_txt:
            try:
                s = parse_time(start_txt)
                e = parse_time(end_txt)
                hours_to_plot = list(range(s.hour, e.hour))
            except ValueError:
                hours_to_plot = []
    
    else: # All
        hours_to_plot = list(range(24))
    
    # if nothing valid, bail out
    # if not hours_to_plot:
    #     st.write("Enter a valid time (or range) above.")
    #     return

    # build counts and labels only for filtered hours
    counts = []
    labels = []
    start_times = []
    end_times = []

    for h in hours_to_plot:
        window_start = time(h, 0)
        window_end   = time((h + 1) % 24, 0)

        # count overlaps in [h:00 → h+1:00)
        c = 0
        for _, row in df_base.iterrows():
            try:
                work_start = parse_time(row["Start Time"])
                work_end   = parse_time(row["End Time"])

                lunch_raw = row.get("Lunch", "").strip()
                has_lunch = lunch_raw and lunch_raw.lower() != "none"

                if has_lunch:
                    lunch_start = parse_time(lunch_raw)
                    lunch_end = (datetime.combine(datetime.today(), lunch_start) + pd.Timedelta(hours=1)).time()
                else:
                    lunch_start = lunch_end = None

                if interval_overlaps(work_start, work_end, window_start, window_end):
                    if lunch_start and interval_overlaps(lunch_start, lunch_end, window_start, window_end):
                        continue  # skip this row — on lunch
                    c += 1
            except Exception:
                continue  # skip rows with bad formatting




        counts.append(c)
        start_times.append(window_start)
        end_times.append(window_end)

        # build the label without %-flags
        start_label = window_start.strftime('%I %p').lstrip('0')
        end_label   = window_end.strftime(  '%I %p').lstrip('0')
        labels.append(f"{start_label}–{end_label}")

    counts = [int(c) if not isinstance(c, (int, float)) else c for c in counts]

    # ✅ collapse if all 0s
    if counts and all(c == 0 for c in counts):
        labels = ["No Data"]
        counts = [0]
        start_times = [time(0, 0)]
        end_times = [time(0, 0)]

    # ✅ now all lists have matching lengths
    df_ranges = pd.DataFrame({
        "Hour Range": labels,
        "Count": counts,
        "Start Time": start_times,
        "End Time": end_times
    })

    # ③ assemble and display
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    import streamlit as st
    import pandas as pd
    from datetime import time

    # Build your summary DataFrame
    # df_ranges = pd.DataFrame({
    #     "Hour Range": labels,
    #     "Count": counts,
    #     "Start Time": [time(h, 0) for h in hours_to_plot],
    #     "End Time": [time((h + 1) % 24, 0) for h in hours_to_plot]
    # })

    # Set up AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_ranges[["Hour Range", "Count"]])
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    grid_options = gb.build()

    # Display AgGrid
    grid_response = AgGrid(
        df_ranges,
        gridOptions=grid_options,
        height=400,
        width='100%',
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )

    # Handle selected row
    with buttons:
        selected_df = pd.DataFrame(grid_response["selected_rows"])

        if not selected_df.empty:
            row = selected_df.iloc[0]
            selected_label = row["Hour Range"]

            # Convert string → time object
            selected_start = datetime.strptime(row["Start Time"], "%H:%M:%S").time()
            selected_end   = datetime.strptime(row["End Time"], "%H:%M:%S").time()

            st.text(f"People working {selected_label}:")

            def get_status(row):
                try:
                    work_start = parse_time(row["Start Time"])
                    work_end   = parse_time(row["End Time"])

                    lunch_raw = row.get("Lunch", "").strip()
                    has_lunch = lunch_raw and lunch_raw.lower() != "none"

                    if has_lunch:
                        lunch_start = parse_time(lunch_raw)
                        lunch_end = (datetime.combine(datetime.today(), lunch_start) + pd.Timedelta(hours=1)).time()
                    else:
                        lunch_start = lunch_end = None

                    if interval_overlaps(work_start, work_end, selected_start, selected_end):
                        if lunch_start and interval_overlaps(lunch_start, lunch_end, selected_start, selected_end):
                            return "At Lunch"
                        return "Working"
                    return None
                except Exception:
                    return None

            # Add status column
            df_base["Status"] = df_base.apply(get_status, axis=1)

            # Only include rows where someone is working (with or without lunch)
            filtered_people = df_base[df_base["Status"].notnull()]

            if "Employee" in filtered_people.columns:
                st.dataframe(filtered_people[["Employee", "Status"]], use_container_width=True)
            elif "Name" in filtered_people.columns:
                st.dataframe(filtered_people[["Name", "Status"]], use_container_width=True)
            else:
                st.warning("Couldn't find a 'Name' or 'Employee' column to display.")



# ============================================================================
# ON CALL

oncall_toggle = st.toggle("On Call")
if oncall_toggle:
    st.dataframe(df_oncall, use_container_width=True)


# ============================================================================
# DISPATCH
import requests
from datetime import datetime
import pytz

DISPATCH_BASE_URL = "https://cw.lincolncomputers.com/v4_6_release/apis/3.0"

HEADERS = {
    "clientid": st.secrets['cw_pto']['clientid'],
    "Authorization": st.secrets['cw_pto']['auth_header'],
    "Accept": "application/json"
}

# === GET TODAY'S DATE RANGE ===
tz = pytz.timezone("America/New_York")  # adjust if needed
today = datetime.now(tz).date()
start_of_day = f"[{today}T00:00:00]"
end_of_day = f"[{today}T23:59:59]"

# === 1. GET SCHEDULE ENTRIES FOR TODAY ===
schedule_url = f"{DISPATCH_BASE_URL}/schedule/entries"
schedule_conditions = f"dateStart<={end_of_day} and dateEnd>={start_of_day}"
params_schedule = {
    "conditions": schedule_conditions,
    "pageSize": 1000
}

schedule_resp = requests.get(schedule_url, headers=HEADERS, params=params_schedule)
schedule_entries = schedule_resp.json()

# Create a mapping from objectId (ticket number) to selected fields
schedule_map = {
    entry["objectId"]: {
        "member": entry.get("member", {}).get("name"),
        "dateStart": entry.get("dateStart"),
        "dateEnd": entry.get("dateEnd")
    }
    for entry in schedule_entries
}

# === 2. GET TICKETS WITH STATUS "Scheduled Onsite" ===
tickets_url = f"{DISPATCH_BASE_URL}/service/tickets"
params_tickets = {
    "conditions": 'status/name="Scheduled Onsite"',
    "pageSize": 1000
}

tickets_resp = requests.get(tickets_url, headers=HEADERS, params=params_tickets)
tickets = tickets_resp.json()

# === 3. FILTER & COMBINE DATA ===
results = []

for ticket in tickets:
    ticket_id = ticket.get("id")
    if ticket_id in schedule_map:
        combined = {
            "ticket_id": ticket_id,
            "company": ticket.get("company", {}).get("name"),
            "site": ticket.get("site", {}).get("name"),
            "summary": ticket.get("summary"),
            **schedule_map[ticket_id]
        }
        results.append(combined)


dispatch_df = pd.DataFrame(results)
dispatch_df_order = ['member','summary','company','site','dateStart','dateEnd']
dispatch_df = dispatch_df[dispatch_df_order]
print(dispatch_df.columns.tolist())
print(dispatch_df["dateStart"].head(5).tolist())


from datetime import datetime
import pytz

utc = pytz.utc
eastern = pytz.timezone("America/New_York")

def extract_date_and_times(start, end):
    try:
        dt_start_utc = utc.localize(datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ"))
        dt_end_utc = utc.localize(datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ"))

        dt_start = dt_start_utc.astimezone(eastern)
        dt_end = dt_end_utc.astimezone(eastern)

        # Check for all-day event: both times at 00:00:00 UTC
        if dt_start_utc.time() == dt_end_utc.time() == datetime.strptime("00:00:00", "%H:%M:%S").time():
            date_formatted = dt_start.strftime("%B %d").lstrip("0").replace(" 0", " ")
            return pd.Series([date_formatted, "All Day", "All Day"])

        date_formatted = dt_start.strftime("%B %d").lstrip("0").replace(" 0", " ")
        start_time = dt_start.strftime("%I:%M %p").lstrip("0")
        end_time = dt_end.strftime("%I:%M %p").lstrip("0")
        return pd.Series([date_formatted, start_time, end_time])
    
    except Exception as e:
        print(f"Error parsing times: {start}, {end} ({e})")
        return pd.Series([None, None, None])


# Apply it across the DataFrame
dispatch_df[["dateFormatted", "startTimeFormatted", "endTimeFormatted"]] = dispatch_df.apply(
    lambda row: extract_date_and_times(row["dateStart"], row["dateEnd"]), axis=1
)


dispatch_df.rename(columns={'member':'Name', 'summary':'Summary', 'company':'Company', 'site':'Site','dateStartFormatted':'Date','startTimeFormatted':'Start Time','endTimeFormatted':'End Time'}, inplace=True)
dispatch_df_cleaned_list = ['Name','Summary','Company','Site','Start Time','End Time']
dispatch_df_cleaned = dispatch_df[dispatch_df_cleaned_list]

dispatch_toggle = st.toggle("Dispatch")
if dispatch_toggle:
    st.dataframe(dispatch_df_cleaned, use_container_width=True)



# ============================================================================
# PTO VIEWER

from pto import fetch_pto_tickets, clean_api_response, clear_pto_cache

on = st.toggle("PTO Viewer")

if on:
    if st.button('Refresh PTO Data'):
        clear_pto_cache()
        st.rerun()

    with st.spinner('Loading PTO data...'):
        all_tickets = fetch_pto_tickets(extra_params=None)
        pto = clean_api_response(all_tickets)
        pto_all = pto["all"]
        pto_happening = pto["happening"]
        pto_requests = pto["requests"]

        # Extract unique Team and Location values from PTO data
        pto_teams = sorted({req['Team'] for req in pto_requests if req.get('Team')})
        pto_locations = sorted({req['Location'] for req in pto_requests if req.get('Location')})

        

    # Convert 'Days' strings to lists
    for entry in pto_requests:
        entry['Days'] = [d.strip() for d in entry['Days'].split(',')]

    for entry in pto_happening:
        entry['Days'] = [d.strip() for d in entry['Days'].split(',')]

    # month filter
    import calendar

    # Extract unique month numbers from all requests
    month_numbers = set()
    for request in pto_requests:
        for day in request.get('Days', []):
            try:
                month = int(day.split('/')[0].strip())
                if 1 <= month <= 12:
                    month_numbers.add(month)
            except (ValueError, IndexError):
                continue  # skip malformed entries

    # --- Build month name lookup ---
    month_name_lookup = {num: calendar.month_name[num] for num in month_numbers}
    month_names_sorted = [month_name_lookup[m] for m in sorted(month_name_lookup)]

    # --- Session key names ---
    WIDGET_KEY = "selected_month_widget"
    STATE_KEY = "selected_month_state"

    # --- Initialize session state if not set ---
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = month_names_sorted

    with st.expander("🕒 PTO Requests"):
        # --- Reset button BEFORE multiselect to take effect before widget renders ---
        # if st.button("Reset Month Filter"):
        #     st.session_state[STATE_KEY] = month_names_sorted
        #     st.session_state[WIDGET_KEY] = month_names_sorted
        #     st.rerun()

        pto_month_filter, pto_team_filter, pto_loc_filter = st.columns(3)

        # --- Multiselect with separate widget key ---
        with pto_month_filter:
            valid_months = month_names_sorted
            default_months = [m for m in st.session_state[STATE_KEY] if m in valid_months]

            selected_months = st.multiselect(
                "Filter requests by month",
                options=valid_months,
                default=default_months,
                key=WIDGET_KEY
            )

        # Team Filter
        with pto_team_filter:
            selected_pto_teams = st.multiselect(
                "Filter by Team",
                options=pto_teams,
                default=pto_teams,
                key="pto_team_filter"
            )

        # Location Filter
        with pto_loc_filter:
            selected_pto_locations = st.multiselect(
                "Filter by Location",
                options=pto_locations,
                default=pto_locations,
                key="pto_location_filter"
            )

        # --- Convert month names to numbers ---
        selected_month_nums = [
            num for num, name in month_name_lookup.items()
            if name in selected_months
        ]

        # Filter PTO requests by selected months, team, and location
        filtered_requests = [
            req for req in pto_requests
            if (
                (
                    not selected_month_nums or
                    any(
                        day.split('/')[0].isdigit() and int(day.split('/')[0]) in selected_month_nums
                        for day in req.get('Days', [])
                    )
                )
                and (
                    not selected_pto_teams or
                    req.get("Team") in selected_pto_teams
                )
                and (
                    not selected_pto_locations or
                    req.get("Location") in selected_pto_locations
                )
            )
        ]

        # display requests
        for i, request in enumerate(filtered_requests):
            with st.container():
                pto1, pto2 = st.columns(2)

                # Left side: show request
                with pto1:
                    if not request.get("TimeRange"):
                        request.pop("TimeRange", None)
                    st.dataframe(
                        {k: ', '.join(v) if isinstance(v, list) else v for k, v in request.items()},
                        use_container_width=True
                    )

                    # Precompute overlaps
                    overlaps = [
                        happening for happening in pto_happening
                        if any(day in request['Days'] for day in happening['Days']) and (
                            (
                                request['Team'] in ['Help Desk', 'Service Desk'] and
                                happening['Team'] in ['Help Desk', 'Service Desk']
                            ) or
                            (
                                request['Team'] not in ['Help Desk', 'Service Desk'] and
                                happening['Team'] == request['Team']
                            )
                        )
                    ]

                    # Precompute overlaps with other PTO requests
                    overlaps_with_requests = [
                        other for other in pto_requests
                        if other is not request and any(day in request['Days'] for day in other['Days']) and (
                            (
                                request['Team'] in ['Help Desk', 'Service Desk'] and
                                other['Team'] in ['Help Desk', 'Service Desk']
                            ) or
                            (
                                request['Team'] not in ['Help Desk', 'Service Desk'] and
                                other['Team'] == request['Team']
                            )
                        )
                    ]

                    # Show toggle if there are overlaps
                    overlap = False  # Ensure it's always defined

                    if overlaps or overlaps_with_requests:
                        overlap = st.toggle("View Overlapping PTO", key=f"overlap_toggle_{i}")
                    else:
                        st.markdown("*No overlapping PTO*")


                    # Right side: show overlaps if toggle is on
                    with pto2:
                        if overlaps and overlap:
                            st.markdown("Overlapping Happening:")
                            for happening in overlaps:
                                if not happening.get("TimeRange"):
                                    happening.pop("TimeRange", None)
                                st.dataframe(
                                    {k: ', '.join(v) if isinstance(v, list) else v for k, v in happening.items()},
                                    use_container_width=True
                                )

                        if overlaps_with_requests and overlap:
                            st.markdown("Overlapping Requests:")
                            for other in overlaps_with_requests:
                                if not other.get("TimeRange"):
                                    other.pop("TimeRange", None)
                                st.dataframe(
                                    {k: ', '.join(v) if isinstance(v, list) else v for k, v in other.items()},
                                    use_container_width=True
                                )


    # PTO COUNTER AND CALENDAR
    pto_counter_col, pto_cal_col = st.columns([1,1])

    # PTO COUNTER
    pto_counter_url = "https://cw.lincolncomputers.com/v4_6_release/apis/3.0/service/tickets?conditions=board/id=42 AND status/name not contains \"Closed\"&orderby=id desc&pageSize=1000"

    headers = {
    'clientid': st.secrets['cw_pto']['clientid'],
    'Authorization': st.secrets['cw_pto']['auth_header']
    }

    response = requests.request("GET", pto_counter_url, headers=headers)

    pto_entries = response.json()

    pto_pattern = re.compile(r"""
        (?P<Name>[^\(]+)                           # Name: everything before first (
        \(
            (?P<Location>[^-]+?)\s*-\s*(?P<Team>[^)]+) # Location and Team inside ()
        \)
        .*?                                        # Any words (like PTO/Personal)
        (?P<Days>(\d{1,2}/\d{1,2})(,\s*\d{1,2}/\d{1,2})*) # Dates like 8/11, 8/12
    """, re.VERBOSE | re.IGNORECASE)

    parsed_pto_data = []

    for entry in pto_entries:
        summary = entry.get('summary', '')
        match = pto_pattern.match(summary)

        if match:
            parsed = match.groupdict()
            parsed['status'] = entry.get('status').get('name')
            parsed_pto_data.append(parsed)

    pto_requests = []
    pto_happening = []

    for entry in parsed_pto_data:
        if "Needs Approval" in entry['status']:
            pto_requests.append(entry)
        else:
            pto_happening.append(entry)
    
    from collections import defaultdict

    # === Normalization helpers ===
    def normalize_team(team):
        return team.lower().replace(" ", "")

    def normalize_date(date):
        return date.strip()

    def build_counts(data):
        result = defaultdict(lambda: defaultdict(int))
        raw_date_map = {}  # Map normalized -> original for printing

        for entry in data:
            team_raw = entry["Team"].strip()
            team = normalize_team(team_raw)

            for date in entry["Days"].split(","):
                raw_date = date.strip()
                norm_date = normalize_date(raw_date)
                result[norm_date][team_raw] += 1
                raw_date_map[norm_date] = raw_date  # Keep original format

        return result, raw_date_map


    # === Count data ===
    counts_requests, date_map_req = build_counts(pto_requests)
    counts_happening, date_map_hap = build_counts(pto_happening)

    # === Overlapping dates ===
    overlapping_dates = set(counts_requests.keys()) & set(counts_happening.keys())

    # === Format helper ===
    def format_team_counts(counts_dict):
        return ", ".join(f"{count} {team}" for team, count in sorted(counts_dict.items()))

            
    with pto_counter_col:
        with st.expander("📊 PTO Requests vs Happening"):
            # === Display output ===
            for date in sorted(overlapping_dates):
                day = date_map_req.get(date, date)
                requested_str = format_team_counts(counts_requests[date])
                happening_str = format_team_counts(counts_happening[date])

                st.markdown(
                    f"**{day}**  \n"
                    f"Requested: {requested_str if requested_str else '—'}  \n"
                    f"Happening: {happening_str if happening_str else '—'}"
                )


# PTO CALENDAR
    # Build params from user date input
    from datetime import datetime, timedelta

    def build_summary_date_params(start_date, end_date):
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        # Format date fragments into summary search conditions
        year = str(start_date.year)
        summary_parts = [f"summary contains '{d.month}/{str(d.day).zfill(2)}'" for d in dates]
        summary_condition = " or ".join(summary_parts)

        condition_str = f"({summary_condition}) and dateEntered contains '{year}' and board/id=42"
        return {
            "conditions": condition_str,
            "pageSize": 1000
        }
    
    # helper function for pto search by date - extract team and location
    def extract_pto_team_location(text):
        pto_match = re.search(r"\(([^()-]+)\s*-\s*([^()]+)\)", text)

        if pto_match:
            pto_location = pto_match.group(1).strip()
            pto_team = pto_match.group(2).strip()

            return pto_location, pto_team
        return None, None

    with pto_cal_col:
        with st.expander("📅 PTO Ticket Search by Date Range"):

            with st.form("search_pto"):
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime.today())
                    pto_team = st.multiselect(
                        "Team(s)",
                        options=['CSC', 'Data Center', 'Documentation Specialist', 'East End', 'Escalations', 'Escalations/NOC', 'Escalations/Service Desk', 'Help Desk', 'HHAR', 'MGE', 'NOC', 'Onboarding', 'Project', 'Safe Horizon', 'Service Desk', 'Supervisor', 'QA', 'Warehouse'],
                    )
                with col2:
                    end_date = st.date_input("End Date (ignore for single-day search)", value=start_date)
                    pto_location = st.multiselect(
                        "Location(s)",
                        options=['Cyprus', 'East Hampton', 'HHAR', 'Hicksville', 'Kosovo', 'NYC', 'NYC/HHAR', 'Philippines', 'Remote/AU', 'Remote/FL', 'Remote/IN', 'Remote/NC', 'Remote/NJ', 'Remote/NY', 'Remote/SC', 'Remote/TX', 'Remote/WVA', 'Safe Horizon']
                    )

                submitted = st.form_submit_button("Search")

                if submitted:
                    if start_date > end_date:
                        st.error("Start date must be before or equal to end date.")
                    else:
                        params = build_summary_date_params(start_date, end_date)
                        try:
                            with st.spinner("Fetching PTO tickets..."):
                                tickets = fetch_pto_tickets(extra_params=params)

                                # normalize multiselect filters
                                selected_teams = [t.lower() for t in pto_team]
                                selected_locations = [l.lower() for l in pto_location]

                                # parse and filter tickets based on selected teams and location
                                pto_filtered = []
                                for t in tickets:
                                    location, team = extract_pto_team_location(t.get('summary', ''))
                                    if location and team:
                                        team_lower = team.lower()
                                        location_lower = location.lower()
                                        t['Team'] = team
                                        t['Location'] = location
                                        if (
                                            (not selected_teams or team_lower in selected_teams) and
                                            (not selected_locations or location_lower in selected_locations)
                                        ):
                                            pto_filtered.append(t)

                                st.success(f"Found {len(tickets)} ticket(s).")
                                if pto_filtered:
                                    for t in pto_filtered:
                                        summary = t.get("summary", "No summary")
                                        ticket_id = t.get("id", "N/A")
                                        status_name = t.get("status", {}).get("name", "No status")
                                        st.write(f"• **{summary.rstrip()}**  \nID: {ticket_id}  \nStatus: {status_name}")

                                else:
                                    st.info("No tickets match the selected filters.")
                        except Exception as e:
                            st.error(f"Error: {e}")
