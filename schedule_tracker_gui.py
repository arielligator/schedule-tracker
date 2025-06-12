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

# ============================================================================
# APPLY TIME FILTERS
filtered_df = dynamic.filter_df()

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

with data:
    st.dataframe(
        filtered_df.style.format({
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
        c = sum(
            interval_overlaps(
                parse_time(row["Start Time"]),
                parse_time(row["End Time"]),
                window_start,
                window_end
            )
            for _, row in df_base.iterrows()
        )


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

            filtered_people = df_base[
                df_base.apply(
                    lambda r: interval_overlaps(
                        parse_time(r["Start Time"]),
                        parse_time(r["End Time"]),
                        selected_start,
                        selected_end
                    ),
                    axis=1
                )
            ]

            # Only show names (assumes column is named "Employee" or "Name")
            if "Employee" in filtered_people.columns:
                st.dataframe(filtered_people[["Employee"]], use_container_width=True)
            elif "Name" in filtered_people.columns:
                st.dataframe(filtered_people[["Name"]], use_container_width=True)
            else:
                st.warning("Couldn't find a 'Name' or 'Employee' column to display.")

# ============================================================================
# ON CALL

oncall_toggle = st.toggle("On Call")
if oncall_toggle:
    st.dataframe(df_oncall, use_container_width=True)

# ============================================================================
# PTO VIEWER

from pto import fetch_pto_tickets, clean_api_response, clear_pto_cache

on = st.toggle("PTO Viewer")

if on:
    if st.button('Refresh PTO Data'):
        clear_pto_cache()
        st.rerun()

    with st.spinner('Loading PTO data...'):
        all_tickets = fetch_pto_tickets()
        pto = clean_api_response(all_tickets)
        pto_all = pto["all"]
        pto_happening = pto["happening"]
        pto_requests = pto["requests"]

    # Convert 'Days' strings to lists
    for entry in pto_requests:
        entry['Days'] = [d.strip() for d in entry['Days'].split(',')]

    for entry in pto_happening:
        entry['Days'] = [d.strip() for d in entry['Days'].split(',')]

    for i, request in enumerate(pto_requests):
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
                    if any(day in request['Days'] for day in happening['Days']) and
                       request['Team'] == happening['Team']
                ]

                # Show toggle if there are overlaps
                if overlaps:
                    overlap = st.toggle("View Overlapping PTO", key=f"overlap_toggle_{i}")
                else:
                    st.markdown("*No overlapping PTO*")

            # Right side: show overlaps if toggle is on
            with pto2:
                if overlaps and overlap:
                    for happening in overlaps:
                        if not happening.get("TimeRange"):
                            happening.pop("TimeRange", None)
                        st.dataframe(
                            {k: ', '.join(v) if isinstance(v, list) else v for k, v in happening.items()},
                            use_container_width=True
                        )
