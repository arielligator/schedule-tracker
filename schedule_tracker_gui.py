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

    # --- Reset button BEFORE multiselect to take effect before widget renders ---
    if st.button("Reset Month Filter"):
        st.session_state[STATE_KEY] = month_names_sorted
        st.session_state[WIDGET_KEY] = month_names_sorted
        st.rerun()

    pto_month_filter, pto_team_filter, pto_loc_filter = st.columns(3)

    # --- Multiselect with separate widget key ---
    with pto_month_filter:
        selected_months = st.multiselect(
            "Filter requests by month",
            options=month_names_sorted,
            default=st.session_state[STATE_KEY],
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


    # Filter PTO requests by selected months
    # Filter PTO requests by selected months, team, and location
    filtered_requests = [
        req for req in pto_requests
        if (
            any(
                day.split('/')[0].isdigit() and int(day.split('/')[0]) in selected_month_nums
                for day in req.get('Days', [])
            )
            and req.get("Team") in selected_pto_teams
            and req.get("Location") in selected_pto_locations
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
    from collections import defaultdict

    today = datetime.today()

    # --- Step 1: Build day → team → count, skipping past dates ---
    def build_future_day_team_counter(data):
        counter = defaultdict(lambda: defaultdict(int))
        for entry in data:
            team = entry.get("Team", "Unknown")
            for day_str in entry.get("Days", []):
                try:
                    day_date = datetime.strptime(day_str.strip(), "%m/%d").replace(year=today.year)
                    if day_date >= today:
                        counter[day_str.strip()][team] += 1
                except ValueError:
                    continue
        return counter

    requests_by_day_team = build_future_day_team_counter(pto_requests)
    happenings_by_day_team = build_future_day_team_counter(pto_happening)

    # --- Step 2: Sort by date ---
    def sorted_by_date(counter_dict):
        return sorted(counter_dict.items(), key=lambda x: datetime.strptime(x[0], '%m/%d'))

    # --- Step 3: Unified display ---
    def render_combined_day_team_list(requests, happenings):
        with pto_counter_col:
            with st.expander("📊 PTO Requests vs Happening"):
                for day, req_teams in sorted_by_date(requests):
                    # Format Requested section
                    requested_str = ', '.join(f"{count} {team}" for team, count in req_teams.items())

                    # Get Happening section only if that day exists
                    hap_teams = happenings.get(day, {})
                    happening_str = ', '.join(f"{count} {team}" for team, count in hap_teams.items())

                    st.markdown(f"**{day}**  \nRequested: {requested_str if requested_str else '—'}  \nHappening: {happening_str if happening_str else '—'}")

    # --- Display ---
    render_combined_day_team_list(requests_by_day_team, happenings_by_day_team)

# PTO CALENDAR
    # Build params from user date input
    from datetime import datetime, timedelta

    def build_summary_date_params(start_date, end_date):
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        # Format condition parts
        year = str(start_date.year)
        summary_parts = [f"summary contains '{d.month}/{str(d.day).zfill(2)}'" for d in dates]
        summary_condition = " or ".join(summary_parts)

        condition_str = f"({summary_condition}) and dateEntered contains '{year}' and board/id=42"
        return {
            "conditions": condition_str,
            "pageSize": 1000
        }
    
    with pto_cal_col:
        with st.expander("📅 PTO Ticket Search by Date Range"):

            with st.form("search_pto"):
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime.today())
                with col2:
                    end_date = st.date_input("End Date (ignore for single-day search)", value=start_date)


                submitted = st.form_submit_button("Search")

                if submitted:
                    if start_date > end_date:
                        st.error("Start date must be before or equal to end date.")
                    else:
                        params = build_summary_date_params(start_date, end_date)
                        try:
                            with st.spinner("Fetching PTO tickets..."):
                                tickets = fetch_pto_tickets(extra_params=params)
                                st.success(f"Found {len(tickets)} ticket(s).")
                                for t in tickets:
                                    st.write(f"• {t['summary']} (ID: {t['id']})")
                        except Exception as e:
                            st.error(f"Error: {e}")

