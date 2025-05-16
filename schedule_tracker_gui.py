# IMPORTS/CONFIGS
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from schedule_tracker import df
from streamlit_dynamic_filters import DynamicFilters
import re
from zoneinfo import ZoneInfo
import os, time


st.set_page_config(
    layout="wide",
    page_title="Schedule Tracker",
    page_icon="📅",
)

# force the whole process to treat “local” as America/New_York
os.environ['TZ'] = 'America/New_York'
time.tzset()

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

# ============================================================================
# SET UP DYNAMIC FILTERS

dynamic = DynamicFilters(
    df,
    filters=["Name", "Team", "Location", "Weekday"],
    filters_name="filters"
)

# ============================================================================
# CONTAINERS / LAYOUT
header = st.container()
filters = st.container()
data, counter = st.columns([5,1])
count = st.container()

with header:
    st.title('LIT Schedule Tracker')

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
# APPLY FILTERS
filtered_df = dynamic.filter_df()
mode = st.session_state.get('time_mode', 'All')
if mode == "Now":
    now = datetime.now().time()
    filtered_df = filtered_df[
        (filtered_df['Start Time'].apply(parse_time) <= now) &
        (filtered_df['End Time'].apply(parse_time)   >= now)
    ]
elif mode == "Custom Time":
    txt = st.session_state.get("custom_time_txt", "")
    if txt:  # only parse if non‐empty
        t = parse_time(st.session_state.get('custom_time_txt', '12:00'))
        filtered_df = filtered_df[
            (filtered_df['Start Time'].apply(parse_time) <= t) &
            (filtered_df['End Time'].apply(parse_time)   > t)
        ]
elif mode == "Custom Time Range":
    start_txt = st.session_state.get("start_txt", "")
    end_txt   = st.session_state.get("end_txt", "")
    if start_txt and end_txt:
        s = parse_time(start_txt)
        e = parse_time(end_txt)
        if e < s:
            st.error("End time must be ≥ start time.")
        else:
            st.success(f"Range: **{s.strftime('%I:%M %p').lstrip('0')}** – **{e.strftime('%I:%M %p').lstrip('0')}**")
            # keep rows whose shift_start ≤ filter_end  AND  shift_end ≥ filter_start
            filtered_df = filtered_df[
                (filtered_df['Start Time'].apply(parse_time) <= e) &
                (filtered_df['End Time'].apply(parse_time)   > s)
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

with counter:
    # ——————————————————————————————
    # 0️⃣ First, apply all your other filters (Name/Team/Location/Day) but skip Time
    #    (if you’re using streamlit_dynamic_filters; otherwise replace with your own filtering logic)
    filtered_df = dynamic.filter_df(except_filter="Time of Day")

    # ——————————————————————————————
    # 1️⃣ Decide which hour(s) to plot
    mode = st.session_state.get("time_mode", "All")

    if mode == "Now":
        now = datetime.now().time()
        hours_to_plot = [now.hour]

    elif mode == "Custom Time":
        txt = st.session_state.get("custom_time_txt", "")
        if txt:
            try:
                t = parse_time(txt)        # naive time, but frame is already Eastern
                hours_to_plot = [t.hour]
            except ValueError:
                hours_to_plot = []
        else:
            hours_to_plot = []

    elif mode == "Custom Time Range":
        start_txt = st.session_state.get("start_txt", "")
        end_txt   = st.session_state.get("end_txt", "")
        if start_txt and end_txt:
            try:
                s = parse_time(start_txt)
                e = parse_time(end_txt)
                hours_to_plot = list(range(s.hour, e.hour))
            except ValueError:
                hours_to_plot = []
        else:
            hours_to_plot = []

    else:  # “All”
        hours_to_plot = list(range(24))

    # ——————————————————————————————
    # 2️⃣ Count “on duty” per hour (all comparisons now between ints)
    if hours_to_plot:
        # Extract integer hours from your Eastern‐time strings
        filtered_df["start_hour"] = filtered_df["Start Time"].apply(lambda s: parse_time(s).hour)
        filtered_df["end_hour"]   = filtered_df["End Time"].  apply(lambda s: parse_time(s).hour)

        counts = [
            ((filtered_df["start_hour"] <= h) & (filtered_df["end_hour"] > h)).sum()
            for h in hours_to_plot
        ]
        counts_by_hour = pd.Series(counts, index=hours_to_plot, name="Employee Count")

        # 3️⃣ Format labels like “9am–10am”
        def fmt_hour(h):
            suffix = "am" if h < 12 else "pm"
            hour   = h % 12 or 12
            return f"{hour}{suffix}"
        def fmt_range(h):
            return f"{fmt_hour(h)}–{fmt_hour((h+1)%24)}"

        # 4️⃣ Build and display the table
        df_counts = counts_by_hour.to_frame().rename_axis("Hour Range")
        df_counts.index = df_counts.index.map(fmt_range)
        st.dataframe(df_counts, use_container_width=True)
    else:
        st.write("⏳ Enter a valid time (or range) above to show hourly counts.")


