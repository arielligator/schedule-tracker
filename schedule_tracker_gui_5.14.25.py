# IMPORTS/CONFIGS
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
from schedule_tracker import df
from streamlit_dynamic_filters import DynamicFilters
import re

st.set_page_config(
    layout="wide",
    page_title="Schedule Tracker",
    page_icon="📅",
)

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
data = st.container()
count = st.container()

with header:
    st.title('LIT Schedule Tracker 5.14')

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
            (filtered_df['End Time'].apply(parse_time)   >= t)
        ]
elif mode == "Custom Time Range":
    start_txt = st.session_state.get("start_txt", "")
    end_txt   = st.session_state.get("end_txt", "")
    if start_txt and end_txt:
        s = parse_time(st.session_state.get('start_txt', '09:00'))
        e = parse_time(st.session_state.get('end_txt', '17:00'))
        if e >= s:
            filtered_df = filtered_df[
                (filtered_df['Start Time'].apply(parse_time) >= s) &
                (filtered_df['End Time'].apply(parse_time)   <= e)
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