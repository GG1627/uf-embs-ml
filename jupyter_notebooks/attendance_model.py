# %%
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jupyter
import supabase
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# %%
# get data
events_df = pd.read_csv("../data/events.csv")
event_attendance_df = pd.read_csv("../data/event_attendance.csv")

# %%
# simple checks

# event_attendance_df.info()
event_attendance_df.head()
# event_attendance_df.head()

# %%
# Safe merge that can be run multiple times
if "total_attendees" not in events_df.columns:
    # Get attendance counts per event
    attendance_counts = event_attendance_df.groupby("event_id").size().reset_index(name="total_attendees")
    
    # Merge with events_df (only keeping the total_attendees column)
    events_df = events_df.merge(attendance_counts[["event_id", "total_attendees"]], left_on="id", right_on="event_id", how="left")
    
    # Fill NaN with 0 and convert to int
    events_df["total_attendees"] = events_df["total_attendees"].fillna(0).astype(int)
    
    # Optional: Remove the redundant event_id column if it was added
    if "event_id" in events_df.columns:
        events_df = events_df.drop("event_id", axis=1)

# Now you can safely rerun this cell multiple times without errors
print(f"Events with attendance data: {events_df['total_attendees'].sum()}")

# %%
# check the new updated data and remove any unneeded columns
# events_df.head(20)

events_df = events_df.drop(columns=["id", "name", "code"], axis=1, errors="ignore")

events_df.head()

# %%
# feature engineering - date and time

# turn boolean columns into int
events_df["is_virtual"] = events_df["is_virtual"].astype(int)
events_df["food_present"] = events_df["food_present"].astype(int)

# one hot encode the event_type column
if "event_type" in events_df.columns:
    events_df = pd.get_dummies(
        events_df,
        columns=["event_type"],
        drop_first=True
    )

# turn the date and time columns into a datetime object with utc timezone
events_df["date"] = pd.to_datetime(events_df["date"], utc=True)
events_df["start_time"] = pd.to_datetime(events_df["start_time"], utc=True)
events_df["end_time"] = pd.to_datetime(events_df["end_time"], utc=True)

# convert to local timezone (est)
events_df["date_local"] = events_df["date"].dt.tz_convert("US/Eastern")
events_df["start_time_local"] = events_df["start_time"].dt.tz_convert("US/Eastern")
events_df["end_time_local"] = events_df["end_time"].dt.tz_convert("US/Eastern")

# get the useful information from datetime objects
events_df["weekday"] = events_df["date_local"].dt.weekday
events_df["month"] = events_df["date_local"].dt.month
events_df["day"] = events_df["date_local"].dt.day
events_df["start_hour"] = events_df["start_time_local"].dt.hour

events_df.head()

# %%
# define x and y variables (y is the target variable)

X = events_df.drop(columns=["date", "start_time", "end_time", "total_attendees", "date_local", "start_time_local", "end_time_local"])
y = events_df["total_attendees"]

X.info()

# %%
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit the model with training data
model = LinearRegression()
model.fit(X_train, y_train)

# %%
# evaluate the model

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R2: {r2_score(y_test, y_pred)}")

# plot the actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Attendees")
plt.ylabel("Predicted Attendees")

# %%
# predict a future event 
future_event = {
    "points": 2,
    "food_present": 1,
    "is_virtual": 0,
    "weekday": 2,        # Wednesday
    "month": 0,
    "day": 15,
    "start_hour": 17,
    "event_type_competition": 0,
    "event_type_fundraising": 0,
    "event_type_gbm": 1,          # <-- THIS is how you say "gbm"
    "event_type_industry_speaker": 0,
    "event_type_workshop": 0
}

# convert to dataframe
future_event_df = pd.DataFrame([future_event])

# make prediction
predicted_attendees = model.predict(future_event_df)

predicted_attendees


