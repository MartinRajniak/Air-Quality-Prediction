import pandas as pd
import holidays

def add_calendar_features(aqi_df: pd.DataFrame):
    svk_holidays = holidays.Slovakia()

    datetime_pd = aqi_df.index

    aqi_df["year"] = datetime_pd.year
    aqi_df["month"] = datetime_pd.month
    aqi_df["day_of_month"] = datetime_pd.day
    aqi_df["day_of_week"] = datetime_pd.dayofweek
    aqi_df["day_of_year"] = datetime_pd.dayofyear
    aqi_df["week_of_year"] = datetime_pd.isocalendar().week
    aqi_df["is_leap_year"] = datetime_pd.is_leap_year.astype(int)
    aqi_df["is_working_day"] = [int(svk_holidays.is_working_day(x)) for x in datetime_pd]
    aqi_df["is_feb29"] = ((aqi_df["month"] == 2) & (aqi_df["day_of_month"] == 29)).astype(int)

    return aqi_df