import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }

def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('\nHello! Let\'s explore some US bikeshare data!')
    # get user input for city (chicago, new york city, washington). HINT: Use a while loop to handle invalid inputs
    city = input("which city data would you like to explore from (chicago, new york city or washington) ?\n")
    while (city not in (CITY_DATA).keys()):
        print("\nPlease enter the city name appropriately!")
        city = input("\nwhich city data would you like to see from (chicago, new york city or washington) ?\n")
        break

    # get user input for month (all, january, february, ... , june)
    month = (input("\nwhich month would you like to filter by (all,January, February, March, April, May, June)?\nPlease type 'all' for no month filter\n")).title()
    if month not in ('All','January','February','March','April','May','June'):
        print("\nPlease enter valid month!")
        month = (input("\nwhich month would you like to filter by (all,January, February, March, April, May, June)?\nPlease type 'all' for no month filter\n")).title()

    # get user input for day of week (all, monday, tuesday, ... sunday)
    day = (input("\nwhich day would you like to filter by (all,Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)?\nPlease type 'all' for no day filter\n")).title()
    if day not in ('All','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'):
        print("\nPlease enter valid day!")
        day = (input("\nwhich day would you like to filter by (all,Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)?\n    Please type 'all' for no day filter\n")).title()

    print('-'*40)
    return city, month, day


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    # load data file into a dataframe
    df = pd.read_csv(CITY_DATA[city])

    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])

    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.weekday_name

    # filter by month if applicable
    if month != 'All':
        # use the index of the months list to get the corresponding int
        months = ['January', 'February', 'March', 'April', 'May', 'June']
        month = months.index(month) + 1

        # filter by month to create the new dataframe
        df = df[df['month'] == month]

    # filter by day of week if applicable
    if day != 'All':
        # filter by day of week to create the new dataframe
        df = df[df['day_of_week'] == day.title()]

    return df


def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # display the most common month
    popular_month = df['month'].mode()[0]
    print("\nThe most common month:", popular_month)

    # display the most common day of week
    popular_day = df['day_of_week'].mode()[0]
    print("\nThe most common day of week:", popular_day )

    # display the most common start hour
    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])

    # extract hour from the Start Time column to create an hour column
    df['hour'] = df['Start Time'].dt.hour

    # find the most common hour (from 0 to 23)
    popular_hour = df['hour'].mode()[0]

    print("\nMost common Start Hour:", popular_hour)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # display most commonly used start station
    popular_start_station = df['Start Station'].mode()[0]
    print("\nMost commonly used start station:", popular_start_station)

    # display most commonly used end station
    popular_end_station = df['End Station'].mode()[0]
    print("\nMost commonly used end station:", popular_end_station)

    # display most frequent combination of start station and end station trip
    df['Trip'] = df['Start Station'] + " , " + df['End Station']
    popular_trip = df['Trip'].mode()[0]
    print("\nMost frequent combination of start and end station trip:\n" , popular_trip)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # display total travel time
    total_travel = df['Trip Duration'].sum()
    print("\nTotal travel time:",total_travel)

    # display mean travel time
    mean_travel = df['Trip Duration'].mean()
    print("\nAverage travel time:", mean_travel)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df,city):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # Display counts of user types
    user_type_counts = df['User Type'].value_counts()
    print("\nCounts of user types:\n",user_type_counts)

    # Calculate gender_count if 'Gender' column is available
    cities = ('chicago','new york city')
    if (city in cities):
        gender_count = df['Gender'].value_counts()
        print("\nCounts of Gender:\n",gender_count)

        # Display earliest, most recent, and most common year of birth
        earliest_birthyear = df['Birth Year'].min()
        print("\nEarliest year of birth:", earliest_birthyear)
        recent_birthyear = df['Birth Year'].max()
        print("\nMost recent year of birth:", recent_birthyear)
        popular_birthyear = df['Birth Year'].mode()[0]
        print("\nMost common year of birth:", popular_birthyear)
    else:
        print("\nGender and Birth Year data are not available")
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def rawdata_input(city):
    """ Prompt the user if they want to see 5 lines of raw data, display that data if the answer is 'yes', and continue these prompts       and displays until the user says 'no' ."""
    index = 0
    nrows = 5
    while True:
        rawdata_input = input("\nWould you like to see 5 lines of raw data? (yes/no)\n")
        if rawdata_input == 'yes':
            #To display raw data without NaN values, set na_filter as 'False'
            df = pd.read_csv(CITY_DATA[city],na_filter=False)
            print(df.iloc[index*nrows:(index+1)*nrows])
            index+=1
        elif rawdata_input == 'no':
            break


def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)

        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df,city)
        rawdata_input(city)
        restart = input('\nWould you like to restart? (yes/no)\n')
        if restart.lower() != 'yes':
            break


if __name__ == "__main__":
	main()
