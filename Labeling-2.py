
import pandas as pd
import numpy as np 
import numpy as np
import pandas as pd
from typing import List

#Function that returns the number of Trading Days of a Event
def count_trading_days_per_event(
    trading_days: pd.DatetimeIndex,
    event_start: pd.Series | np.ndarray,
    event_end: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Count number of trading days between event_start and event_end for each event.

    trading_days: sorted DatetimeIndex of valid trading days
    event_start, event_end: same length, datetime-like (Series or array)
    """
    # Make sure these are arrays of Timestamps
    start_vals = pd.to_datetime(event_start).to_numpy()
    end_vals   = pd.to_datetime(event_end).to_numpy()

    # Position of first trading day >= start
    start_idx = trading_days.searchsorted(start_vals, side="left")

    # Position of last trading day <= end
    end_idx = trading_days.searchsorted(end_vals, side="right") - 1

    # Clip to valid range in case start/end are outside trading_days
    start_idx = np.clip(start_idx, 0, len(trading_days) - 1)
    end_idx   = np.clip(end_idx,   0, len(trading_days) - 1)

    # Number of trading days (if end before start, set to 0)
    n_days = end_idx - start_idx + 1
    n_days = np.where(n_days < 0, 0, n_days)

    return n_days


def cusum_filter_events_dynamic_threshold(
        prices: pd.Series,
        threshold: pd.Series
) -> pd.DatetimeIndex:
    """
    Detect events using the Symmetric Cumulative Sum (CUSUM) filter.

    The Symmetric CUSUM filter is a change-point detection algorithm used to identify events where the price difference
    exceeds a predefined threshold.

    :param prices: A pandas Series of prices.
    :param threshold: A pandas Series containing the predefined threshold values for event detection.
    :return: A pandas DatetimeIndex containing timestamps of detected events.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: 39)
    """
    time_events, shift_positive, shift_negative = [], 0, 0
    price_delta = prices.diff().dropna()
    thresholds = threshold.copy()
    price_delta, thresholds = price_delta.align(thresholds, join="inner", copy=False)

    for (index, value), threshold_ in zip(price_delta.to_dict().items(), thresholds.to_dict().values()):
        shift_positive = max(0, shift_positive + value)
        shift_negative = min(0, shift_negative + value)

        if shift_negative < -threshold_:
            shift_negative = 0
            time_events.append(index)

        elif shift_positive > threshold_:
            shift_positive = 0
            time_events.append(index)

    return pd.DatetimeIndex(time_events)


def daily_volatility_with_log_returns(
        close: pd.Series,
        span: int = 100
) -> pd.Series:
    """
    Calculate the daily volatility at intraday estimation points using Exponentially Weighted Moving Average (EWMA).

    :param close: A pandas Series of daily close prices.
    :param span: The span parameter for the Exponentially Weighted Moving Average (EWMA).
    :return: A pandas Series containing daily volatilities.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: Page 44)
    """
    df1 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df1 = df1[df1 > 0]
    df = pd.DataFrame(index=close.index[close.shape[0] - df1.shape[0]:])
 
    df['yesterday'] = close.iloc[df1].values
    df['two_days_ago'] = close.iloc[df1 - 1].values
    returns = np.log(df['yesterday'] / df['two_days_ago'])
    
    stds = returns.ewm(span=span).std().rename("std")

    return stds


def vertical_barrier(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    number_days: int
) -> pd.Series:
    """
    Define a vertical barrier.

    :param close: A dataframe of prices and dates.
    :param time_events: A vector of timestamps.
    :param number_days: A number of days for the vertical barrier.
    :return: A pandas series with the timestamps of the vertical barriers.
    """
    timestamp_array = close.index.searchsorted(time_events + pd.Timedelta(days=number_days))
    timestamp_array = timestamp_array[timestamp_array < close.shape[0]]
    timestamp_array = pd.Series(close.index[timestamp_array], index=time_events[:timestamp_array.shape[0]])
    return timestamp_array


def triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    profit_taking_stop_loss: list[float, float],
    molecule: list
) -> pd.DataFrame:
    
    # Filter molecule to ensure all timestamps exist in events
    molecule = [m for m in molecule if m in events.index]

    # Continue filtered Events
    events_filtered = events.loc[molecule]
    output = events_filtered[['End Time']].copy(deep=True)

    # If profit Taking is set compute Level based on volatility scaled Base Width
    if profit_taking_stop_loss[0] > 0:
        profit_taking = profit_taking_stop_loss[0] * events_filtered['Base Width']
    else:
        profit_taking = pd.Series(index=events.index)

    # If Stop Loss is set compute Level based on volatility scaled Base Width
    if profit_taking_stop_loss[1] > 0:
        stop_loss = -profit_taking_stop_loss[1] * events_filtered['Base Width']
    else:
        stop_loss = pd.Series(index=events.index)

    #determine earliest time where stop loss, profit taking or vertical barrier is hit
    for location, timestamp in events_filtered['End Time'].fillna(close.index[-1]).items():
        df = close[location:timestamp] #takes the price path from start to the vertical barrier:
        df = np.log(df / close[location]) * events_filtered.at[location, 'Side']
        output.loc[location, 'stop_loss'] = df[df < stop_loss[location]].index.min()#earliest time where stop loss is hit within vertical time horizon
        output.loc[location, 'profit_taking'] = df[df > profit_taking[location]].index.min()#earliest time where profit taking is hit within vertical time horizon

    return output


def meta_events(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    ptsl: List[float],
    target: pd.Series,
    return_min: float,
    num_threads: int,
    timestamp: pd.Series = False,
    side: pd.Series = None
) -> pd.DataFrame:
    
    # Filter target by time_events and return_min
    target = target.loc[time_events]
    target = target[target > return_min]

    # Ensure timestamp is correctly initialized
    if timestamp is False:
        timestamp = pd.Series(pd.NaT, index=time_events)
    else:
        #set timestamps to events start date.
        timestamp = timestamp.loc[time_events]

    if side is None:
        #if none. side_position is filled entirely with one, so we always go long at every event.
        #both profit and loss barrier is set to the same value.
        side_position, profit_loss = pd.Series(1., index=target.index), [ptsl[0], ptsl[0]]
    else:
        #if side is set then side_position is either 1 for long or -1 for short. 
        #profit and loss barrier is set to the same value.
        side_position, profit_loss = side.loc[target.index], ptsl[:2]

    # Include 'target' and 'timestamp' in the events DataFrame
    events = pd.concat({'End Time': timestamp, 'Base Width': target, 'Side': side_position, 'target': target, 'timestamp': timestamp}, axis=1).dropna(subset=['Base Width'])

    df0 = list(map(
        triple_barrier,
        [close] * num_threads,
        [events] * num_threads,
        [profit_loss] * num_threads,
        np.array_split(time_events, num_threads)
    ))
    df0 = pd.concat(df0, axis=0)

    #set End Time to earliest barrier hit.
    events['End Time'] = df0.dropna(how='all').min(axis=1)

    if side is None:
        events = events.drop('Side', axis=1)

    # Return events including the 'target' and 'timestamp' columns
    return events , df0



def triple_barrier_labeling(
    events: pd.DataFrame,
    close: pd.Series,
    return_min: float | None = None,
    three_class: bool = True,
) -> pd.DataFrame:
    """
    Label events by the sign of the realized return at End Time, with an
    optional threshold on the absolute return.

    Parameters
    ----------
    events : DataFrame
        Must contain at least 'End Time' and 'timestamp' columns, and be
        indexed by the event start time.
    close : Series
        Price series indexed by timestamps.
    return_min : float or None
        If not None, events with |return| < return_min get label 0 (if
        three_class=True) or are left in the DataFrame with Side=0 so you
        can filter them out later.
    three_class : bool
        If True: Side ∈ {-1, 0, 1} (small returns → 0).
        If False: Side ∈ {-1, 1} and you can manually drop small-return
        events when training.

    Returns
    -------
    out : DataFrame
        Index = events_filtered.index
        Columns: ['timestamp', 'End Time', 'Return', 'Side', 'trade_days', 'Daily_Return']
    """
    # Drop events without a valid End Time
    events_filtered = events.dropna(subset=['End Time'])

    # Align close series
    all_dates = events_filtered.index.union(events_filtered['End Time'].values).drop_duplicates()
    close_filtered = close.reindex(all_dates, method='bfill')

    # Start and end prices
    start_prices = close_filtered.loc[events_filtered.index].values
    end_prices = close_filtered.loc[events_filtered['End Time'].values].values

    # Realized return
    ret = end_prices / start_prices - 1

    out = pd.DataFrame(index = events_filtered.index)
    out['timestamp'] = events_filtered.index
    out['End Time'] = events_filtered['End Time']
    out['Return'] = ret

    #compute Number of Trading Days of the Event and Daily Return.
    trading_days = close.index.unique()
    n_days = count_trading_days_per_event(
        trading_days = trading_days,
        event_start  = events_filtered.index,
        event_end    = events_filtered["End Time"],
    )
    out["trade_days"] = n_days
    out["Daily_Return"] = np.where(n_days > 0, ret / n_days, 0.0)

    # Base label: sign of return
    side = np.sign(ret)

    if return_min is not None:
        small = np.abs(ret) < return_min
        if three_class:
            # Small moves -> label 0
            side[small] = 0
        else:
            pass

    out['Side'] = side

    return out


def meta_labeling(
    events: pd.DataFrame,
    close: pd.Series,
    threshold = 0.4
) -> pd.DataFrame:
    """
    Expands label to incorporate meta-labeling.

    :param events: DataFrame with timestamp of vertical barrier and unit width of the horizontal barriers.
    :param close: Series of close prices with date indices.
    :return: DataFrame containing the return and binary labels for each event.

    Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 51
    """
    events_filtered = events.dropna(subset=['End Time'])
    all_dates = events_filtered.index.union(events_filtered['End Time'].values).drop_duplicates()
    close_filtered = close.reindex(all_dates, method='bfill')
    out = pd.DataFrame(index=events_filtered.index)
    
    #time of first barrier hit
    #Return of Label: close price at end time/ close price at start time - 1.
    out['Return of Label'] = close_filtered.loc[events_filtered['End Time'].values].values / close_filtered.loc[events_filtered.index] - 1

    #timestamp contains original vertical barrier dates per event. The equality returns zero if unequal so pt or sl
    # is hit or 1 if vertical barrier is hit. So Label = 1 if a barrier is hit and 0 if vertical barrier is hit.    
    if 'pred_Side' in events_filtered:
        out['Return of Label'] *= events_filtered['pred_Side']
    
    out['Label'] = np.sign(out['Return of Label'])  * (1 - (events['End Time'] == events['timestamp']))
    
    if 'pred_Side' in events_filtered:
        out.loc[out['Return of Label'] <= 0, 'Label'] = 0

    if 'pred_Side_proba' in events_filtered:
        out.loc[events_filtered['pred_Side_proba'] <= threshold, 'Label'] = 0

    return out

