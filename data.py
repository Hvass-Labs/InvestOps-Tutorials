###############################################################################
#
# Functions and classes for loading and processing data.
#
###############################################################################
#
# This file is part of InvestOps Tutorials:
#
# https://github.com/Hvass-Labs/InvestOps-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2022 by Magnus Erik Hvass Pedersen
#
###############################################################################

import pandas as pd
import os
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from investops.rel_change import rel_change, mean_rel_change
from investops.stock_forecast import StockForecast

###############################################################################

# Directory for the data-files on disk.
_data_dir = 'data/'


def set_data_dir(data_dir):
    """Set the data-directory where the files will be loaded from."""
    global _data_dir
    _data_dir = data_dir

###############################################################################
# Names for the data-columns.

# Date / timestamp.
DATE = 'Date'

# Closing share-price only adjusted for stock-splits.
CLOSE = 'Close'

# Closing share-price adjusted for both stock-splits and dividends.
ADJ_CLOSE = 'Adj Close'

# Another name for closing share-price adjusted only for stock-splits.
SHARE_PRICE = 'Share-Price'

# Total Return which is the share-price with reinvestment of dividends.
TOTAL_RETURN = 'Total Return'

# Dividends per share.
DIVIDENDS = 'Dividends'

# Sales Per Share.
SALES_PER_SHARE = 'Sales Per Share'

# Annual growth in Sales Per Share.
SALES_GROWTH = 'Sales Growth (Per Share)'

# Annual growth in Earnings Per Share.
EARNINGS_GROWTH = 'Earnings Growth (Per Share)'

# Dividend Yield = Dividend Per Share / Share-Price
DIV_YIELD = 'Dividend Yield'

# P/Sales Ratio = Share-Price / Sales Per Share
PSALES = 'P/Sales'

# P/E Ratio = Share-Price / Earnings Per Share
PE = 'P/E'

# Mean annualized return.
MEAN_ANN_RETURN = 'Mean Ann. Return'

###############################################################################

class StockData:
    """
    Load share-prices and financial data for a stock, and plot the historical
    data and use it in a model for long-term stock forecasting.

    Please inspect the supplied data-files to see the required file-formats.

    - The CSV-files with share-prices and dividends can be downloaded for free
      from the Yahoo Finance web-site.

    - The CSV-files with Sales Per Share data have been gathered manually by
      the author. NOTE: It is important that this only has annual data-points!

    NOTE: Most methods in this class have their results cached, so they can
    return the same results instantly instead of e.g. having to reload the
    data-files from disk every time the function is called. But this also
    means that you must NOT modify the data that is being returned by these
    class-methods, as that will also change the data being held in the cache,
    and therefore corrupt any calculations using that data.
    """
    def __init__(self, ticker):
        """
        :param ticker: String with the stock-ticker name.
        """
        # Copy args to self.
        self._ticker = ticker

    @staticmethod
    def _read_csv(filename):
        """
        Helper-function for reading a CSV-file in the correct format.

        :param filename: String with the filename.
        :return: Pandas Series or DataFrame.
        """
        path = os.path.join(_data_dir, filename)
        return pd.read_csv(path, index_col=DATE,
                           parse_dates=True, dayfirst=False, squeeze=True)

    @lru_cache
    def all_prices(self):
        """
        Get all the share-prices for this stock.

        :return: Pandas DataFrame.
        """
        filename = f'{self._ticker} Share-Price (Yahoo).csv'
        return self._read_csv(filename=filename)

    @lru_cache
    def prices(self):
        """
        Get the closing share-prices for this stock.

        :return: Pandas Series.
        """
        return self.all_prices()[CLOSE].rename(SHARE_PRICE)

    @lru_cache
    def total_return(self, normalize=True):
        """
        Get the Total Return for this stock, which is the share-price with
        dividends assumed to having been reinvested immediately.

        :param normalize: Boolean whether to make the data start at 1.
        :return: Pandas Series.
        """
        # Get the Total Return data, which is the ADJ_CLOSE column from Yahoo.
        tot_ret = self.all_prices()[ADJ_CLOSE].rename(TOTAL_RETURN)

        # Normalize to begin the data at 1?
        if normalize:
            tot_ret = tot_ret / tot_ret.iloc[0]

        return tot_ret

    @lru_cache
    def mean_ann_return(self, min_years, max_years,
                        start_date=None, end_date=None, future=True):
        """
        Mean annualized return for a range of investment periods between the
        given `min_years` and `max_years`. This is calculated using the Total
        Return data, so it takes reinvestment of dividends into account.

        :param min_years: Int with the min number of investment years.
        :param max_years: Int with the max number of investment years.
        :param start_date: Only use the Total Return data from this date.
        :param end_date: Only use the Total Return data until this date.
        :param future:
            Boolean whether to calculate future (True) or past (False) returns.

        :return: Pandas Series.
        """
        # Get the Total Return for this stock.
        tot_ret = self.total_return(normalize=False)

        # Only use the desired date-range. An index-value of None is ignored.
        tot_ret = tot_ret[start_date:end_date]

        # Calculate the mean and std.dev. for the annualized returns of all
        # investment periods ranging between the given min_years and max_years.
        mean, std = mean_rel_change(df=tot_ret, freq='b', future=future,
                                    min_years=min_years, max_years=max_years,
                                    new_names_mean=MEAN_ANN_RETURN,
                                    annualized=True)

        return mean, std

    @lru_cache
    def sales_per_share(self):
        """
        Get the Sales Per Share data for this stock.

        :return: Pandas Series.
        """
        filename = f'{self._ticker} Sales Per Share.csv'
        return self._read_csv(filename=filename)

    @lru_cache
    def earnings_per_share(self):
        """
        Get the Earnings Per Share data for this stock.

        :return: Pandas Series.
        """
        filename = f'{self._ticker} Earnings Per Share.csv'
        return self._read_csv(filename=filename)

    @lru_cache
    def dividend_per_share(self):
        """
        Get the Dividend Per Share data for this stock.

        :return: Pandas Series.
        """
        filename = f'{self._ticker} Dividend Per Share.csv'
        return self._read_csv(filename=filename)

    @lru_cache
    def dividend_yield(self):
        """
        Calculate the Dividend Yield = Dividend Per Share TTM / Share-Price.

        :return: Pandas Series.
        """
        # Get the Dividend Per Share data.
        dividends = self.dividend_per_share()

        # Get the Share-Prices.
        prices = self.prices()

        # Estimate the Dividend Per Share TTM (Trailing-Twelve-Months).
        # Note: We take the sum of a rolling window of e.g. 320 days which
        # has to be less than a full year of 365 days, so we don't double-
        # count the dividend payouts. This also has some flexibility for
        # changes in the schedule of dividend payouts.
        dividends_ttm = dividends.dropna().rolling('320d').sum()

        # Up-sample to daily data and forward-fill missing values.
        # The forward-fill limit is in case the company stops paying dividend.
        dividends_ttm_daily = dividends_ttm.resample('D').ffill(limit=365)

        # Ensure the dividend data and share-prices have the same index.
        dividends_ttm_daily = dividends_ttm_daily.reindex(prices.index)

        # Calculate the Dividend Yield.
        dividend_yield = dividends_ttm_daily / prices

        # Rename the data.
        dividend_yield.rename(DIV_YIELD, inplace=True)

        return dividend_yield

    @lru_cache
    def val_ratio(self, kind, interpolate=True):
        """
        Calculate a valuation ratio such as P/Sales or P/E.

        :param kind:
            String with the name of the valuation ratio: 'P/Sales' or 'P/E'.

        :param interpolate:
            Boolean whether to interpolate (True) the data, which is a form of
            cheating because future values are being used in the calculation.
            Or use forward-fill of the data (False) which only uses past data,
            but this may lead to a visible step-pattern in the resulting data.

        :return:
            Pandas Series.
        """
        # Get the divisor for use in the valuation ratios.
        if kind == PSALES:
            # The divisor is the Sales Per Share.
            divisor = self.sales_per_share()
        elif kind == PE:
            # The divisor is the Earnings Per Share.
            divisor = self.earnings_per_share()
        else:
            # Error.
            msg = 'Argument \'kind\' should be either \'P/Sales\' or \'P/E\'.'
            raise ValueError(msg)

        # Get the share-prices.
        prices = self.prices()

        # Up-sample the divisor data to daily data-points.
        divisor = divisor.resample('D')

        # Fill in the missing values in the divisor data.
        if interpolate:
            # Interpolate the divisor data to get more smooth changes.
            # This uses both the past and future value, so it is "cheating".
            divisor = divisor.interpolate()
        else:
            # Forward-fill the divisor data to only use the last-known values.
            divisor = divisor.ffill()

        # Ensure the divisor data and share-prices have the same index.
        divisor = divisor.reindex(prices.index)

        # Calculate the valuation ratio.
        val_ratio = prices / divisor

        # Rename the data.
        val_ratio.rename(kind, inplace=True)

        return val_ratio

    def pe(self, interpolate=True):
        """
        Calculate the P/E or Price-To-Earnings ratio for this stock.

        :param interpolate:
            Boolean whether to interpolate (True) or forward-fill (False) data.

        :return: Pandas Series.
        """
        return self.val_ratio(kind=PE, interpolate=interpolate)

    def psales(self, interpolate=True):
        """
        Calculate the P/Sales or Price-To-Sales ratio for this stock.

        :param interpolate:
            Boolean whether to interpolate (True) or forward-fill (False) data.

        :return: Pandas Series.
        """
        return self.val_ratio(kind=PSALES, interpolate=interpolate)

    @lru_cache
    def growth_sales_per_share(self, future=True):
        """
        Calculate the annual growth in Sales Per Share for this stock.

        NOTE: The Sales Per Share data is assumed to have ANNUAL data-points!

        :param future:
            Boolean whether to calculate future (True) or past (False) growth.

        :return: Pandas Series.
        """
        # Get the Sales Per Share data.
        sales_per_share = self.sales_per_share()

        # Calculate and return the annual growth-rates.
        return rel_change(df=sales_per_share, future=future,
                          freq='y', years=1, new_names=SALES_GROWTH)

    @lru_cache
    def growth_earnings_per_share(self, future=True):
        """
        Calculate the annual growth in Earnings Per Share for this stock.

        NOTE: The Earnings Per Share data is assumed to have ANNUAL data-points!

        :param future:
            Boolean whether to calculate future (True) or past (False) growth.

        :return: Pandas Series.
        """
        # Get the Earnings Per Share data.
        earnings_per_share = self.earnings_per_share()

        # Calculate and return the annual growth-rates.
        return rel_change(df=earnings_per_share, future=future,
                          freq='y', years=1, new_names=EARNINGS_GROWTH)

    @lru_cache
    def common_date_range(self, start_date=None, end_date=None):
        """
        Get the common date-range between the Sales Per Share and Share-Price
        data for this stock. This is useful for ensuring we are plotting data
        from the same period, and for using the correct data in the forecasting
        model.

        :param start_date: Optional start-date to further limit the date-range.
        :param end_date: Optional end-date to further limit the date-range.
        :return:
            - min_date: Minimum common date.
            - max_date: Maximum common date.
        """
        # Get the P/Sales data, optionally limited to the given date-range,
        # and with all NaN-values dropped. We only need to use this data,
        # because this will be NaN-values if either the Share-Price or the
        # Sales Per Share values are missing.
        psales = self.psales(interpolate=True)[start_date:end_date].dropna()

        # Get the min and max dates.
        min_date, max_date = psales.index[[0, -1]]

        return min_date, max_date

    def plot_forecast(self, min_years, max_years, rng,
                      start_date=None, end_date=None, cur_val_ratio=None):
        """
        Fit a forecasting model to the historical data for this stock,
        and plot both the forecasting model and the historical valuation
        ratios and stock-returns.

        Instead of using a fixed investment period of e.g. 5 years, a range of
        investment periods should be used between e.g. 4-6 years, as this gives
        smoother plots for the historical annualized stock-returns.

        This uses the P/Sales ratio and growth in Sales Per Share, which
        are much more stable than P/E and Earnings Per Share, and therefore
        give a much better forecasting model and smoother plots.

        Also note that the P/Sales ratio is calculated using interpolated
        Sales Per Share data, which gives smoother plots but is "cheating"
        as it uses future values to calculate intermediate data-points.

        :param min_years:
            Integer with the minimum years when calculating Mean Ann. Returns.

        :param max_years:
            Integer with the maximum years when calculating Mean Ann. Returns.

        :param rng:
            `Numpy.random.Generator` object from `np.random.default_rng()`

        :param start_date:
            Optional string to limit the start-point of the data.

        :param end_date:
            Optional string to limit the end-point of the data.

        :param cur_val_ratio:
            Optional valuation ratio to show as a vertical line in the plot.

        :return:
            - Matplotlib Figure object.
            - InvestOps StockForecast object.
        """
        # Get the common date-range for Share-Price and Sales Per Share data.
        min_date, max_date = \
            self.common_date_range(start_date=start_date, end_date=end_date)

        # Get the historical P/Sales ratios.
        # NOTE: This uses "cheating" interpolation to get smoother plots.
        psales = self.psales(interpolate=True)[min_date:max_date]

        # Get the historical annual growth-rates for Sales Per Share.
        growth = self.growth_sales_per_share()[min_date:max_date]

        # Get the historical Dividend Yields.
        div_yield = self.dividend_yield()[min_date:max_date]

        # Calculate the FUTURE Mean Annualized Returns.
        mean_ann_rets, _ = \
            self.mean_ann_return(min_years=min_years,
                                 max_years=max_years, future=True)

        # Average number of years for the Mean Ann. Returns of historical data.
        avg_years = (max_years + min_years) / 2

        # Fit a forecasting model with the historical data.
        model = StockForecast(div_yield=div_yield, val_ratio=psales,
                              growth=growth, dependent=False,
                              years=avg_years, rng=rng)

        # Create a standardized title for the plot.
        title = model.make_title(ticker=self._ticker,
                                 min_years=min_years, max_years=max_years,
                                 start_year=min_date.year,
                                 end_year=max_date.year)

        # Plot the forecasting model overlaid with the historical data for the
        # P/Sales ratios and the future Mean Annualized Returns.
        fig = model.plot(title=title, cur_val_ratio=cur_val_ratio,
                         hist_val_ratios=psales,
                         hist_ann_rets=mean_ann_rets,
                         name_val_ratio=PSALES)

        return fig, model

    def plot_basic_data(self, log_shareprice=True, figsize=(10, 12.5)):
        """
        Create a plot with the basic financial data for this stock.

        :param log_shareprice:
            Boolean whether to use log-scale on y-axis for share-prices.

        :param figsize:
            Tuple with the figure-size.

        :return:
            Matplotlib Figure object.
        """

        # Helper-function for making a sub-plot.
        def plot_with_mean_line(column, ax, percentage=False, y_decimals=None):
            # Make a line-plot with the data.
            sns.lineplot(x=DATE, y=column, data=df, ax=ax, label=column)

            # Mean of the given data-column.
            mean = df[column].mean()

            # Create label for the mean.
            if percentage:
                # Convert y-ticks to percentages.
                formatter = PercentFormatter(xmax=1.0, decimals=y_decimals)
                ax.yaxis.set_major_formatter(formatter=formatter)

                label_mean = 'Mean = {:.1%}'.format(mean)
            else:
                label_mean = 'Mean = {:.1f}'.format(mean)

            # Plot the mean of the given data-column.
            ax.axhline(mean, c='k', ls=':', label=label_mean)

            # Show the legend.
            ax.legend()

        # Get the common date-range for Share-Price and Sales Per Share data.
        min_date, max_date = self.common_date_range()

        # Combine all the data-columns into a single Pandas DataFrame.
        data = \
            {
                SHARE_PRICE: self.prices()[min_date:max_date],
                SALES_PER_SHARE: self.sales_per_share()[min_date:max_date],
                PSALES: self.psales(interpolate=True)[min_date:max_date],
                SALES_GROWTH: self.growth_sales_per_share()[min_date:max_date],
                DIV_YIELD: self.dividend_yield()[min_date:max_date],
            }
        df = pd.DataFrame(data)

        # Create a new plot with rows for sub-plots.
        plt.rc('figure', figsize=figsize)
        fig, axs = plt.subplots(nrows=4)

        # Set the main plot-title.
        axs[0].set_title(self._ticker)

        # Plot the Share-Price and Sales Per Share.
        sns.lineplot(data=df[[SHARE_PRICE, SALES_PER_SHARE]], ax=axs[0])

        # Use log-scale on the y-axis for the Share-Prices?
        # This MUST be called AFTER sns.lineplot() otherwise result is wrong!
        if log_shareprice:
            axs[0].set_yscale('log')

        # Plot the P/Sales.
        plot_with_mean_line(column=PSALES, ax=axs[1])

        # Plot the Sales Growth.
        plot_with_mean_line(column=SALES_GROWTH, ax=axs[2],
                            percentage=True, y_decimals=0)

        # Plot the Dividend Yield.
        plot_with_mean_line(column=DIV_YIELD, ax=axs[3],
                            percentage=True, y_decimals=1)

        # Adjust all the sub-plots.
        for ax in axs:
            # Don't show the x-axis label.
            ax.set_xlabel(None)

            # Show grid.
            ax.grid(which='both')

        return fig

###############################################################################
