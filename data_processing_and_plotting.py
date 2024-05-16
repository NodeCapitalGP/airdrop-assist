import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from IPython.display import display

# from scipy import stats


# display settings
pd.set_option("display.float_format", "{:,.3}".format)
pd.set_option("display.max_columns", 500)
pd.options.display.float_format = "{:,.3f}".format
pd.set_option("display.max_colwidth", 2000)
pd.set_option("display.width", 1000)

filepaths = {
    "snapshot_2": "path/to/snapshot_2.csv",
    # "users_list": "path/to/users_list.csv",
    "banned_sybil_addresses": "path/to/banned_sybil_addresses.csv",
    "contract_accounts": "path/to/CA_Adresses_Top_10%.csv",
}

# Airdrop Parameters
total_supply = 10**9  # 1 Billion tokens
airdrop_size = 0.05  # 5%
airdrop_allocation = total_supply * airdrop_size

TOKEN_VALUE = 1  # 1 USD

# Eligibility Parameters
top_k_users_eligibility_threshold = 100 * 1000  # Top 50/100 ,000 users are eligible
min_loyalty_points_threshold = 10**2 * 3  # Min loyalty points threshold
max_loyalty_points_threshold = 10**12  # Max loyalty points threshold

# distribution_transformation_factor = 0.8  # Default factor for distribution transformation e.g. x^0.8


def load_and_clean_data(filepath):
    """Load data from a CSV file, suppress DtypeWarning, and clean the data."""
    df = pd.read_csv(filepath, low_memory=False)

    for column in df.columns:
        if column == "timestamp":
            df[column] = pd.to_datetime(df[column]).dt.date
        elif column != "address" and df[column].dtype == "object":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

def filter_sybils(users_df, banned_sybil_addresses_df):
    """Filter out sybil addresses from the users dataframe."""
    return users_df[~users_df["address"].isin(banned_sybil_addresses_df["address"])]

def calculate_total_loyalty_points(df, loyalty_cols):
    """
    Calculates the total loyalty points for a dataframe and adds percentile columns.

    Args:
        df (DataFrame): The DataFrame.
        loyalty_cols (list): A list of column names representing the loyalty points.

    Returns:
        DataFrame: The updated DataFrame with total loyalty points and percentiles calculated.
    """
    percentiles = [0.01, 0.02, 0.03, 0.05, 0.1, 0.9, 0.95, 0.97, 0.98, 0.99]
    df["total_loyalty_points"] = df[loyalty_cols].sum(axis=1)
    for percentile in percentiles:
        df[f"{percentile*100}%_loyalty_points"] = df["total_loyalty_points"].quantile(percentile)
    return df


def select_users_by_loyalty_points(
    df,
    user_limit,
    min_points,
    max_points,
):
    """
    Filters the dataframe to select the top 'user_limit' users based on total loyalty points.
    Only users with a total loyalty points between 'min_points' and 'max_points' are considered.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the user data.
    user_limit (int): The number of top users to select.
    min_points (int): The minimum loyalty points a user must have to be considered.
    max_points (int): The maximum loyalty points a user can have to be considered.

    Returns:
    pandas.DataFrame: A dataframe containing the data for the top 'user_limit' users.
    """
    eligible_df = df[(df["total_loyalty_points"] >= min_points) & (df["total_loyalty_points"] <= max_points)]
    eligible_df = eligible_df.sort_values(by="total_loyalty_points", ascending=False).head(user_limit)
    return eligible_df


def select_top_percentile_users(df, percentile, min_loyalty_points_threshold, max_loyalty_points_threshold):
    """Select the top percentile of users based on total loyalty points."""
    eligible_df = df[
        (df["total_loyalty_points"] >= min_loyalty_points_threshold) & (df["total_loyalty_points"] <= max_loyalty_points_threshold)
    ]
    threshold = eligible_df["total_loyalty_points"].quantile(percentile / 100.0)
    return eligible_df[eligible_df["total_loyalty_points"] >= threshold]


def calculate_loyalty_cols(df):
    """Calculate loyalty_cols based on the given dataframe."""
    # return [col for col in df.columns if "loyalty_points" in col and "eigen_loyalty" not in col]
    return [col for col in df.columns if "_points" in col]


def convert_wei_to_eth(wei):
    """Convert wei to eth."""
    return wei / 10**18


def load_data():
    users_df = load_and_clean_data("path/to/snapshot_2.csv")
    banned_sybil_addresses_df = load_and_clean_data("path/to/banned_sybil_addresses.csv")
    
    
    contract_accounts_df = load_and_clean_data("path/to/CA_Adresses_Top_10%.csv")
    return users_df, banned_sybil_addresses_df, contract_accounts_df


def merge_contract_accounts(users_df, contract_accounts_df):
    """
    Merge contract accounts with user data based on the 'address' column.

    Parameters:
    - users_df (pandas.DataFrame): DataFrame containing user data.
    - contract_accounts_df (pandas.DataFrame): DataFrame containing contract account data.

    Returns:
    - merged_df (pandas.DataFrame): DataFrame with contract accounts merged with user data.
    """
    # Select only the 'address' column from contract_accounts_df
    contract_accounts_df = contract_accounts_df[["address"]]
    return pd.merge(contract_accounts_df, users_df, on="address", how="left")


def filter_ca_addresses(users_df, contract_accounts_df):
    ca_addresses = set(contract_accounts_df["address"])
    return users_df[~users_df["address"].isin(ca_addresses)].copy()


def calculate_and_print_counts(eoa_accounts_df, eoa_accounts_no_sybils_df):
    print(f"EOA Accounts Count: {len(eoa_accounts_df)} \nEOA Accounts without Sybils Count: {len(eoa_accounts_no_sybils_df)}")


def calculate_total_loyalty_for_all(dfs, loyalty_cols):
    """
    Calculates the total loyalty points for all dataframes in the given list or for a single dataframe.

    Args:
        dfs (list or DataFrame): A list of DataFrames or a single DataFrame.
        loyalty_cols (list): A list of column names representing the loyalty points.

    Returns:
        list or DataFrame: The updated list of DataFrames or the updated DataFrame with total loyalty points calculated.
    """
    if isinstance(dfs, list):
        for i in range(len(dfs)):
            dfs[i] = calculate_total_loyalty_points(dfs[i], loyalty_cols)
    else:
        dfs = calculate_total_loyalty_points(dfs, loyalty_cols)
    return dfs


def select_columns_for_comparison(df):
    """Select the first three columns and 'total_loyalty_points' for comparison horizontally."""
    print("in select columns")
    # select columns 1 to 2 and 'total_loyalty_points'
    # df.describe()
    # df.info()
    columns_to_compare = df.columns[1:3].tolist() + ["total_loyalty_points"]
    return columns_to_compare  # return list of column names


# def calculate_stats_for_columns(df,columns):stats=df[columns].describe(percentiles=[.01,.1,.25,.5,.75,.9,.99]).loc[['count','mean','std','min','1%','10%','25%','50%','75%','90%','99%','max']];return stats


def compare_multiple_total_loyalty_points_stats(dfs):
    # """Compare the statistics of all columns in multiple dataframes."""
    # stats = {df_name: df.describe() for df_name, df in dfs.items()}
    # comparison_df = pd.concat(stats, axis=1)
    # comparison_df.index.name = "Loyalty Points"
    # return comparison_df

    """Compare the statistics of the 'total_loyalty_points' column in multiple dataframes."""
    # print ('Entering compare_multiple_total_loyalty_points_stats')
    # print that this is based on loyalty points
    print("\033[1;94mComparing statistics based on loyalty points\033[0m")
    stats = {df_name: df["total_loyalty_points"].describe() for df_name, df in dfs.items()}
    comparison_df = pd.concat(stats, axis=1)

    return comparison_df


def print_descriptive_stats(df1, df2):
    print("-" * 55)
    print("MOOOOOOOOOOOOO" * 55)
    print("eligible_eoa_accounts_df:\n", df1[["total_loyalty_points"]].describe(), "\n")
    print("eligible_eoa_accounts_no_sybils_df:\n", df2[["total_loyalty_points"]].describe())
    print("-" * 55)


def print_top_loyalty_points(df1, df2, num_top_users):
    print(
        f"Sum of the top {num_top_users} 'total_loyalty_points' for 'eligible_eoa_accounts' and 'eligible_eoa_accounts_no_sybils' datasets"
    )
    print("-" * 80)
    print(f"Total Loyalty Points for top {num_top_users} Eligible EOA: {df1['total_loyalty_points'].nlargest(num_top_users).sum():,.0f}")
    print(
        f"Total Loyalty Points for top {num_top_users} Eligible EOA no sybils: {df2['total_loyalty_points'].nlargest(num_top_users).sum():,.0f}"
    )


def calculate_tokens_per_point(total_loyalty_points):
    """Calculate the number of tokens allocated per loyalty point."""
    return airdrop_allocation / total_loyalty_points



def calculate_tokens_per_user(total_loyalty_points, total_eligible_users, is_print=True):
    # Convert total_loyalty_points and total_eligible_users to floats
    total_loyalty_points = float(total_loyalty_points)
    total_eligible_users = float(total_eligible_users)

    # Calculate tokens per user
    tokens_per_user = airdrop_allocation / total_eligible_users

    if is_print:
        print(f"Eligible Accounts: {total_eligible_users:,.0f}")
        print(f"Tokens per User: {tokens_per_user:,.2f}")

    return tokens_per_user


def calculate_totals(df, is_print=True):
    total_eligible_users = len(df)
    total_loyalty_points = df["total_loyalty_points"].sum()

    tokens_per_user = calculate_tokens_per_user(total_loyalty_points, total_eligible_users, is_print)
    tokens_per_point = calculate_tokens_per_point(total_loyalty_points)

    if is_print:
        print(f"Total Eligible Users: {total_eligible_users:,}")
        print(f"Total Loyalty Points: {total_loyalty_points:,.2f}")
        print(f"Tokens per User: {tokens_per_user:,.2f}")
        print(f"Tokens per Point: {tokens_per_point:,.6f}")

    return total_eligible_users, total_loyalty_points, tokens_per_user, tokens_per_point


def calculate_tokens_eligible(df, tokens_per_point):
    """
    Calculate tokens eligible and tokens eligible percentage for each row in the DataFrame.
    """
    df["tokens_eligible"] = df["total_loyalty_points"] * tokens_per_point
    df["tokens_eligible_percentage"] = df["tokens_eligible"] / airdrop_allocation * 100
    df["tokens_eligible_percentage"] = df["tokens_eligible_percentage"].apply(lambda x: "{:.3f}%".format(x))
    return df


def calculate_airdrop_value(df):
    """
    Calculate airdrop value for each row in the DataFrame.
    """
    df["airdrop_value"] = df["tokens_eligible"] * TOKEN_VALUE
    df["airdrop_value"] = df["airdrop_value"].apply(lambda x: "${:,.2f}".format(x))
    return df


def display_top_and_bottom_rows_with_airdrop_info(df, num_rows, df_name):
    """
    Display the top and bottom rows of the DataFrame after adding additional columns for tokens eligible,
    tokens eligible percentage of the whole airdrop allocation, and airdrop value. The number of rows displayed
    from the top and bottom of the DataFrame is determined by the num_rows parameter.
    """
    df_copy = df.copy()
    tokens_per_point_temp = calculate_totals(df, False)[3]

    df_copy = calculate_tokens_eligible(df_copy, tokens_per_point_temp)
    df_copy = calculate_airdrop_value(df_copy)

    # Remove $ sign and convert airdrop_value to float
    df_copy["airdrop_value"] = df_copy["airdrop_value"].replace({"\$": "", ",": ""}, regex=True).astype(float)

    print(f"\033[1m\033[94mTHE TOP AND BOTTOM {num_rows} rows of: {df_name}\033[0m")
    print(f"Tokens per Point: {tokens_per_point_temp:,.6f}")
    print(f"Points per Token: {1/tokens_per_point_temp:,.6f}")
    print(f"Avg Airdrop Value: ${df_copy['airdrop_value'].mean():,.2f}")
    print(f"Median Airdrop Value: ${df_copy['airdrop_value'].median():,.2f}")

    display(
        df_copy[
            [
                "address",
                "total_loyalty_points",
                "tokens_eligible",
                "tokens_eligible_percentage",
                "airdrop_value",
            ]
        ].head(num_rows)
    )
    display(
        df_copy[
            [
                "address",
                "total_loyalty_points",
                "tokens_eligible",
                "tokens_eligible_percentage",
                "airdrop_value",
            ]
        ].tail(num_rows)
    )


def display_bottom_rows_with_airdrop_info(df, num_rows, df_name):
    """
    Display the bottom rows of the DataFrame after adding additional columns for tokens eligible,
    tokens eligible percentage of the whole airdrop allocation, and airdrop value. The number of rows displayed
    from the bottom of the DataFrame is determined by the num_rows parameter.
    """
    print(f"\033[1m\033[94mTHE BOTTOM {num_rows} rows of: {df_name}\033[0m")
    df_copy = df.copy()
    tokens_per_point_temp = calculate_totals(df, False)[3]
    print(f"Tokens per Point: {tokens_per_point_temp:,.6f}")
    print(f"Points per Token: {1/tokens_per_point_temp:,.6f}")

    df_copy = calculate_tokens_eligible(df_copy, tokens_per_point_temp)
    df_copy = calculate_airdrop_value(df_copy)

    # Remove $ sign and convert airdrop_value to float
    df_copy["airdrop_value"] = df_copy["airdrop_value"].replace({"\$": "", ",": ""}, regex=True).astype(float)

    # Print avg and median airdrop value
    print(f"Avg Airdrop Value: ${df_copy['airdrop_value'].mean():,.2f}")
    print(f"Median Airdrop Value: ${df_copy['airdrop_value'].median():,.2f}")

    display(
        df_copy[
            [
                "address",
                "total_loyalty_points",
                "tokens_eligible",
                "tokens_eligible_percentage",
                "airdrop_value",
            ]
        ].tail(num_rows)
    )


def print_and_chart_top_n(df, n, df_name="DataFrame", tokens_per_point=0.0):
    """Prints and charts the top n DataFrame's token_eligible percentage of overall amount."""
    # print the df top 10 rows
    print(f"\033[1m\033[94mTHE TOP {n} rows of: {df_name}\033[0m")
    # print tokens_per_point
    print(f"Tokens per Point: {tokens_per_point:,.6f}")


    # if n is greater than df length, set n to df length
    if n > len(df):
        n = len(df)
    print(f"Top {n} of {df_name}'s token_eligible percentage of overall amount:\n")
    df_copy = df.copy()
    df_copy["tokens_eligible"] = df_copy["total_loyalty_points"] * tokens_per_point
    df_copy["token_eligible_percentage"] = df_copy["tokens_eligible"] / df_copy["tokens_eligible"].sum() * 100
    top_n_df = df_copy.nlargest(n, "tokens_eligible")

    print(
        top_n_df[
            [
                "address",
                "tokens_eligible",
                "token_eligible_percentage",
                "total_loyalty_points",
            ]
        ]
    )

    print("-" * 80)  # Add a divider line
    print(f"Top {n} accounts tokens_eligible sum total: {top_n_df['tokens_eligible'].sum():,.2f}")
    print(f"Total accounts tokens available for airdrop: {airdrop_allocation:,.2f}")
    print(f"Top {n} accounts token_eligible_percentage sum total: {top_n_df['token_eligible_percentage'].sum():,.2f}%")  # Added line

    if n > 300:
        print(f"Can't plot {n} bars, please choose a smaller number to plot (or change the function).\n")
        return

    fig_width = n // 2
    plt.figure(figsize=(fig_width, 6))
    # create a color variable that randomally generates n colors
    colors = np.random.rand(n, 3)
    bars = plt.bar(top_n_df["address"], top_n_df["token_eligible_percentage"], color=colors)
    plt.xlabel("Address")
    plt.ylabel("Token Eligible Percentage")
    plt.title(f"Top {n} {df_name} Token Eligible Percentage")
    plt.xticks(rotation=90)

    for bar in bars:
        yval = bar.get_height()
        text = f"{round(yval, 2)}%"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            text,
            va="bottom",
            ha="center",
            fontsize=7.5,
        )  # va: vertical alignment, ha: horizontal alignment, fontsize: font size

    plt.show()


# print_and_chart_top_n(eligible_eoa_accounts_no_sybils_df, 8, "eligible_eoa_accounts_no_sybils_df")
# print_and_chart_top_n(eligible_eoa_accounts_no_sybils_df, 16, "eligible_eoa_accounts_no_sybils_df")
# print_and_chart_top_n(eligible_eoa_accounts_no_sybils_df, 32, "eligible_eoa_accounts_no_sybils_df")
# print_and_chart_top_n(eligible_eoa_accounts_no_sybils_df, 64, "eligible_eoa_accounts_no_sybils_df")
# # print_and_chart_top_n(eligible_eoa_accounts_no_sybils_df, 128)
# print_and_chart_top_n(eligible_eoa_accounts_no_sybils_df, 256, "eligible_eoa_accounts_no_sybils_df")
# # print_and_chart_top_n(eligible_eoa_accounts_df, 16, "eligible_eoa_accounts_df")
# # print_and_chart_top_n(eligible_eoa_accounts_df, 32, "eligible_eoa_accounts_df")
# # print_and_chart_top_n(eligible_eoa_accounts_df, 128, "eligible_eoa_accounts_df")
# # print_and_chart_top_n(eligible_eoa_accounts_df, 256, "eligible_eoa_accounts_df")
# print_and_chart_top_n(eligible_contract_accounts_df, 8, "eligible_contract_accounts_df")
# print_and_chart_top_n(eligible_users_all_df, 100000, "eligible_users_all_df")

# Data Overview: Use display and print to show an overview of the filtered data.

# print("Current eligible_eoa_accounts_no_sybils overview:")
# display(eligible_eoa_accounts_no_sybils_df.head())
# display(eligible_eoa_accounts_no_sybils_df.describe())
# print(eligible_eoa_accounts_no_sybils_df.info())


def plot_log_histogram(df, data_column_to_plot):
    """
    Plots a histogram of the log-transformed data in the specified column of the DataFrame.

    Parameters:
    df: The pandas DataFrame containing the data to be plotted.
    data_column_to_plot (str): The name of the column in the DataFrame for which the distribution is plotted.

    Returns:
    None
    """
    # Log-transform the data
    log_data = np.log1p(df[data_column_to_plot])

    # Calculate the mean and standard deviation of the log-transformed data
    mean = log_data.mean()
    std_dev = log_data.std()

    # Calculate the variance of the log-transformed data
    variance = std_dev**2

    print(f"Mean of log-transformed data: {mean}, Variance of log-transformed data: {variance}")

    # Plot the histogram with logarithmic bins
    plt.hist(log_data, bins=np.logspace(np.log10(0.1), np.log10(log_data.max() + 1), 50), edgecolor="black")
    plt.xscale("log")

    # Set the labels and title
    plt.xlabel(f"Log-transformed {data_column_to_plot}")
    plt.ylabel("Number of Users")
    plt.title(f"Distribution of Log-transformed {data_column_to_plot}")

    plt.show()


def plot_dynamic_range_barchart(df, data_column_to_plot, num_intervals=11, df_name=None):
    """
    Plots a bar chart with dynamically calculated value ranges for each bar. The dataset is
    segmented into a specified number of intervals (bars), with each bar's height representing
    the count of data points within the calculated range. The range for each bar is determined
    based on the overall distribution of the specified column in the dataset.

    Parameters:
    df: The pandas DataFrame containing the data to be plotted.
    value_column (str): The name of the column in the DataFrame for which the distribution is plotted.
    num_intervals (int): The number of equal-frequency intervals (bars) to divide the data into.
    df_name (str): The name of the DataFrame to be included in the title of the graph.

    Returns:
    matplotlib.axes.Axes: An Axes object with the bar chart plotted on it.
    """

    num_intervals += 1
    font_size = 9

    min_value = df[data_column_to_plot].min()
    max_value = df[data_column_to_plot].max()

    min_order_of_magnitude = np.floor(np.log10(min_value))
    max_order_of_magnitude = np.ceil(np.log10(max_value))
    is_too_many_intervals = False

    while True:
        bin_edges = np.logspace(min_order_of_magnitude, max_order_of_magnitude, num=num_intervals)
        bin_edges = np.sort(bin_edges)
        bin_edges[0] = min_value
        bin_edges[-1] = max_value
        if np.all(np.diff(bin_edges) > 0):
            break
        else:
            num_intervals -= 1
            if max_order_of_magnitude - min_order_of_magnitude > 1:
                min_order_of_magnitude += 1
                max_order_of_magnitude -= 1
            is_too_many_intervals = True

    if is_too_many_intervals:
        print("\033[91m {}\033[00m".format(f"Too many intervals, reducing to {num_intervals-1}"))

    # print(bin_edges)
    counts, _ = np.histogram(df[data_column_to_plot], bins=bin_edges)

    # Generate labels for the ranges
    range_labels = [f"{int(np.round(bin_edges[i]))} - {int(np.round(bin_edges[i+1]))}" for i in range(len(bin_edges) - 1)]
    counts, _ = np.histogram(df[data_column_to_plot], bins=bin_edges)

    last_non_empty_bin = num_intervals - 1

    if counts[-1] == 0:
        last_non_empty_bin = np.max(np.where(counts > 0))

    bin_edges = bin_edges[: last_non_empty_bin + 2]

    plt.figure(figsize=(12, 6))

    colors = np.random.rand(num_intervals, 3)
    bars = plt.bar(range_labels, counts, color=colors, edgecolor="black")

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=font_size)

    plt.xlabel("Range of " + data_column_to_plot, fontsize=font_size)
    plt.ylabel("Number of Users", fontsize=font_size)
    title = f"Distribution of {data_column_to_plot} by Range for: {last_non_empty_bin} bins"
    if df_name:
        title += f" in "
        plt.title(title, fontsize=font_size, color="black")
        plt.title(df_name, fontsize=font_size, color="blue", loc="right")
    else:
        plt.title(title, fontsize=font_size, color="black")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_stats(df, value_column):
    return df[value_column].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])


def compare_dfs_stats_side_by_side_dict(df_dict, columns):
    print("Starting compare_dfs_stats_side_by_side")  # Check if function is entered
    # print data frames' name
    print("Comparing:", list(df_dict.keys()))
    print("cols:", columns)
    for column in columns:
        print(f"\033[1;94mStats for column: {column}\033[0m")
        stats = []
        for name, df in df_dict.items():
            stats.append(get_stats(df, column).rename(name))
        combined_stats = pd.concat(stats, axis=1)
        print(combined_stats)


def compare_dfs_stats_side_by_side(dfs, df_names, columns):
    print("Starting compare_dfs_stats_side_by_side")  # Check if function is entered
    print("Comparing:", df_names)
    assert len(dfs) == len(df_names), "Number of dataframes and dataframe names must match"
    print("cols:", columns)
    for column in columns:
        print(f"\033[1;94mStats for column: {column}\033[0m")
        stats = []
        for df, name in zip(dfs, df_names):
            stats_df = get_stats(df, column).rename(name)
            stats.append(stats_df)
        combined_stats = pd.concat(stats, axis=1)
        print(combined_stats)


# Update the process_dfs function to send both the dataframe and its name to print_and_chart_top_n
def analyze_and_chart_dataframes(df_eligible_dict):
    """
    This function takes a dictionary of dataframes and their names.
    For each dataframe, it prints and charts the top n rows for n in [8, 16, 32].
    If the value in the dictionary is a tuple, it assumes the first element is the dataframe and the second is its name.
    Otherwise, it uses the dictionary key as the name.

    Parameters:
    df_eligible_dict (dict): A dictionary where the keys are names of dataframes and the values are either dataframes or tuples of (dataframe, name).

    Returns:
    None
    """
    for df_name, value in df_eligible_dict.items():
        if isinstance(value, tuple) and len(value) == 2:
            df, name = value
        else:
            df = value
            name = df_name
        tokens_per_point = calculate_tokens_per_point(df["total_loyalty_points"].sum())
        for n in [8, 16, 32]:
            print_and_chart_top_n(df, n, name, tokens_per_point)


def get_dataframe_from_dict(dict, key):
    value = dict[key]
    if isinstance(value, tuple):
        return value[0]  # return the first element of the tuple
    else:
        return value  # return the value as is


def convert_tuples_to_dfs(dict):
    for key, value in dict.items():
        if isinstance(value, tuple):
            dict[key] = value[0]  # replace the tuple with its first element
    return dict


def plot_all_dfs(dict, value_column, bins):
    for key, value in dict.items():
        if isinstance(value, tuple):
            df = value[0]  # get the dataframe from the tuple
        else:
            df = value  # the value is a dataframe
        plot_dynamic_range_barchart(df, value_column, bins, key)


def calculate_users_per_bin(df, num_bins):
    """
    Calculates the total number of users and the number of users per bin.
    """
    total_users = len(df)
    users_per_bin = total_users // num_bins
    print(f"Total users: {total_users}, Users per bin: {users_per_bin}")
    return total_users, users_per_bin


def calculate_bins(df, num_bins, users_per_bin, total_users):
    """
    Calculates the min and max values for each bin.
    """
    bins = [df["total_loyalty_points"].min()]
    for i in range(1, num_bins):
        max_value_index = min(users_per_bin * i, total_users - 1)
        bins.append(df["total_loyalty_points"].iloc[max_value_index])
    bins.append(df["total_loyalty_points"].max())
    print(f"Bins: {bins}")  # Print bins
    return bins


def sort_dataframe(df):
    """
    Sorts the DataFrame by 'total_loyalty_points' and resets the index.
    """
    sorted_df = df.sort_values("total_loyalty_points").reset_index(drop=True)
    return sorted_df


def plot_bins(df, num_bins, users_per_bin, bins):
    """
    Plots each bin individually.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    # ? ax.set_yscale("log")  # Set logarithmic scale for the y-axis
    bin_widths = [(bins[i + 1] - bins[i]) for i in range(len(bins) - 1)]
    bin_edges = bins[:-1]
    for i in range(num_bins):
        color = plt.cm.viridis(i / num_bins)
        if users_per_bin is not None:
            bars = ax.bar(
                f"{(i+1)*users_per_bin}-{(i+2)*users_per_bin}",
                bin_widths[i],
                bottom=bin_edges[i],
                color=color,
                edgecolor="k",
            )
        else:
            bars = ax.bar(
                f"Bin {i+1}",
                bin_widths[i],
                bottom=bin_edges[i],
                color=color,
                edgecolor="k",
            )

        # Add the top value of the bin on top of the bar
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bin_edges[i],
                f"{bin_edges[i] + bin_widths[i]:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,  # Set the fontsize to 7
            )
    print("Plot created")


def plot_uniform_bins_histogram(df, num_bins, ranges=None):
    """
    Main function to plot the histogram.
    """
    if ranges is None:
        ranges = [(df["total_loyalty_points"].min(), df["total_loyalty_points"].max())]

    for range_start, range_end in ranges:
        # Filter the DataFrame for the current range
        filtered_df = df[(df["total_loyalty_points"] >= range_start) & (df["total_loyalty_points"] <= range_end)]
        sorted_df = sort_dataframe(filtered_df)
        print(sorted_df["total_loyalty_points"])  # Print 'total_loyalty_points' column

        total_users, users_per_bin = calculate_users_per_bin(sorted_df, num_bins)
        bins = calculate_bins(sorted_df, num_bins, users_per_bin, total_users)
        plot_bins(sorted_df, num_bins, users_per_bin, bins)

        plt.ylabel("Total Loyalty Points")
        plt.xlabel("User Groups")
        # rephrase the print statement to explain that each bar is the lower and upper values of each bin
        plt.title(f"Bar Chart of Total Loyalty Points with Uniform User Distribution ({range_start:,.1f}-{range_end:,.1f})")
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent clipping of legend
        plt.show()


def plot_quantile_bins_histogram(df, num_bins, ranges=None):
    """
    Main function to plot the histogram using quantiles.
    """
    if ranges is None:
        ranges = [(df["total_loyalty_points"].min(), df["total_loyalty_points"].max())]

    for range_start, range_end in ranges:
        # Filter the DataFrame for the current range
        filtered_df = df[(df["total_loyalty_points"] >= range_start) & (df["total_loyalty_points"] <= range_end)]
        sorted_df = sort_dataframe(filtered_df)

        # Calculate quantiles
        quantiles = sorted_df["total_loyalty_points"].quantile(np.linspace(0, 1, num_bins + 1))

        # Use the quantiles as bin edges
        bins = quantiles.values

        plot_bins(sorted_df, num_bins, users_per_bin=None, bins=bins)

        plt.ylabel("Total Loyalty Points")
        plt.xlabel("User Groups")
        # plt.title(f"Bar Chart of Total Loyalty Points with Uniform User Distribution ({range_start:,}-{range_end:,})")
        plt.title(
            f"Bar Chart of Total Loyalty Points: Each bar represents a range of values (bin) from {range_start:,.0f} to {range_end:,.0f}"
        )
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent clipping of legend
        plt.show()
