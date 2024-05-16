import pandas as pd
import importlib
from data_processing_and_plotting import *

import matplotlib.pyplot as plt
import data_processing_and_plotting  # or whatever your module's name is

importlib.reload(data_processing_and_plotting)

"""
This script performs analysis and visualization on airdrop data.
It loads data from various sources, processes and filters the data,
calculates loyalty points, selects eligible users based on loyalty points,
and generates various statistics and charts.

The main functions in this script include:
- load_data(): Loads the data from different sources.
- merge_contract_accounts(): Merges contract accounts with user data to get loyalty points information.
- filter_ca_addresses(): Filters out contract account addresses to create EOA (Externally Owned Account) accounts DataFrame.
- calculate_loyalty_cols(): Defines loyalty columns based on user data.
- filter_sybils(): Filters out sybil accounts from the EOA accounts DataFrame.
- calculate_total_loyalty_for_all(): Calculates total loyalty points for all account types.
- compare_multiple_total_loyalty_points_stats(): Compares the total loyalty points statistics for different account types.
- generate_filtered_dfs(): Generates filtered DataFrames based on loyalty points thresholds.
- select_users_by_loyalty_points(): Selects eligible users based on loyalty points thresholds.
- calculate_and_print_counts(): Calculates and prints counts of EOA accounts.
- print_descriptive_stats(): Prints descriptive statistics of eligible EOA accounts.
- print_top_loyalty_points(): Prints the top loyalty points of eligible EOA accounts.
- calculate_totals(): Calculates totals of eligible EOA accounts.
- display_top_and_bottom_rows_with_airdrop_info(): Displays the top and bottom rows of the DataFrame with airdrop information.
- display_bottom_rows_with_airdrop_info(): Displays the bottom rows of the DataFrame with airdrop information.
- display_all_airdrop_info(): Displays a summary of airdrop information for different DataFrames.
- plot_log_histogram(): Plots a log histogram of total loyalty points.
- plot_dynamic_range_barchart(): Plots a dynamic range bar chart of total loyalty points.
- compare_dfs_stats_side_by_side(): Compares statistics of multiple DataFrames side by side.
- select_top_percentile_users(): Selects top percentile users based on loyalty points thresholds.
- get_dataframe_from_dict(): Retrieves a DataFrame from a dictionary.
- convert_tuples_to_dfs(): Converts tuples to DataFrames in a dictionary.
- plot_all_dfs(): Plots all DataFrames in a dictionary.
- analyze_and_chart_dataframes(): Analyzes and charts DataFrames in a dictionary.
- compare_dfs_stats_side_by_side_dict(): Compares statistics of multiple DataFrames in a dictionary side by side.
- plot_uniform_bins_histogram(): Plots a histogram with uniform bins.
- plot_quantile_bins_histogram(): Plots a histogram with quantile bins.
"""

# settings and prints start here for now
# Load data
(users_df, banned_sybil_addresses_df, contract_accounts_df) = load_data()

# Merge contract accounts with the users_df to get loyalty points information
contract_accounts_df = merge_contract_accounts(users_df, contract_accounts_df)

# Filter out CA addresses to create EOA accounts DataFrame
eoa_accounts_df = filter_ca_addresses(users_df, contract_accounts_df)

# Define loyalty_cols based on users_df
loyalty_cols = calculate_loyalty_cols(users_df)

# Filter out sybil accounts from the DataFrame of EOA accounts and create a copy of the resulting DataFrame
eoa_accounts_no_sybils_df = filter_sybils(eoa_accounts_df, banned_sybil_addresses_df).copy()
# Calculate the total loyalty points for each account in the DataFrame
eoa_accounts_no_sybils_df.loc[:, "total_loyalty_points"] = eoa_accounts_no_sybils_df[loyalty_cols].sum(axis=1)

# Calculate total loyalty points for ALL account types
# add additional percentiles to the dfs (1%, 2%, 3%, 5%, 10%, 90%, 95%, 97%, 98%, 99%)

dfs = [users_df, contract_accounts_df, eoa_accounts_df, eoa_accounts_no_sybils_df]
(users_df, contract_accounts_df, eoa_accounts_df, eoa_accounts_no_sybils_df) = calculate_total_loyalty_for_all(dfs, loyalty_cols)

def save_top_eoa_accounts_to_csv(df, num, filename):
    """
    Save the top num rows of df sorted by loyalty points to a CSV file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to sort and save.
    num (int): The number of top rows to save.
    filename (str): The name of the CSV file to save to.
    """
    top_df = df.sort_values(by='total_loyalty_points', ascending=False).head(num)
    #filter all columns but the address and total_loyalty_points
    top_df = top_df[['address', 'total_loyalty_points']]
    
    top_df.to_csv(filename, index=False)

# Usage:
save_top_eoa_accounts_to_csv(eoa_accounts_df, 1000, 'top_eoa_accounts.csv')

comparison = compare_multiple_total_loyalty_points_stats(
    {
        "Users": users_df,
        "EOA Accounts": eoa_accounts_df,
        "EOA Accounts No Sybils": eoa_accounts_no_sybils_df,
        "Contract Accounts": contract_accounts_df,
    }
)
display(comparison)

# Select eligible users from each DataFrame

eligible_users_all_df = select_users_by_loyalty_points(
    users_df, top_k_users_eligibility_threshold, min_loyalty_points_threshold, max_loyalty_points_threshold
)
eligible_contract_accounts_df = select_users_by_loyalty_points(
    contract_accounts_df, top_k_users_eligibility_threshold, min_loyalty_points_threshold, max_loyalty_points_threshold
)
eligible_eoa_accounts_df = select_users_by_loyalty_points(
    eoa_accounts_df, top_k_users_eligibility_threshold, min_loyalty_points_threshold, max_loyalty_points_threshold
)
eligible_eoa_accounts_no_sybils_df = select_users_by_loyalty_points(
    eoa_accounts_no_sybils_df, top_k_users_eligibility_threshold, min_loyalty_points_threshold, max_loyalty_points_threshold
)
# Print counts
calculate_and_print_counts(eoa_accounts_df, eoa_accounts_no_sybils_df)

num_top_users = 30
print_descriptive_stats(eligible_eoa_accounts_df, eligible_eoa_accounts_no_sybils_df)
print_top_loyalty_points(eligible_eoa_accounts_df, eligible_eoa_accounts_no_sybils_df, num_top_users)
(total_eligible_users, total_loyalty_points, tokens_per_user, tokens_per_point) = calculate_totals(eligible_eoa_accounts_no_sybils_df, True)

print("tokens_per_point:", tokens_per_point)

# Display the top and bottom n rows of the DataFrame of airdrop_value, tokens_eligible, and tokens_eligible_percentage etc
display_top_and_bottom_rows_with_airdrop_info(eligible_eoa_accounts_no_sybils_df, 5, "eligible_eoa_accounts_no_sybils_df")
display_top_and_bottom_rows_with_airdrop_info(eligible_contract_accounts_df, 5, "eligible_contract_accounts_df")


def generate_filtered_dfs(eoa_accounts_df, loyalty_points_list):
    dfs = {}
    for points in loyalty_points_list:
        filtered_df = eoa_accounts_df[eoa_accounts_df["total_loyalty_points"] >= points]
        dfs[f"EOAs {points}+ points"] = filtered_df
    return dfs

# Usage:
loyalty_points_list = [100, 1000, 3000, 10_000, 20_000, 40_000, 60_000, 70_000]
loyalty_filtered_dfs = generate_filtered_dfs(eoa_accounts_df, loyalty_points_list)
comparison = compare_multiple_total_loyalty_points_stats(loyalty_filtered_dfs)
display(comparison)


def display_filtered_dfs_info(filtered_dfs):
    """
    This function displays the top and bottom rows of each DataFrame in the filtered_dfs dictionary.
    The DataFrames are sorted by 'total_loyalty_points' in descending order.
    For the first DataFrame, both top and bottom rows are displayed.
    For the rest, only the bottom rows are displayed.

    Parameters:
    filtered_dfs (dict): A dictionary where the key is the name of the DataFrame and the value is the DataFrame itself.

    Returns:
    None
    """
    first_df = True
    for key, df in filtered_dfs.items():
        # sort df by total_loyalty_points
        df = df.sort_values(by="total_loyalty_points", ascending=False)
        if first_df:
            display_top_and_bottom_rows_with_airdrop_info(df, 5, key)
            first_df = False
        else:
            display_bottom_rows_with_airdrop_info(df, 5, key)


display_filtered_dfs_info(loyalty_filtered_dfs)

def display_all_airdrop_info(dfs):
    """
    This function displays a summary of airdrop information for different DataFrames.
    It calculates and displays the following information for each DataFrame:
    - Eligible Accounts
    - Tokens per User
    - Tokens per Point
    - Points per Token
    - Min Airdrop Value ($)
    - Avg Airdrop Value ($)
    - Median Airdrop Value ($)
    """

    all_info = []
    for df_name, df in dfs.items():
        df_copy = df.copy()
        tokens_per_point_temp = calculate_totals(df, False)[3]

        df_copy = calculate_tokens_eligible(df_copy, tokens_per_point_temp)
        df_copy = calculate_airdrop_value(df_copy)

        # Remove $ sign and convert airdrop_value to float
        df_copy["airdrop_value"] = df_copy["airdrop_value"].replace({"\$": "", ",": ""}, regex=True).astype(float)

        # Create a temporary DataFrame to hold the values
        temp_df = pd.DataFrame(
            {
                "Eligible Accounts": [f"{len(df):,}"],
                "Tokens per User": [f"{df_copy['tokens_eligible'].mean():,.2f}"],
                "Tokens per Point": [f"{tokens_per_point_temp:,.7f}"],
                "Points per Token": [f"{1/tokens_per_point_temp:,.2f}"],
                "Min Airdrop Value ($)": [f"{df_copy['airdrop_value'].min():,.2f}"],
                "Avg Airdrop Value ($)": [f"{df_copy['airdrop_value'].mean():,.2f}"],
                "Median Airdrop Value ($)": [f"{df_copy['airdrop_value'].median():,.2f}"],
            },
            index=[df_name],
        )

        all_info.append(temp_df)

    # Concatenate all the temporary dataframes into a single dataframe
    all_info_df = pd.concat(all_info)
    display(all_info_df)


display_all_airdrop_info(loyalty_filtered_dfs)

def generate_filtered_dfs_by_percentile(eoa_accounts_df, percentile_list):
    dfs = {}
    for percentile in percentile_list:
        threshold = eoa_accounts_df["total_loyalty_points"].quantile(percentile / 100)
        filtered_df = eoa_accounts_df[eoa_accounts_df["total_loyalty_points"] >= threshold]
        dfs[f"EOAs top {100-percentile}% points"] = filtered_df
    return dfs


# Usage:
percentile_list = [1, 2, 3, 5, 7, 10, 15, 20, 30, 33, 42]
percentile_filtered_dfs = generate_filtered_dfs_by_percentile(eoa_accounts_df, percentile_list)
display_filtered_dfs_info(percentile_filtered_dfs)
display_all_airdrop_info(percentile_filtered_dfs)

plot_log_histogram(eligible_eoa_accounts_no_sybils_df, "total_loyalty_points")

df_dict = {
    "Users": users_df,
    "Contract Accounts": contract_accounts_df,
    "EOA Accounts": eoa_accounts_df,
    "EOA Accounts No Sybils": eoa_accounts_no_sybils_df,
}

df_eligible_dict = {
    "eligible Users": eligible_users_all_df,
    "eligible Contract Accounts": eligible_contract_accounts_df,
    "eligible EOA Accounts": eligible_eoa_accounts_df,
    "eligible EOA Accounts No Sybils": eligible_eoa_accounts_no_sybils_df,
}

plot_dynamic_range_barchart(eligible_eoa_accounts_no_sybils_df, "total_loyalty_points", 12, "eligible_eoa_accounts_no_sybils")

def convert_loyalty_points_to_airdrop_value(loyalty_points, tokens_per_point, TOKEN_VALUE):
    tokens_eligible = loyalty_points * tokens_per_point
    airdrop_value = tokens_eligible * TOKEN_VALUE
    return airdrop_value

def calculate_airdrop_value_float(value):
    # Assuming the input is a string with a dollar sign
    # Remove the dollar sign and convert to float
    # print the type to make sure it is a string
    print(type(value))
    return float(value.replace("$", ""))

def plot_dynamic_range_barchart_airdrop_value(df, data_column_to_plot, num_intervals=11, df_name=None):
    """
    Plots a bar chart with dynamically calculated airdrop value ranges for each bar. The dataset is
    segmented into a specified number of intervals (bars), with each bar's height representing
    the count of data points within the calculated range. The range for each bar is determined
    based on the overall distribution of the specified column in the dataset.

    The function calculates the airdrop value for each bin edge and generates the range labels.
    It then plots a histogram using the range labels for the x-ticks and prints the total user's
    value on top of the bars. The x-tick labels are rotated if necessary for better readability.

    Parameters:
    df (pandas.DataFrame): The pandas DataFrame containing the data to be plotted.
    data_column_to_plot (str): The name of the column in the DataFrame for which the distribution is plotted.
    num_intervals (int, optional): The number of equal-frequency intervals (bars) to divide the data into. Defaults to 11.
    df_name (str, optional): The name of the DataFrame to be included in the title of the graph. Defaults to None.

    Returns:
    None: The function does not return anything. It directly plots the graph using matplotlib.
    """

    num_intervals += 1  # don't know why it plot 1 less TODO: fix
    font_size = 9

    # Calculate the min and max of the data
    min_value = df[data_column_to_plot].min()
    max_value = df[data_column_to_plot].max()

    # Calculate the orders of magnitude for the min and max
    min_order_of_magnitude = np.floor(np.log10(min_value))
    max_order_of_magnitude = np.ceil(np.log10(max_value))

    # Generate the bin edges using logspace
    bin_edges = np.logspace(min_order_of_magnitude, max_order_of_magnitude, num=num_intervals)

    # Sort the bin edges
    bin_edges = np.sort(bin_edges)

    # Adjust the first and last bin edges to match the min and max of the data
    bin_edges[0] = min_value
    bin_edges[-1] = max_value

    # Calculate the number of unique values in the data
    unique_values = df[data_column_to_plot].nunique()

    # Ensure num_intervals does not exceed the number of unique values
    num_intervals = min(num_intervals, unique_values)

    # Calculate the airdrop value for each bin edge
    # airdrop_bin_edges = [convert_loyalty_points_to_airdrop(edge) for edge in bin_edges]
    airdrop_bin_edges = [convert_loyalty_points_to_airdrop_value(edge, tokens_per_point, TOKEN_VALUE) for edge in bin_edges]

    # Generate the range labels
    range_labels = [f"${airdrop_bin_edges[i]:,.2f}-${airdrop_bin_edges[i+1]:,.2f}" for i in range(len(airdrop_bin_edges) - 1)]

    # Make sure we have one more bin edge than label
    assert len(airdrop_bin_edges) == len(range_labels) + 1, "There should be one more bin edge than labels"

    # Calculate the histogram data without plotting it
    counts, _ = np.histogram(df[data_column_to_plot], bins=bin_edges)

    last_non_empty_bin = num_intervals - 1

    # Check if the last bin is empty
    if counts[-1] == 0:
        # Find the last non-empty bin
        last_non_empty_bin = np.max(np.where(counts > 0))

    # Adjust the bin edges to exclude the empty bins
    bin_edges = bin_edges[: last_non_empty_bin + 2]

    # Set the figure size to be twice as wide
    plt.figure(figsize=(12, 6))

    colors = np.random.rand(num_intervals, 3)
    # Plot the histogram using the range_labels for the x-ticks
    bars = plt.bar(range_labels, counts, color=colors, edgecolor="black")

    # Print the total user's value on top of the bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom", fontsize=font_size)

    # Set the labels and title
    plt.xlabel("Range of Airdrop Value (USD)", fontsize=font_size)
    plt.ylabel("Number of Users", fontsize=font_size)
    title = f"Distribution of Airdrop Value by Range for: {last_non_empty_bin} bins"
    if df_name:
        title += f" in "
        plt.title(title, fontsize=font_size, color="black")
        plt.title(df_name, fontsize=font_size, color="blue", loc="right")
    else:
        plt.title(title, fontsize=font_size, color="black")

    # Rotate the x-tick labels if necessary
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust the layout to fit everything
    plt.show()


plot_dynamic_range_barchart_airdrop_value(eligible_eoa_accounts_no_sybils_df, "total_loyalty_points", 12, "eligible_eoa_accounts_no_sybils")

# compare Accounts and EOA no sybils Accounts using compare_dfs_stats

dfs = [users_df, eoa_accounts_no_sybils_df]
df_names = ["Users", "EOA Accounts No Sybils"]
columns = select_columns_for_comparison(eoa_accounts_no_sybils_df)
compare_dfs_stats_side_by_side(dfs, df_names, columns)

# compare Accounts, Contract Accounts, EOA Accounts, and EOA no sybils Accounts using compare_dfs_stats

dfs = [users_df, contract_accounts_df, eoa_accounts_df, eoa_accounts_no_sybils_df]
df_names = ["Users", "Contract Accounts", "EOA Accounts", "EOA Accounts No Sybils"]
compare_dfs_stats_side_by_side(dfs, df_names, columns)

dfs = [eligible_users_all_df, eligible_contract_accounts_df, eligible_eoa_accounts_df, eligible_eoa_accounts_no_sybils_df]
df_names = ["eligible Users", "eligible Contract Accounts", "eligible EOA Accounts", "eligible EOA Accounts No Sybils"]
compare_dfs_stats_side_by_side(dfs, df_names, columns)

dfs = [eligible_eoa_accounts_df, eoa_accounts_df]
df_names = ["eligible EOA Accounts", "EOA Accounts"]
compare_dfs_stats_side_by_side(dfs, df_names, columns)

eligible_eoa_accounts_df = select_top_percentile_users(eoa_accounts_df, 0, min_loyalty_points_threshold, max_loyalty_points_threshold)

percentiles = [1, 3, 5, 7, 10, 15, 20]

# Update the dictionary with the dataframe and its name

for percentile in percentiles:
    df_eligible_dict.update(
        {
            f"eligible EOA Accounts {percentile}%": (
                select_top_percentile_users(eoa_accounts_df, percentile, min_loyalty_points_threshold, max_loyalty_points_threshold),
                f"eligible EOA Accounts {percentile}%",
            )
        }
    )
print("Top 1% of EOA Accounts:", df_eligible_dict["eligible EOA Accounts 1%"])

plot_dynamic_range_barchart(df_eligible_dict["eligible EOA Accounts"], "total_loyalty_points", 7, "eligible EOA Accounts")
# plot_dynamic_range_barchart(df_eligible_dict["eligible EOA Accounts"], "total_loyalty_points", 10, "eligible EOA Accounts")
# plot_dynamic_range_barchart(df_eligible_dict["eligible EOA Accounts"], "total_loyalty_points", 12, "eligible EOA Accounts")

df = get_dataframe_from_dict(df_eligible_dict, "eligible EOA Accounts 10%")
plot_dynamic_range_barchart(df, "total_loyalty_points", 7, "eligible EOA Accounts 10%")


df_eligible_dict = convert_tuples_to_dfs(df_eligible_dict)
plot_dynamic_range_barchart(df_eligible_dict["eligible EOA Accounts 10%"], "total_loyalty_points", 7)
# Call the function with your dictionary, value_column, and bins

# plot_all_dfs(df_eligible_dict, "total_loyalty_points", 12)
# now plot all for percentile filtered dfs
plot_all_dfs(percentile_filtered_dfs, "total_loyalty_points", 10)
# now plot all for loyalty filtered dfs
plot_all_dfs(loyalty_filtered_dfs, "total_loyalty_points", 7)

# analyze_and_chart_dataframes(df_eligible_dict)
# analyze_and_chart_dataframes(loyalty_filtered_dfs )
subgroup_dfs = {k: loyalty_filtered_dfs[k] for k in ['EOAs 10000+ points', 'EOAs 40000+ points', 'EOAs 50000+ points']}
analyze_and_chart_dataframes(subgroup_dfs)

dfs = [eligible_eoa_accounts_df, eoa_accounts_df]
df_names = ["eligible EOA Accounts", "EOA Accounts"]
compare_dfs_stats_side_by_side(dfs, df_names, columns)
compare_dfs_stats_side_by_side_dict(df_eligible_dict, columns)

# plot_uniform_bins_histogram(eligible_eoa_accounts_no_sybils_df, num_bins=10)

# ranges = [(3000, 100000), (100000, 1000000), (1000000, eligible_eoa_accounts_no_sybils_df["total_loyalty_points"].max())]
# plot_uniform_bins_histogram(eligible_eoa_accounts_no_sybils_df, num_bins=10, ranges=ranges)

# plot_dynamic_value_range_barchart_airdrop(eligible_eoa_accounts_no_sybils_df, "total_loyalty_points", 10, "eligible_eoa_accounts_no_sybils")
# plot_dynamic_value_range_barchart_airdrop(eligible_eoa_accounts_no_sybils_df, "total_loyalty_points", 7, "eligible_eoa_accounts_no_sybils")
plot_dynamic_range_barchart_airdrop_value(percentile_filtered_dfs["EOAs top 80% points"], "total_loyalty_points", 7, "EOAs top 80% points")
# and now for loyalty filtered dfs for 10000
plot_dynamic_range_barchart_airdrop_value(loyalty_filtered_dfs["EOAs 3000+ points"], "total_loyalty_points", 12, "EOAs 3000+ points")
plot_dynamic_range_barchart_airdrop_value(loyalty_filtered_dfs["EOAs 10000+ points"], "total_loyalty_points", 7, "EOAs 10000+ points")
plot_dynamic_range_barchart_airdrop_value(loyalty_filtered_dfs["EOAs 20000+ points"], "total_loyalty_points", 7, "EOAs 20000+ points")


# now create a function that uses plot_dynamic_range_barchart_airdrop_value for all dfs in a dictionary
def plot_all_dfs_airdrop_value(dfs, data_column_to_plot, num_intervals=11):
    for df_name, df in dfs.items():
        plot_dynamic_range_barchart_airdrop_value(df, data_column_to_plot, num_intervals, df_name)


# Usage:
plot_all_dfs_airdrop_value(loyalty_filtered_dfs, "total_loyalty_points", 7)


